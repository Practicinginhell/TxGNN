"""Config-driven experiment sweeps for TxGNN.

Give it a sweep config and it expands variants x seeds into individual runs,
executes each one in its own process, collects the metrics, and prints a
comparison table with mean +/- std across seeds.

    python -m experiments.run_sweep experiments/configs/init_scheme_mini.json
    python -m experiments.run_sweep <config> --dry-run       # list runs, run nothing
    python -m experiments.run_sweep <config> --only kaiming  # subset of variants
    python -m experiments.run_sweep <config> --aggregate     # re-table existing results
    python -m experiments.run_sweep <config> --set finetune.n_epoch=5

Runs are resumable: a run whose result JSON already exists is skipped, so an
interrupted sweep continues where it stopped. Pass --force to redo them.

Sweep config (JSON):

    {
      "name": "init_schemes",
      "results_dir": "experiments/results/init_schemes",
      "seeds": [1, 2, 3],
      "metric": "test_macro_auroc",     # what the table ranks by
      "baseline": "xavier_uniform",     # variant the deltas are measured against
      "base": { ... run config, see experiments/run_single.py ... },
      "variants": [
        {"name": "xavier_uniform", "overrides": {"model.init_scheme": null}},
        {"name": "kaiming_uniform", "overrides": {"model.init_scheme": "kaiming_uniform"}}
      ],
      "grid": {"model.init_scheme": ["orthogonal", "xavier_normal"]}
    }

`variants` is an explicit list, `grid` is a cartesian product over dotted keys,
and both may be present. Any dotted key that exists in a run config works, so
the same machinery sweeps learning rates, hidden sizes or anything else, not
just initialization.
"""

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

RUN_SINGLE = os.path.join(REPO, 'experiments', 'run_single.py')
DEFAULT_METRIC = 'test_macro_auroc'

# shown in the table, in this order, when present in the results
REPORT_METRICS = ('test_macro_auroc', 'test_macro_auprc',
                  'test_micro_auroc', 'test_micro_auprc',
                  'best_valid_macro_auroc', 'test_loss')


# ---------------------------------------------------------------- config load

def load_config(path):
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            try:
                import yaml
            except ImportError:
                raise SystemExit('PyYAML is not installed, use a .json sweep config '
                                 'or pip install pyyaml')
            return yaml.safe_load(f)
        return json.load(f)


def set_nested(config, dotted_key, value):
    """set_nested(cfg, 'model.init_scheme', 'orthogonal') -> cfg['model']['init_scheme']."""
    parts = dotted_key.split('.')
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return config


def parse_set_arg(arg):
    """--set finetune.n_epoch=5 -> ('finetune.n_epoch', 5)."""
    if '=' not in arg:
        raise SystemExit('--set expects key=value, got %r' % arg)
    key, raw = arg.split('=', 1)
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw  # bare strings do not need quoting
    return key, value


def slug(text):
    text = re.sub(r'[^0-9a-zA-Z._-]+', '_', str(text)).strip('_')
    return text or 'x'


def config_fingerprint(run_config):
    """Stable hash of a run config, ignoring key order.

    Used to notice that a finished run's config no longer matches the sweep, so
    a resumed sweep reruns it instead of reporting stale numbers under the new
    settings.
    """
    payload = json.dumps(run_config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def describe_value(value):
    if isinstance(value, dict):
        name = value.get('name')
        rest = {k: v for k, v in value.items() if k != 'name'}
        if name and rest:
            return '%s_%s' % (name, '_'.join('%s%s' % kv for kv in sorted(rest.items())))
        if name:
            return str(name)
    if value is None:
        return 'default'
    return str(value)


# ------------------------------------------------------------------- expansion

def expand_variants(config):
    """Return [{'name': ..., 'overrides': {...}}] from `variants` and `grid`."""
    variants = []

    for entry in config.get('variants') or []:
        if not isinstance(entry, dict):
            raise SystemExit('each variant must be an object, got %r' % (entry,))
        overrides = entry.get('overrides', {})
        name = entry.get('name') or '_'.join(
            '%s=%s' % (k.split('.')[-1], describe_value(v))
            for k, v in sorted(overrides.items())) or 'base'
        variants.append({'name': name, 'overrides': dict(overrides)})

    grid = config.get('grid') or {}
    if grid:
        keys = sorted(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            overrides = dict(zip(keys, combo))
            name = '_'.join('%s=%s' % (k.split('.')[-1], describe_value(v))
                            for k, v in sorted(overrides.items()))
            variants.append({'name': name, 'overrides': overrides})

    if not variants:
        variants = [{'name': 'base', 'overrides': {}}]

    seen = set()
    for variant in variants:
        if variant['name'] in seen:
            raise SystemExit('duplicate variant name %r, give the variants '
                             'distinct "name" fields' % variant['name'])
        seen.add(variant['name'])
    return variants


def validate_init_schemes(runs):
    """Reject bad init specs before any data is loaded.

    Without this the error surfaces inside the run, which on the full KG means
    waiting through several minutes of graph construction to learn about a typo.
    """
    from txgnn.init_schemes import resolve_init_spec

    problems = []
    for variant in sorted({r['variant'] for r in runs}):
        run = next(r for r in runs if r['variant'] == variant)
        try:
            resolve_init_spec((run.get('model') or {}).get('init_scheme'))
        except (ValueError, TypeError) as exc:
            problems.append('%s: %s' % (variant, exc))
    if problems:
        raise SystemExit('invalid init_scheme in %d variant(s):\n  %s'
                         % (len(problems), '\n  '.join(problems)))


def build_runs(config):
    """Expand the sweep into concrete run configs."""
    base = copy.deepcopy(config.get('base') or {})
    seeds = config.get('seeds') or [base.get('seed', 42)]
    runs = []
    used_slugs = {}

    for variant in expand_variants(config):
        # distinct variant names can slugify to the same string ('a/b' and
        # 'a_b'), which would make two runs share one result file
        variant_slug = slug(variant['name'])
        if used_slugs.setdefault(variant_slug, variant['name']) != variant['name']:
            suffix = 2
            while '%s_%d' % (variant_slug, suffix) in used_slugs:
                suffix += 1
            variant_slug = '%s_%d' % (variant_slug, suffix)
            used_slugs[variant_slug] = variant['name']

        for seed in seeds:
            run_config = copy.deepcopy(base)
            for key, value in variant['overrides'].items():
                set_nested(run_config, key, value)
            run_config['seed'] = seed
            run_config['variant'] = variant['name']
            run_config['run_id'] = '%s__seed%s' % (variant_slug, seed)
            # otherwise every run in the sweep writes the same checkpoint
            if run_config.get('save_dir'):
                run_config['save_dir'] = os.path.join(run_config['save_dir'],
                                                      run_config['run_id'])
            runs.append(run_config)
    return runs


# ------------------------------------------------------------------- execution

def execute(run_config, results_dir, inline=False, quiet=False):
    """Run one config, returning its result dict."""
    runs_dir = os.path.join(results_dir, 'runs')
    logs_dir = os.path.join(results_dir, 'logs')
    for d in (runs_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    run_id = run_config['run_id']
    config_path = os.path.join(runs_dir, run_id + '.json')
    result_path = os.path.join(runs_dir, run_id + '.result.json')
    log_path = os.path.join(logs_dir, run_id + '.log')

    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)

    if inline:
        # same process: easier to debug, but one crash takes the sweep with it
        from experiments.run_single import main as run_single_main
        run_single_main(['--run-config', config_path, '--out', result_path])
        where = result_path
    else:
        cmd = [sys.executable, '-u', RUN_SINGLE,
               '--run-config', config_path, '--out', result_path]
        with open(log_path, 'w') as log:
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, universal_newlines=True)
            for line in proc.stdout:
                log.write(line)
                if not quiet:
                    sys.stdout.write(line)
            proc.wait()
        where = log_path

    result = read_result(result_path)
    if result is None:
        # killed by the OS, or died before it could write anything
        result = {'run_id': run_id, 'variant': run_config.get('variant'),
                  'seed': run_config.get('seed'), 'status': 'failed',
                  'error': 'run produced no result file, see %s' % where,
                  'config': run_config}
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    if not inline:
        result['log'] = log_path
    return result


def read_result(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        try:
            return json.load(f)
        except ValueError:
            return None


# ----------------------------------------------------------------- aggregation

def collect_results(runs, results_dir):
    out = []
    for run_config in runs:
        result = read_result(os.path.join(results_dir, 'runs',
                                          run_config['run_id'] + '.result.json'))
        if result is not None:
            out.append(result)
    return out


def mean_std(values):
    """Mean and sample standard deviation. std is NaN below two samples.

    NaN rather than 0.0 on purpose: a single run has unknown spread, and
    printing '+/- 0.0000' would read as a perfectly reproducible result.
    """
    if not values:
        return float('nan'), float('nan')
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float('nan')
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def summarize(results, metric, baseline_name=None):
    """Group ok runs by variant and compute per-metric mean/std."""
    # a run that claims success but carries no metrics cannot contribute to any
    # average, so treat it as failed rather than letting it KeyError later
    ok, failed = [], []
    for result in results:
        if result.get('status') == 'ok' and result.get('metrics'):
            ok.append(result)
        else:
            failed.append(result)

    by_variant = {}
    for result in ok:
        by_variant.setdefault(result.get('variant'), []).append(result)

    metric_names = [m for m in REPORT_METRICS
                    if any(m in r['metrics'] for r in ok)]
    extra = sorted({k for r in ok for k in r['metrics']} - set(metric_names))
    metric_names += extra

    rows = []
    for variant, runs in by_variant.items():
        row = {'variant': variant, 'n': len(runs),
               'seeds': sorted(r.get('seed') for r in runs),
               'per_seed': {r.get('seed'): r['metrics'].get(metric) for r in runs},
               'total_sec': sum(r.get('timing', {}).get('total_sec', 0) for r in runs)}
        for name in metric_names:
            values = [r['metrics'][name] for r in runs if name in r['metrics']]
            row[name + '_mean'], row[name + '_std'] = mean_std(values)
        rows.append(row)

    # best first, and for losses "best" means smallest
    higher_is_better = 'loss' not in metric

    def sort_key(row):
        value = row.get(metric + '_mean')
        if value is None or math.isnan(value):
            return (1, 0.0)  # runs missing the metric sink to the bottom
        return (0, -value if higher_is_better else value)

    rows.sort(key=sort_key)

    baseline = None
    if baseline_name:
        baseline = next((r for r in rows if r['variant'] == baseline_name), None)
    if baseline is not None:
        for row in rows:
            mean = row.get(metric + '_mean')
            base_mean = baseline.get(metric + '_mean')
            if mean is None or base_mean is None:
                continue  # the ranking metric is missing, nothing to compare
            row['delta'] = mean - base_mean
            # a paired delta over shared seeds is the honest comparison when the
            # seeds line up, since it cancels the seed-to-seed variance
            shared = [s for s in row['per_seed']
                      if s in baseline['per_seed']
                      and row['per_seed'][s] is not None
                      and baseline['per_seed'][s] is not None]
            if shared and row is not baseline:
                diffs = [row['per_seed'][s] - baseline['per_seed'][s] for s in shared]
                row['paired_delta_mean'], row['paired_delta_std'] = mean_std(diffs)
                row['paired_n'] = len(diffs)

    return {'rows': rows, 'metric': metric, 'metric_names': metric_names,
            'baseline': baseline_name, 'failed': failed, 'n_ok': len(ok)}


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '-'
    return ('%.' + str(digits) + 'f') % value


def markdown_table(summary):
    metric = summary['metric']
    show = [m for m in summary['metric_names'] if m in REPORT_METRICS][:4]
    if metric in show:
        show.remove(metric)
    show = [metric] + show

    header = ['variant', 'n'] + show
    has_delta = any('delta' in row for row in summary['rows'])
    if has_delta:
        header += ['delta vs baseline', 'paired delta']

    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']

    for row in summary['rows']:
        cells = [str(row['variant']), str(row['n'])]
        for name in show:
            cells.append('%s +/- %s' % (fmt(row.get(name + '_mean')),
                                        fmt(row.get(name + '_std'))))
        if has_delta:
            delta = row.get('delta')
            cells.append('%+.4f' % delta if delta is not None else '-')
            if 'paired_delta_mean' in row:
                cells.append('%+.4f +/- %s (n=%d)' % (row['paired_delta_mean'],
                                                      fmt(row['paired_delta_std']),
                                                      row['paired_n']))
            else:
                cells.append('baseline' if row['variant'] == summary['baseline'] else '-')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_outputs(summary, results, results_dir, sweep_name):
    os.makedirs(results_dir, exist_ok=True)

    # one row per run, the raw material for any further analysis
    csv_path = os.path.join(results_dir, 'results.csv')
    metric_names = summary['metric_names']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'variant', 'seed', 'status'] + metric_names
                        + ['total_sec'])
        for result in results:
            metrics = result.get('metrics', {})
            writer.writerow([result.get('run_id'), result.get('variant'),
                             result.get('seed'), result.get('status')]
                            + [metrics.get(m, '') for m in metric_names]
                            + [result.get('timing', {}).get('total_sec', '')])

    summary_path = os.path.join(results_dir, 'summary.md')
    with open(summary_path, 'w') as f:
        f.write('# %s\n\n' % sweep_name)
        f.write('Ranked by %s, mean +/- std over seeds.\n\n' % summary['metric'])
        f.write(markdown_table(summary) + '\n')
        if summary['failed']:
            f.write('\n## Failed runs\n\n')
            for result in summary['failed']:
                f.write('- %s: %s\n' % (result.get('run_id'),
                                        result.get('error', 'unknown error')))
    return csv_path, summary_path


# ------------------------------------------------------------------------ main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config', help='sweep config (JSON, or YAML if PyYAML is installed)')
    p.add_argument('--results-dir', help='override results_dir from the config')
    p.add_argument('--only', action='append', default=[],
                   help='only run variants whose name contains this (repeatable)')
    p.add_argument('--seeds', help='comma-separated seeds, overrides the config')
    p.add_argument('--set', action='append', default=[], dest='overrides',
                   metavar='KEY=VALUE',
                   help='override a base config key, e.g. --set finetune.n_epoch=5')
    p.add_argument('--metric', help='metric to rank by (default %s)' % DEFAULT_METRIC)
    p.add_argument('--dry-run', action='store_true', help='list the runs and exit')
    p.add_argument('--force', action='store_true', help='rerun runs that already have results')
    p.add_argument('--inline', action='store_true',
                   help='run in this process instead of a subprocess (for debugging)')
    p.add_argument('--aggregate', action='store_true',
                   help='skip execution, just rebuild the table from existing results')
    p.add_argument('--quiet', action='store_true', help='do not echo run output, only log it')
    args = p.parse_args(argv)

    config = load_config(args.config)
    sweep_name = config.get('name') or os.path.splitext(os.path.basename(args.config))[0]
    results_dir = args.results_dir or config.get('results_dir') \
        or os.path.join('experiments', 'results', slug(sweep_name))
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(REPO, results_dir)
    metric = args.metric or config.get('metric') or DEFAULT_METRIC

    if args.seeds:
        config['seeds'] = [int(s) for s in args.seeds.split(',')]
    set_keys = []
    for override in args.overrides:
        key, value = parse_set_arg(override)
        set_nested(config.setdefault('base', {}), key, value)
        set_keys.append(key)

    # --set edits the base, and variants are applied on top of the base, so a
    # --set on a key some variant also sets has no effect for that variant.
    # Silently doing nothing is worse than saying so.
    shadowed = sorted({key for key in set_keys
                       for variant in expand_variants(config)
                       if key in variant['overrides']})
    if shadowed:
        print('warning: --set %s overridden by the variants that also set '
              'those keys\n' % ', '.join(shadowed))

    all_runs = build_runs(config)
    validate_init_schemes(all_runs)
    runs = all_runs
    if args.only:
        runs = [r for r in all_runs
                if any(needle.lower() in r['variant'].lower() for needle in args.only)]
        if not runs:
            raise SystemExit('no variant matched %s' % ', '.join(args.only))

    print('sweep       : %s' % sweep_name)
    print('results dir : %s' % results_dir)
    print('runs        : %d (%d variants x %d seeds)'
          % (len(runs), len({r['variant'] for r in runs}),
             len({r['seed'] for r in runs})))
    print('ranking by  : %s\n' % metric)

    if args.dry_run:
        for run_config in runs:
            print('  %-40s %s' % (run_config['run_id'],
                                  json.dumps(run_config.get('model', {}).get('init_scheme'))))
        return 0

    if not args.aggregate:
        started = time.time()
        for index, run_config in enumerate(runs, 1):
            result_path = os.path.join(results_dir, 'runs',
                                       run_config['run_id'] + '.result.json')
            existing = read_result(result_path)
            if existing and existing.get('status') == 'ok' and not args.force:
                stale = (config_fingerprint(existing.get('config') or {})
                         != config_fingerprint(run_config))
                if stale:
                    # reporting the old numbers under the new config would be a
                    # silently wrong answer, so rerun instead of skipping
                    print('[%d/%d] %s: config changed since that result, rerunning'
                          % (index, len(runs), run_config['run_id']))
                else:
                    print('[%d/%d] %s: already done, skipping (--force to redo)'
                          % (index, len(runs), run_config['run_id']))
                    continue

            print('\n[%d/%d] %s' % (index, len(runs), run_config['run_id']), flush=True)
            result = execute(run_config, results_dir, inline=args.inline, quiet=args.quiet)
            if result.get('status') == 'ok':
                value = result['metrics'].get(metric)
                print('[%d/%d] %s: %s = %s (%.1fs)'
                      % (index, len(runs), run_config['run_id'], metric, fmt(value),
                         result.get('timing', {}).get('total_sec', 0)), flush=True)
            else:
                print('[%d/%d] %s FAILED: %s'
                      % (index, len(runs), run_config['run_id'],
                         result.get('error', 'unknown')), flush=True)
        print('\nall runs finished in %.1fs' % (time.time() - started))

    # aggregate over the whole sweep, not just what --only selected, otherwise a
    # filtered rerun overwrites results.csv and summary.md with a subset
    results = collect_results(all_runs, results_dir)
    if not results:
        print('no results found in %s' % results_dir)
        return 1

    summary = summarize(results, metric, config.get('baseline'))
    csv_path, summary_path = write_outputs(summary, results, results_dir, sweep_name)

    print('\n' + '=' * 70)
    print('%s: %d successful runs' % (sweep_name, summary['n_ok']))
    print('=' * 70)
    print(markdown_table(summary))
    if summary['failed']:
        print('\n%d failed run(s):' % len(summary['failed']))
        for result in summary['failed']:
            print('  %s: %s' % (result.get('run_id'), result.get('error', 'unknown')))
    print('\nper-run results : %s' % csv_path)
    print('summary table   : %s' % summary_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
