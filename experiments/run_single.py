"""Run one TxGNN configuration and write its metrics to JSON.

This is the unit of work the sweep runner schedules. It is also usable on its
own when you want to reproduce a single run from a sweep:

    python -m experiments.run_single \
        --run-config experiments/results/my_sweep/runs/kaiming__seed1.json

The run config is a JSON object:

    {
      "run_id":     "kaiming_uniform__seed1",
      "variant":    "kaiming_uniform",
      "seed":       1,
      "data":       "./data_mini",
      "split":      "complex_disease",
      "split_seed": 42,
      "device":     "cpu",
      "model":      {...}   -> TxGNN.model_initialize(**model)
      "pretrain":   {...}   -> TxGNN.pretrain(**pretrain), or null to skip
      "finetune":   {...}   -> TxGNN.finetune(**finetune)
      "save_dir":   null
    }

Only `data` and `finetune` are really required, everything else has a default.
"""

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DD_RELATIONS = ('indication', 'contraindication', 'off-label use')


def set_seed(seed):
    """Seed every RNG the training path touches.

    Note txgnn/TxGNN.py calls torch.manual_seed(0) at import time, so this has
    to run after the import to take effect.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # without these, two CUDA runs of the same seed still diverge, which would
    # quietly undermine every paired comparison the sweep reports
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(requested):
    import torch
    if requested:
        return requested
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def flatten_relation_metrics(metrics):
    """Turn {('drug','indication','disease'): 0.9} into JSON-safe flat keys."""
    out = {}
    for key in ('test_auroc_per_relation', 'test_auprc_per_relation'):
        per_rel = metrics.pop(key, None) or {}
        stat = 'auroc' if 'auroc' in key else 'auprc'
        for etype, value in per_rel.items():
            if isinstance(etype, tuple):
                # only the drug -> disease direction, the reverse duplicates it
                if etype[0] != 'drug' or etype[1] not in DD_RELATIONS:
                    continue
                name = etype[1].replace(' ', '_')
            else:
                name = str(etype)
            out['test_%s_%s' % (stat, name)] = float(value)
    metrics.update(out)
    return metrics


def run(config):
    """Execute one run and return its result dict."""
    started = time.time()

    from txgnn import TxData, TxGNN
    from txgnn.init_schemes import describe as describe_init_scheme

    seed = config.get('seed', 42)
    device = pick_device(config.get('device'))
    model_kwargs = dict(config.get('model') or {})
    pretrain_kwargs = config.get('pretrain')
    finetune_kwargs = dict(config.get('finetune') or {})

    print('=' * 70)
    print('run_id : %s' % config.get('run_id', '<unnamed>'))
    print('variant: %s' % config.get('variant', '<unnamed>'))
    print('seed   : %s   device: %s' % (seed, device))
    print('init   : %s' % describe_init_scheme(model_kwargs.get('init_scheme')))
    print('=' * 70, flush=True)

    set_seed(seed)

    data = TxData(data_folder_path=config['data'])
    data.prepare_split(split=config.get('split', 'complex_disease'),
                       seed=config.get('split_seed', 42),
                       no_kg=config.get('no_kg', False))

    # the split itself must not depend on the run seed, so reseed after it
    set_seed(seed)

    model = TxGNN(data=data,
                  weight_bias_track=False,
                  proj_name=config.get('proj_name', 'TxGNN_experiments'),
                  exp_name=config.get('run_id', 'run'),
                  device=device)
    model.model_initialize(**model_kwargs)

    t_pretrain = 0.0
    if pretrain_kwargs:
        t0 = time.time()
        model.pretrain(**pretrain_kwargs)
        t_pretrain = time.time() - t0

    t0 = time.time()
    metrics = model.finetune(**finetune_kwargs)
    t_finetune = time.time() - t0

    if config.get('save_dir'):
        # TxGNN.save_model uses os.mkdir, which cannot create intermediate
        # directories, and sweep checkpoints are nested one level per run
        os.makedirs(config['save_dir'], exist_ok=True)
        model.save_model(config['save_dir'])

    metrics = flatten_relation_metrics(dict(metrics))
    metrics = {k: (float(v) if isinstance(v, (int, float)) else v)
               for k, v in metrics.items()}

    return {
        'run_id': config.get('run_id'),
        'variant': config.get('variant'),
        'seed': seed,
        'status': 'ok',
        'metrics': metrics,
        'timing': {'pretrain_sec': round(t_pretrain, 2),
                   'finetune_sec': round(t_finetune, 2),
                   'total_sec': round(time.time() - started, 2)},
        'env': {'device': device,
                'python': platform.python_version(),
                'torch': __import__('torch').__version__},
        'config': config,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-config', required=True, help='JSON file describing one run')
    p.add_argument('--out', help='where to write the result JSON '
                                 '(default: alongside the run config, *.result.json)')
    args = p.parse_args(argv)

    with open(args.run_config) as f:
        config = json.load(f)

    out_path = args.out or os.path.splitext(args.run_config)[0] + '.result.json'

    try:
        result = run(config)
    except Exception as exc:
        result = {'run_id': config.get('run_id'),
                  'variant': config.get('variant'),
                  'seed': config.get('seed'),
                  'status': 'failed',
                  'error': '%s: %s' % (type(exc).__name__, exc),
                  'traceback': traceback.format_exc(),
                  'config': config}
        traceback.print_exc()

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print('\nwrote %s' % out_path)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
