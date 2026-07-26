"""Pluggable weight initialization for TxGNN.

The stock model applies Xavier-uniform in three places (the node input
embeddings, the DistMult relation matrix w_rels, and the prototype gates) and
leaves the per-relation Linear layers on PyTorch's default Kaiming-uniform.
Passing ``init_scheme = None`` reproduces that exactly, so the historical
behaviour stays the baseline of any sweep.

Any other spec replaces the initializer at every one of those sites, which is
what makes a comparison across schemes a one-variable experiment.

Spec forms::

    None or 'default'      repo default described above
    'kaiming_uniform'      one scheme everywhere
    {'name': 'normal', 'std': 0.02}
                           one scheme everywhere, with keyword arguments
    {'embedding': 'normal', 'layer': 'kaiming_uniform'}
                           per component, unnamed components keep the default
    {'name': 'orthogonal', 'layer': 'xavier_uniform'}
                           orthogonal everywhere except the Linear layers

Components:

``embedding``   node input features (``G.nodes[t].data['inp']``, frozen)
``layer``       per-relation Linear weights in both HeteroRGCN layers
``relation``    ``w_rels``, the DistMult decoder matrix
``gate``        the prototype gating Linear in DistMultPredictor

A spec must be picklable: it is stored verbatim in the model config so a
checkpoint records how it was initialized.
"""

import inspect

import torch.nn as nn

COMPONENTS = ('embedding', 'layer', 'relation', 'gate')

# name -> in-place torch init function. 'torch_default' is the absence of an
# explicit init, which only means something for modules that initialize
# themselves (the Linear layers).
SCHEMES = {
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_,
    'normal': nn.init.normal_,
    'uniform': nn.init.uniform_,
    'trunc_normal': nn.init.trunc_normal_,
    'torch_default': None,
}

# Spec keys that configure this module rather than the torch init function.
META_KEYS = ('zero_bias',)

# What the repo did before init_scheme existed. None means "leave the module's
# own initialization alone".
REPO_DEFAULT = {
    'embedding': ('xavier_uniform', {}),
    'layer': ('torch_default', {}),
    'relation': ('xavier_uniform', {'gain': 'relu'}),
    'gate': ('xavier_uniform', {}),
}

# These tensors are allocated with torch.Tensor(...), i.e. uninitialized
# memory, so they have no module-level default to fall back on.
_REQUIRES_INIT = ('embedding', 'relation')


class Initializer:
    """A named init function bound to its keyword arguments.

    Callable on a tensor, and carries enough information to print itself and to
    round-trip back into a config spec.

    Schemes apply to weights only. Biases keep PyTorch's own initialization
    unless the spec passes ``zero_bias: true``, which keeps the baseline path
    bit-identical to the code before this module existed.
    """

    def __init__(self, name, kwargs=None):
        if name not in SCHEMES:
            raise ValueError(
                "unknown init scheme %r, choose one of: %s"
                % (name, ', '.join(sorted(SCHEMES)))
            )
        self.name = name
        self.kwargs = dict(kwargs or {})
        self.zero_bias = bool(self.kwargs.get('zero_bias', False))
        self.fn = SCHEMES[name]
        init_kwargs = {k: v for k, v in self.kwargs.items() if k not in META_KEYS}
        if self.fn is None:
            if init_kwargs:
                raise ValueError("scheme 'torch_default' takes no arguments, got %s"
                                 % ', '.join(sorted(init_kwargs)))
            self._call_kwargs = {}
        else:
            self._call_kwargs = _prepare_kwargs(self.fn, name, init_kwargs)

    def __call__(self, tensor):
        if self.fn is None:
            return tensor
        return self.fn(tensor, **self._call_kwargs)

    def spec(self):
        """The picklable spec that rebuilds this initializer."""
        if not self.kwargs:
            return self.name
        out = {'name': self.name}
        out.update(self.kwargs)
        return out

    def __repr__(self):
        if not self.kwargs:
            return self.name
        args = ', '.join('%s=%r' % kv for kv in sorted(self.kwargs.items()))
        return '%s(%s)' % (self.name, args)

    def __eq__(self, other):
        return (isinstance(other, Initializer)
                and self.name == other.name and self.kwargs == other.kwargs)

    def __hash__(self):
        # defining __eq__ alone would make these unhashable
        return hash((self.name, tuple(sorted(self.kwargs.items()))))


def _prepare_kwargs(fn, name, kwargs):
    """Validate kwargs against the torch init signature, resolving `gain`."""
    accepted = set(inspect.signature(fn).parameters) - {'tensor'}
    unknown = set(kwargs) - accepted
    if unknown:
        raise ValueError(
            "init scheme %r does not accept %s (accepts: %s)"
            % (name, ', '.join(sorted(unknown)), ', '.join(sorted(accepted)))
        )
    prepared = dict(kwargs)
    if isinstance(prepared.get('gain'), str):
        # 'relu' -> nn.init.calculate_gain('relu'), so configs can name the
        # nonlinearity instead of hard-coding a number
        prepared['gain'] = nn.init.calculate_gain(prepared['gain'])
    return prepared


def _parse_one(spec):
    """Turn a single scheme spec (str or dict) into an Initializer."""
    if spec is None:
        return None
    if isinstance(spec, Initializer):
        return spec
    if isinstance(spec, str):
        return Initializer(spec)
    if isinstance(spec, dict):
        kwargs = dict(spec)
        name = kwargs.pop('name', None)
        if name is None:
            raise ValueError("init scheme dict needs a 'name' key, got %r" % (spec,))
        return Initializer(name, kwargs)
    raise TypeError("init scheme must be a str or dict, got %r" % type(spec).__name__)


def resolve_init_spec(spec):
    """Expand any spec form into ``{component: Initializer or None}``.

    ``None`` for a component means "leave the module's own initialization",
    which is only legal for components that have one.
    """
    if spec is None or spec == 'default':
        resolved = {c: _parse_one(_default_spec(c)) for c in COMPONENTS}
        return _check(resolved)

    if isinstance(spec, (str, Initializer)):
        return _check({c: _parse_one(spec) for c in COMPONENTS})

    if not isinstance(spec, dict):
        raise TypeError("init_scheme must be None, a str or a dict, got %r"
                        % type(spec).__name__)

    per_component = {k: v for k, v in spec.items() if k in COMPONENTS}
    globals_ = {k: v for k, v in spec.items() if k not in COMPONENTS}

    if globals_ and 'name' not in globals_:
        raise ValueError(
            "init_scheme dict keys must be component names (%s) or a global "
            "scheme with a 'name' key, got %s"
            % (', '.join(COMPONENTS), ', '.join(sorted(globals_)))
        )

    if globals_:
        base = _parse_one(globals_)
        resolved = {c: base for c in COMPONENTS}
    else:
        resolved = {c: _parse_one(_default_spec(c)) for c in COMPONENTS}

    for component, sub in per_component.items():
        resolved[component] = _parse_one(sub)
    return _check(resolved)


def _default_spec(component):
    name, kwargs = REPO_DEFAULT[component]
    if not kwargs:
        return name
    out = {'name': name}
    out.update(kwargs)
    return out


def _check(resolved):
    for component in _REQUIRES_INIT:
        init = resolved[component]
        if init is None or init.fn is None:
            raise ValueError(
                "component %r is allocated uninitialized and needs a real init "
                "scheme, 'torch_default' will leave it as garbage memory"
                % component
            )
    return resolved


def describe(spec):
    """One-line human-readable summary of a spec, for logs and run names."""
    resolved = resolve_init_spec(spec)
    if resolved == resolve_init_spec(None):
        return 'default (xavier_uniform, PyTorch defaults for the Linear layers)'
    if len({repr(v) for v in resolved.values()}) == 1:
        return repr(resolved['embedding'])
    return ', '.join('%s=%s' % (c, resolved[c]) for c in COMPONENTS)


def init_linear_(linear, initializer):
    """Apply an initializer to a Linear's weight, and its bias if asked.

    A no-op when ``initializer`` is None or 'torch_default', which is how the
    baseline keeps PyTorch's own Linear initialization.
    """
    if initializer is None or initializer.fn is None:
        return linear
    initializer(linear.weight)
    if initializer.zero_bias and linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear
