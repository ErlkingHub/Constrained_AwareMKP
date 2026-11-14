"""Execution-only CUDA graphs for the unchanged cold, float64 LP solve.

Every call returns owned tensors. Later replays cannot overwrite a root context
or inputs saved by autograd. No precision, Newton schedule or RNG is changed.
"""
from collections import OrderedDict
from dataclasses import fields
from types import SimpleNamespace
import warnings
import torch

_BACKEND='eager'
_CACHE=OrderedDict()
_DISABLED={}
_STATS={}
MAX_GRAPHS=16


def set_backend(name):
    global _BACKEND
    if name not in ('eager','cuda_graph'):raise ValueError('LP backend must be eager or cuda_graph')
    if name!=_BACKEND:clear()
    _BACKEND=name


def clear():
    _CACHE.clear();_DISABLED.clear();_STATS.clear()


def statistics():
    return dict(backend=_BACKEND,cached_graphs=len(_CACHE),disabled_shapes=len(_DISABLED),**_STATS)


def _increment(name,count=1):_STATS[name]=_STATS.get(name,0)+count


class _Graph:
    def __init__(self,state,config,eager):
        self.inputs=SimpleNamespace(**{n:getattr(state,n).clone() for n in ('profits','weights','remaining','feasible')})
        self.infos=[]
        def solve(a,b):
            result,info=torch.linalg.solve_ex(a,b,check_errors=False)
            self.infos.append(info)
            return result
        from mkp_v6_lp import _STEPS
        # This is allocated OUTSIDE graph capture and must outlive every replay.
        self.steps=_STEPS.to(device=state.profits.device,dtype=torch.float64)
        stream=torch.cuda.Stream(device=state.profits.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                eager(self.inputs,config,_linear_solve=solve,_steps=self.steps)
                _increment('warmup_batches');_increment('warmup_states',len(state.profits))
        torch.cuda.current_stream().wait_stream(stream)
        if int(torch.stack(self.infos).abs().max()):raise RuntimeError('LP warmup linear solve failed')
        self.infos.clear()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output=eager(self.inputs,config,_linear_solve=solve,_steps=self.steps)
            self.status=torch.stack(self.infos).abs().amax()
        self.infos.clear();_increment('captures')

    def replay(self,state):
        for name in ('profits','weights','remaining','feasible'):
            getattr(self.inputs,name).copy_(getattr(state,name))
        self.graph.replay()
        # One check covers every solve. Numerical failures still fail closed.
        status=int(self.status)
        if status:raise RuntimeError(f'LP CUDA graph linear solve failed, info={status}')
        _increment('replays')
        return type(self.output)(**{f.name:getattr(self.output,f.name).clone() for f in fields(self.output)})


@torch.no_grad()
def execute(state,config,eager):
    if _BACKEND=='eager' or not state.profits.is_cuda:
        _increment('eager_calls');return eager(state,config)
    from mkp_v6_lp import lp_signature
    # Dense permutations (including r.transpose(1,2)) preserve their strides on
    # clone. Unusual sliced/overlapping views keep the reference execution path.
    tensors=tuple(getattr(state,n) for n in ('profits','weights','remaining','feasible'))
    for value in tensors:
        expected=1
        for stride,size in sorted((s,n) for s,n in zip(value.stride(),value.shape) if n>1):
            if stride!=expected:
                _increment('layout_fallback_calls');return eager(state,config)
            expected*=size
    key=(str(state.profits.device),tuple((tuple(v.shape),v.dtype,v.stride()) for v in tensors),lp_signature(config))
    if key in _DISABLED:
        _increment('fallback_calls');return eager(state,config)
    with torch.cuda.device(state.profits.device):
        if key not in _CACHE:
            try:item=_Graph(state,config,eager)
            except RuntimeError as exc:
                _DISABLED[key]=str(exc);_increment('capture_failures')
                warnings.warn(f'LP CUDA graph unavailable for {key[:3]}: {exc}; using eager',RuntimeWarning)
                _increment('fallback_calls');return eager(state,config)
            if len(_CACHE)>=MAX_GRAPHS:_CACHE.popitem(last=False);_increment('evictions')
            _CACHE[key]=item
        _CACHE.move_to_end(key)
        return _CACHE[key].replay(state)
