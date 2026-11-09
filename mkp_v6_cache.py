# Ported state_stamp from RL_ModelInhance/mkp_performance.py, commit 4f93555.
"""State-local LP memo identity. Optional rollout geometry-bank caching is disabled.

This omits only the legacy memory24 execution optimization; no PPO arithmetic,
features, LP schedule or training targets are changed.
"""
def state_stamp(state):
    return tuple((id(x),x._version) for x in
                 (state.profits,state.weights,state.capacities,state.remaining,state.selected))


def cached_features(state,config):
    return None


def prepare_state(state,config):
    from mkp_v6_lp import lp_relaxation
    lp_relaxation(state,config)
    return state


def build_rollout_cache(model,data,rollout,times,batches,config):
    if model.reuse_state_features:
        raise ValueError("T8 uses the reference arithmetic path, not memory24 geometry banks")
    return None,{"feature_cache_gib":0.,"feature_cache_seconds":0.}
