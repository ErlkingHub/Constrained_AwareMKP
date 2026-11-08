"""T7 static-state interface measurements and single-episode actor wrappers.

No project model is imported here. Legacy classes/functions are supplied by the
isolated caller. Theta draws are score probes, never sampled item trajectories.
"""
import hashlib
import math

import torch
from torch import nn

MINIMAL = {"profit_exponent", "scarcity_exponent", "lp_primal_weight", "companion_weights", "temperature"}


def schedule_theta(table,tau,phase):
    value = {n:torch.tensor(v,dtype=torch.float32) for n,v in table["base"].items()}
    for n in phase.DYNAMIC:
        value[n] = phase.interpolate(table["knots"],torch.tensor(table["values"][n],dtype=torch.float32),
                                     torch.tensor([tau],dtype=torch.float64))
    return value


def residual_scales(path,tables,phase):
    """Fit diagonal residual scales on synthetic traces only, never evaluation states."""
    import json
    import numpy as np
    from collections import defaultdict
    groups = defaultdict(list)
    with open(path,encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            if r["dataset"] != "synthetic_trace":raise ValueError("Residual fitting cannot use evaluation data")
            groups[phase.key(r,"category")].append(r)
    output = {}
    for group,members in groups.items():
        table = tables[group]
        x = np.asarray([r["tau_cap"] for r in members]);edges = np.quantile(x,np.linspace(0,1,11))
        bins = np.clip(np.searchsorted(edges,x,side="right")-1,0,9)
        residual = {}
        for name in table["base"]:
            observed = torch.tensor([r["theta"][name] for r in members],dtype=torch.float32)
            if name in phase.DYNAMIC:
                predicted = phase.interpolate(table["knots"],torch.tensor(table["values"][name]),torch.tensor(x)).unsqueeze(1)
            else:
                predicted = torch.tensor(table["base"][name],dtype=torch.float32).unsqueeze(0)
            residual[name] = observed-predicted
        scales = []
        for i in range(10):
            mask = torch.tensor(bins==i)
            if not bool(mask.any()):mask = torch.ones(len(x),dtype=torch.bool)
            std = {}
            for name,values in residual.items():
                values = values[mask]
                if name == "resource_prices":
                    sigma = values.double().std(unbiased=False)
                elif name == "resource_interactions":
                    m = values.shape[-1];tri = torch.triu_indices(m,m)
                    sigma = values[:,:,tri[0],tri[1]].double().std(unbiased=False)
                else:
                    sigma = values.double().std(0,unbiased=False)
                std[name] = sigma.float().tolist()
            scales.append(std)
        output[group] = {"edges":edges.tolist(),"bin_counts":[int((bins==i).sum()) for i in range(10)],"std":scales}
    return output


def scales_at(scales,tau):
    import bisect
    return scales["std"][min(9,max(0,bisect.bisect_right(scales["edges"],tau)-1))]


def restrict(theta, executor):
    if executor == "E_full":
        return theta
    if executor != "E_min":
        raise ValueError(executor)
    return {n: (v if n in MINIMAL else torch.ones_like(v) if n == "resource_prices" else torch.zeros_like(v))
            for n,v in theta.items()}


def bounds(cfg):
    return {"profit_exponent":(cfg.profit_exponent_min,cfg.profit_exponent_max),
        "scarcity_exponent":(0.,cfg.scarcity_exponent_max), "bottleneck_weight":(0.,cfg.bottleneck_weight_max),
        "temperature":(cfg.temperature_min,cfg.temperature_max), "opportunity_weight":(0.,cfg.opportunity_weight_max),
        "reserve_weight":(0.,cfg.reserve_weight_max), "lp_primal_weight":(0.,cfg.lp_primal_weight_max),
        "lp_dual_weight":(0.,cfg.lp_dual_weight_max)}


def seeded(seed, dataset, iid, state, kind):
    selected = bytes(state.selected[0].detach().cpu().to(torch.uint8).tolist())
    raw = f"{seed}|{dataset}|{iid}|{kind}|".encode()+selected
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(raw).digest()[:8],"little") % (2**63-1))


def global_draws(cfg, m, count, generator):
    result = {n: lo+(hi-lo)*torch.rand(count,generator=generator) for n,(lo,hi) in bounds(cfg).items()}
    logits = -4+8*torch.rand(count,m,generator=generator)
    result["resource_prices"] = cfg.price_floor+(1-cfg.price_floor)*m*logits.softmax(-1)
    tri = torch.triu_indices(m,m)
    def allocated(d, maximum):
        shares = -torch.rand(count,d,generator=generator).clamp_min(1e-12).log()
        return shares/shares.sum(-1,keepdim=True) * (maximum*torch.rand(count,1,generator=generator))
    values = allocated(tri.shape[1],cfg.interaction_weight_max)
    matrix = torch.zeros(count,m,m); matrix[:,tri[0],tri[1]] = values
    result["resource_interactions"] = matrix+matrix.transpose(-1,-2)-torch.diag_embed(matrix.diagonal(dim1=-2,dim2=-1))
    result["companion_weights"] = allocated(len(cfg.companion_scales),cfg.companion_weight_max)
    validate_batch(result,cfg)
    return result


def local_draws(center, std, cfg, count, generator):
    result = {}
    for name,value in center.items():
        value = value.detach().cpu().float()
        sigma = torch.as_tensor(std[name],dtype=torch.float32)
        result[name] = value+sigma*torch.randn((count,*value.shape[1:]),generator=generator)
    for name,(lo,hi) in bounds(cfg).items():
        result[name] = result[name].clamp(lo,hi)
    p = (result["resource_prices"]-cfg.price_floor).clamp_min(0)
    p = torch.where(p.sum(-1,keepdim=True)>0,p,torch.ones_like(p))
    result["resource_prices"] = cfg.price_floor+(1-cfg.price_floor)*p/p.mean(-1,keepdim=True)
    matrix = result["resource_interactions"].triu().clamp_min(0)
    total = matrix.sum((-1,-2),keepdim=True)
    matrix = matrix*torch.minimum(torch.ones_like(total),cfg.interaction_weight_max/total.clamp_min(1e-12))
    result["resource_interactions"] = matrix+matrix.transpose(-1,-2)-torch.diag_embed(matrix.diagonal(dim1=-2,dim2=-1))
    comp = result["companion_weights"].clamp_min(0)
    result["companion_weights"] = comp*torch.minimum(torch.ones(count,1),cfg.companion_weight_max/comp.sum(-1,keepdim=True).clamp_min(1e-12))
    validate_batch(result,cfg)
    return result


def validate_batch(theta,cfg):
    for n,v in theta.items():
        if not bool(torch.isfinite(v).all()):
            raise ValueError(f"Non-finite theta: {n}")
    for n,(lo,hi) in bounds(cfg).items():
        if bool(((theta[n]<lo-1e-5)|(theta[n]>hi+1e-5)).any()):
            raise ValueError(f"Illegal scalar: {n}")
    p,a,c = theta["resource_prices"],theta["resource_interactions"],theta["companion_weights"]
    if not torch.allclose(p.mean(-1),torch.ones(len(p)),atol=1e-5,rtol=0) or bool((p<cfg.price_floor-1e-5).any()):
        raise ValueError("Illegal prices")
    if (bool((a<0).any()) or not torch.equal(a,a.transpose(-1,-2)) or
            bool((a.triu().sum((-1,-2))>cfg.interaction_weight_max+1e-5).any()) or
            bool((c<0).any()) or bool((c.sum(-1)>cfg.companion_weight_max+1e-5).any())):
        raise ValueError("Illegal combination budgets")


class StateScores:
    """Exact v6 formula with once-only, cached theta-independent state features."""
    def __init__(self,state,cfg,decoder,policy_class,prepare,cached):
        if state.profits.shape[0] != 1:
            raise ValueError("T7 probes require one physical state")
        prepare(state,cfg)
        cache = cached(state,cfg)
        self.cfg,self.policy_class = cfg,policy_class
        self.items = state.feasible[0].nonzero().flatten()
        if not len(self.items):
            raise ValueError("Terminal state has no reachable choice")
        eps = cfg.eps
        p,w,c,rem = (v.float() for v in (state.profits,state.weights,state.capacities,state.remaining))
        candidate_w = w[0,self.items]
        self.log_load = (state.profits.shape[1]*candidate_w/c[0]).clamp_min(eps).log()
        self.log_load = torch.where(candidate_w>0,self.log_load,-torch.inf)
        self.zero = (candidate_w==0).all(-1)
        self.log_scarcity = c[0].clamp_min(eps).log()-rem[0].clamp_min(eps).log()
        self.log_profit = (p[0,self.items]/p.mean(-1)[0].clamp_min(eps)).clamp_min(eps).log()
        self.bottleneck = (candidate_w/rem[0].clamp_min(eps)).amax(-1)
        self.blocked,self.reserve = cache["blocked"][0,self.items],cache["reserve"][0,self.items]
        self.primal = (cache["lp_primal"][0,self.items]+cfg.lp_primal_floor).log()
        self.reduced = cache["lp_reduced"][0,self.items].clamp(-cfg.feature_clip,cfg.feature_clip)
        self.interaction,self.companion = cache["interaction"][0,self.items],cache["companion"][0,self.items]

    def logits(self,theta,executor="E_full"):
        theta = restrict({n:v.to(self.items.device,dtype=torch.float32) for n,v in theta.items()},executor)
        components = (theta["resource_prices"][:,None,:].log()+self.log_load[None]
                      +theta["scarcity_exponent"][:,None,None]*self.log_scarcity[None,None])
        components = torch.where(self.zero[None,:,None],torch.zeros_like(components),components)
        cost = torch.logsumexp(components,-1)-math.log(theta["resource_prices"].shape[1])
        cost = torch.where(self.zero[None],torch.full_like(cost,math.log(self.cfg.eps)),cost)
        scores = theta["profit_exponent"][:,None]*self.log_profit[None]-cost
        scores = scores-theta["bottleneck_weight"][:,None]*self.bottleneck[None]
        scores = scores-theta["opportunity_weight"][:,None]*self.blocked[None]-theta["reserve_weight"][:,None]*self.reserve[None]
        scores = scores+theta["lp_primal_weight"][:,None]*self.primal[None]+theta["lp_dual_weight"][:,None]*self.reduced[None]
        z = self.interaction
        penalty = (theta["resource_interactions"][:,None].triu()*z[None,:,:,None]*z[None,:,None,:]).sum((-1,-2))
        bonus = (theta["companion_weights"][:,None]*self.companion[None]).sum(-1)
        return (scores-penalty+bonus)/theta["temperature"][:,None]

    def choices(self,theta,executor="E_full",chunk=128):
        output = []
        count = len(theta["temperature"])
        for start in range(0,count,chunk):
            logits = self.logits({n:v[start:start+chunk] for n,v in theta.items()},executor)
            k,f = logits.shape
            owners = torch.arange(k,device=logits.device).repeat_interleave(f)
            policy = self.policy_class(owners,self.items.repeat(k),logits.flatten(),k)
            output.append(policy.choose(False))
        return torch.cat(output)

    def ranking(self,theta,executor="E_full"):
        """Single-theta legacy ranking, including normalization and tie rounding."""
        logits=self.logits(theta,executor)
        if logits.shape[0]!=1:raise ValueError("Ranking needs one theta")
        owners=torch.zeros(len(self.items),device=self.items.device,dtype=torch.long)
        return self.policy_class(owners,self.items,logits[0],1).log_probs


class ControlActor(nn.Module):
    """SL/LS switching or period-k updates, with decoder features left untouched."""
    def __init__(self,learned,schedule,phase,theta_class,*,mode="SL",threshold=.8,interval=1,lp_elapsed=None):
        super().__init__()
        self.learned,self.schedule,self.phase,self.theta_class = learned,schedule,phase,theta_class
        self.mode,self.threshold,self.interval = mode,threshold,interval
        self.lp_elapsed = lp_elapsed or (lambda:0.)
        self.updates,self.schedule_updates = 0,0
        self.controller_s,self.controller_non_lp_s = 0.,0.
        self.switch_step,self.switch_feasible = None,None
        self.held = None

    def forward(self,state):
        import time
        step = int(state.selected[0].sum())
        if self.mode == "interval":
            use_learned = step%self.interval == 0
            if not use_learned:
                return self.held,state.profits.new_zeros(state.profits.shape,dtype=torch.float32)
        else:
            tau = float(self.phase.clocks(state)["tau_cap"][0])
            after = self.threshold <= 0 or (self.threshold < 1 and tau >= self.threshold)
            if after and self.switch_step is None:
                self.switch_step,self.switch_feasible = step,int(state.feasible.sum())
            use_learned = after if self.mode == "SL" else not after
        if use_learned:
            def sync():
                if state.profits.is_cuda:torch.cuda.synchronize(state.profits.device)
            sync(); start = time.perf_counter(); lp_before = self.lp_elapsed()
            theta,_ = self.learned(state)
            sync(); elapsed = time.perf_counter()-start
            self.controller_s += elapsed
            self.controller_non_lp_s += max(0.,elapsed-(self.lp_elapsed()-lp_before))
            self.updates += 1
            if self.mode == "interval":self.held = theta.detach()
        else:
            theta,_ = self.schedule(state);self.schedule_updates += 1
        return theta,state.profits.new_zeros(state.profits.shape,dtype=torch.float32)


def oracle_rollout(env,target,provider,executor="E_full"):
    """One target-aware diagnostic episode. No branch, repair or best-of search.

    provider(state,executor) returns StateScores, sampled reachable item IDs and
    S2 logits. A final S2-only completion starts when the target is exhausted.
    """
    target = torch.as_tensor(target,device=env.state.profits.device,dtype=torch.bool).reshape(-1)
    sequence,first_empty,prefix,completion = [],None,[],[]
    all_nonempty = True
    while bool(env.state.active[0]):
        state = env.state
        features,reachable,s2_logits = provider(state,executor)
        remaining = target & ~state.selected[0]
        if not bool(remaining.any()):
            chosen = int(features.items[s2_logits.argmax()]);completion.append(chosen)
        else:
            allowed = torch.isin(features.items,reachable) & remaining[features.items]
            if bool(allowed.any()):
                chosen = int(features.items[s2_logits.masked_fill(~allowed,-torch.inf).argmax()])
            else:
                all_nonempty = False
                if first_empty is None:
                    first_empty = {"step":len(sequence),"feasible_count":len(features.items),
                        "tau_cap":float(1-(state.remaining/state.capacities).mean()),
                        "step_over_N":len(sequence)/state.profits.shape[1]}
                chosen = int(features.items[s2_logits.argmax()])
            prefix.append(chosen)
        env.step(torch.tensor([chosen],device=target.device,dtype=torch.long));sequence.append(chosen)
    selected = env.state.selected[0]
    prefix_subset = all(bool(target[i]) for i in prefix)
    if all_nonempty and not prefix_subset:
        raise AssertionError("Nonempty target intersections must produce a target-only prefix")
    if bool(env.state.feasible.any()):raise AssertionError("Oracle did not finish maximal")
    return {"objective":env.objective,"selected":env.state.selected,
            "sequence":torch.tensor([sequence],device=target.device,dtype=torch.long)}, {
        "first_empty":first_empty,"all_target_intersections_nonempty":all_nonempty,
        "target_prefix_subset":prefix_subset,"target_prefix_items":prefix,"completion_items":completion,
        "final_subset_target":bool((~selected|target).all()),"steps":len(sequence)}
