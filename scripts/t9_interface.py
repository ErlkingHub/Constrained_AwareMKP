"""Port-only static score probes. Oracle rollouts are diagnostics, never inference."""
import hashlib
import numpy as np
import torch
from mkp_cert_basis import cert_basis
from mkp_cert_exec import CONSTANTS
from mkp_v6exec import WeakDecoder,CandidatePolicy,StrategyTheta
from mkp_v6_lp import lp_relaxation
from mkp_v6_basis import interaction_loads,companion_basis
from scripts.t7_interface import StateScores,global_draws,local_draws,oracle_rollout

EXTENSIONS=('x_lp','scar','comp3','bneck','inter1')


def seed_for(iid,step,kind):
    return int.from_bytes(hashlib.sha256(f'1234|{iid}|{step}|{kind}'.encode()).digest()[:8],'little')%(2**63-1)


def legacy_features(state,cfg):
    """Adapter supplying the exact old StateScores geometry from ported code."""
    decoder=WeakDecoder(cfg);owners,items=state.feasible.nonzero(as_tuple=True);n=state.profits.shape[1]
    lp=lp_relaxation(state,cfg);blocked,reserve=decoder.opportunity_features(state,owners,items)
    def dense(values):
        result=values.new_zeros((1,n,*values.shape[1:]));result[owners,items]=values;return result
    cache=dict(blocked=dense(blocked),reserve=dense(reserve),lp_primal=lp.primal,lp_reduced=lp.reduced,
        interaction=dense(interaction_loads(state,owners,items,cfg.eps)),companion=dense(companion_basis(state,owners,items,cfg)))
    return StateScores(state,cfg,decoder,CandidatePolicy,lambda *args:None,lambda *args:cache)


class Probe:
    def __init__(self,state,cfg,family,center,draws,iid,step,temperature=.2):
        self.items=state.feasible[0].nonzero().flatten();self.cfg=cfg;self.state=state;self.family=family
        self.center=center;self.draws=draws;self.generator=torch.Generator().manual_seed(seed_for(iid,step,family))
        self.base=cert_basis(state,cfg);self.v6=None
        self.iid=iid;self.step=step
        self.temperature=temperature

    def scores(self):
        family=self.family;g=self.generator;device=self.items.device;count=self.draws
        k=int(family.split('rand')[-1]) if 'rand' in family else 0
        if family.startswith(('E_full','E_min','v6c')):
            self.v6=legacy_features(self.state,self.cfg)
            half=count//2;global_=global_draws(self.cfg,self.state.weights.shape[-1],half,g)
            # Legal local perturbations around the registered constant, including
            # projection onto interaction/companion gate budgets (T7 protocol).
            std={name:torch.ones_like(torch.as_tensor(value,dtype=torch.float32))*.1 for name,value in self.center.items()}
            local=local_draws({n:torch.as_tensor(v,dtype=torch.float32) for n,v in self.center.items()},std,self.cfg,count-half,g)
            theta={n:torch.cat((global_[n],local[n],torch.as_tensor(self.center[n],dtype=torch.float32)),0).to(device) for n in global_}
            if family.startswith('v6c'):
                for n in ('resource_prices','resource_interactions','bottleneck_weight','reserve_weight'):
                    theta[n]=torch.ones_like(theta[n]) if n=='resource_prices' else torch.zeros_like(theta[n])
            scores=self.v6.logits(theta,'E_min' if family.startswith('E_min') else 'E_full')
            fixed=self.v6.ranking({n:torch.as_tensor(v,device=device,dtype=torch.float32) for n,v in self.center.items()},'E_min' if family.startswith('E_min') else 'E_full')
            temperature_scale=theta['temperature'][:,None]
        else:
            signed=self.base.values[0,self.items].float()*torch.tensor([1,-1,-1,1,1],device=device)
            columns=[signed[:,i] for i in range(5)];names=['sur','reg','blk','fill','pair']
            extensions=family.split('+')[1:];lp=lp_relaxation(self.state,self.cfg)
            for ext in extensions:
                if ext.startswith('rand'):continue
                if ext=='x_lp':columns.append((lp.primal[0,self.items]+self.cfg.lp_primal_floor).log());names.append(ext)
                elif ext=='scar':
                    # Joint sampled beta and nonnegative mixture coefficient.
                    columns.append(torch.zeros_like(columns[0]));names.append(ext)
                elif ext=='comp3':
                    columns.pop(4);names.pop(4)
                    values=companion_basis(self.state,torch.zeros_like(self.items),self.items,self.cfg)
                    columns.extend(values.unbind(-1));names.extend(['comp3_0','comp3_1','comp3_2'])
                elif ext=='bneck':columns.append(-(self.state.weights[0,self.items]/self.state.remaining[0].clamp_min(self.cfg.eps)).amax(-1).float());names.append(ext)
                elif ext=='inter1':
                    z=interaction_loads(self.state,torch.zeros_like(self.items),self.items,self.cfg.eps)
                    columns.append(-(z[:,:,None]*z[:,None,:]).triu().sum((-1,-2)))
                    names.append(ext)
                else:raise ValueError(ext)
            basis=torch.stack(columns,-1)
            pi=-torch.rand(count,len(columns),generator=g).clamp_min(1e-12).log();pi=pi/pi.sum(-1,keepdim=True)
            # Half global, half local around K*. Basis extensions start near zero.
            original=torch.as_tensor(self.center,dtype=torch.float32).flatten();base_names=['sur','reg','blk','fill','pair']
            center=torch.tensor([float(original[base_names.index(n)]) if n in base_names else float(original[4])/3 if n.startswith('comp3_') else 0. for n in names])
            center=center/center.sum()
            local=(center+.1*torch.randn(count//2,len(columns),generator=g)).clamp_min(0);local=local/local.sum(-1,keepdim=True).clamp_min(1e-12)
            pi[-len(local):]=local
            points=torch.stack([torch.tensor([v[base_names.index(n)] if n in base_names else v[4]/3 if n.startswith('comp3_') else 0. for n in names])/sum(v) for v in CONSTANTS.values()])
            pi=torch.cat((pi,center[None],points),0).to(device)
            scores=pi@basis.T;fixed=center.to(device)@basis.T
            if 'scar' in extensions:
                scar_index=names.index('scar')
                loads=(self.state.profits.shape[1]*self.state.weights[0,self.items].float()/self.state.capacities[0].float()).clamp_min(self.cfg.eps).log()
                scarcity=(self.state.capacities[0]/self.state.remaining[0].clamp_min(self.cfg.eps)).log().float()
                beta=torch.rand(len(pi),generator=g).to(device)*self.cfg.scarcity_exponent_max
                scar=-(loads[None]+beta[:,None,None]*scarcity).logsumexp(-1)+np.log(self.state.weights.shape[-1])
                scores=scores+pi[:,scar_index,None]*scar
                fixed_scar=-(loads+scarcity).logsumexp(-1)+np.log(self.state.weights.shape[-1])
                fixed=fixed+center[scar_index].to(device)*fixed_scar
            fixed=CandidatePolicy(torch.zeros(len(self.items),device=device,dtype=torch.long),self.items,fixed/self.temperature,1).log_probs
            scores=scores/self.temperature
            temperature_scale=self.temperature
        if k:
            gen=torch.Generator().manual_seed(seed_for(self.iid,0,'fixed random item bases'))
            noise=torch.randn(self.state.profits.shape[1],32,generator=gen)[:,:k].to(device)
            noise=noise[self.items]*self.base.values[0,self.items,1].std(unbiased=False).float()
            signed=-1+2*torch.rand(len(scores),k,generator=g).to(device)
            scores=scores+(signed@noise.T)/temperature_scale
        return scores,fixed

    def choices(self,scores):
        # Use the same normalized categorical path for tie behavior.
        b,f=scores.shape;policy=CandidatePolicy(torch.arange(b,device=scores.device).repeat_interleave(f),self.items.repeat(b),scores.flatten(),b)
        return policy.choose(False).unique()


def oracle(env,target,cfg,family,center,draws,iid,temperature=.2):
    pointer=[];step=0
    def provider(state,unused):
        nonlocal step
        probe=Probe(state,cfg,family,center,draws,iid,step,temperature);scores,fixed=probe.scores();reachable=probe.choices(scores)
        from scripts.run_t9_measure import exact_lp
        exact=exact_lp(state);p=state.profits[0].cpu().numpy();w=state.weights[0].cpu().numpy()
        reduced=p-w@exact['prices'];feasible=state.feasible[0].cpu().numpy()
        ratio=np.where(feasible,np.abs(reduced)/np.maximum(p,1e-12),np.inf)
        core_np=(exact['x']>1e-8)&(exact['x']<1-1e-8)
        order=np.argsort(ratio,kind='stable');core_np[order[:min(int(feasible.sum()),2*w.shape[-1]+10)]]=True
        core=torch.as_tensor(core_np,device=probe.items.device)[probe.items]
        core_count=int(core.sum());in_core=int(torch.isin(probe.items[core],reachable).sum())
        pointer.append(dict(step=step,feasible_count=len(probe.items),reachable_count=len(reachable),rho_all=len(reachable)/len(probe.items),
            core_count=core_count,reachable_core_count=in_core,rho_core=in_core/core_count if core_count else None,
            core_definition='exact LP dual top2M+10 plus fractional variables'))
        step+=1;return probe,reachable,fixed
    result,diagnostics=oracle_rollout(env,target,provider,family)
    return dict(profit=result['objective'],sequence=result['sequence'],steps=len(result['sequence'][0]),solution=result['selected']),diagnostics,pointer
