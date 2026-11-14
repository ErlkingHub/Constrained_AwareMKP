"""T9 compact controller: five simplex weights plus one temperature, for any N/M."""
import math
import torch
from torch import nn
from mkp_cert_exec import CertTheta,CertDecoder,STATISTICS,constant_theta,injected_theta
from mkp_v6_model import TokenEncoder,executor_config,patch_theta
from mkp_v6_env import MKPEnv
from mkp_v6_lp import lp_relaxation
import mkp_v6_lp as lp_module


def initial_logits(cfg):
    weights=torch.as_tensor(cfg.cert_weights,dtype=torch.float32)
    if weights.shape!=(5,) or bool((weights<0).any()) or not bool(torch.isfinite(weights).all()) or float(weights.sum())<=0:
        raise ValueError('cert_weights must be a five-entry simplex point')
    # Finite logits permit training every channel, including initially zero ones.
    return weights.clamp_min(1e-8).log()


def make_head(cfg):
    if not 0<cfg.cert_temperature_min<cfg.cert_temperature<cfg.cert_temperature_max:
        raise ValueError('cert temperature must be strictly inside positive bounds')
    head=nn.Linear(cfg.embed_dim,6);nn.init.zeros_(head.weight)
    t=(cfg.cert_temperature-cfg.cert_temperature_min)/(cfg.cert_temperature_max-cfg.cert_temperature_min)
    with torch.no_grad():head.bias.copy_(torch.cat((initial_logits(cfg),torch.tensor([math.log(t/(1-t))]))))
    return head


def read_head(head,ctx,cfg):
    with torch.autocast(device_type=ctx.device.type,enabled=False):
        raw=head(ctx.float())
        return CertTheta(raw[:,:5].softmax(-1),cfg.cert_temperature_min+(cfg.cert_temperature_max-cfg.cert_temperature_min)*raw[:,5].sigmoid())


class CertActor(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from mkp_Model import StrategyController
        self.tokens=TokenEncoder(cfg.model,cfg.env);self.controller=StrategyController(cfg.model)
        self.config=cfg;self.exec_cfg=executor_config(cfg.model)

    def forward(self,state):
        inputs=self.tokens.inputs(state,self.exec_cfg)
        if self.training and torch.is_grad_enabled() and self.config.model.grad_checkpoint:
            from torch.utils.checkpoint import checkpoint
            weights,temp=checkpoint(self.core,*inputs,use_reentrant=False)
        else:weights,temp=self.core(*inputs)
        return CertTheta(weights,temp)

    def core(self,xs,ys,edge_s,item_d,res_d,edge_d,glob,available):
        x,y,g=self.tokens.encoder.fuse_dynamic(xs,ys,item_d,res_d,glob)
        _,_,ctx=self.controller.representation(x,y,g,torch.cat((edge_s,edge_d),-1),available)
        theta=read_head(self.controller.cert_head,ctx,self.config.model)
        return theta.weights,theta.temperature


class CertPolicy(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        if cfg.train.signal=='ppo_gae' or cfg.model.v6_critic:raise ValueError('cert supports item_pg only, no PPO/critic')
        if cfg.model.theta_family!='cert' or cfg.env.feasibility_mode!='hard_mask' or cfg.model.theta_update!='step' or cfg.model.theta_interval!=1:
            raise ValueError('cert requires matching family, hard mask, and every-step theta')
        self.cfg=cfg;self.exec_cfg=executor_config(cfg.model)
        self.actor=CertActor(cfg);self.decoder=CertDecoder(self.exec_cfg)

    def default_theta(self,state):
        c=self.cfg.model;head=self.actor.controller.cert_head
        # Only the zero-head activation, independent of trained parameters/RNG.
        logits=initial_logits(c).to(state.profits.device)
        t=(c.cert_temperature-c.cert_temperature_min)/(c.cert_temperature_max-c.cert_temperature_min)
        temp=c.cert_temperature_min+(c.cert_temperature_max-c.cert_temperature_min)*torch.tensor(math.log(t/(1-t)),device=logits.device).sigmoid()
        return CertTheta(logits.softmax(-1).expand(len(state.profits),-1),temp.expand(len(state.profits)))

    def policy(self,state):
        theta=self.actor(state)
        return self.decoder(state,theta),theta,None

    def step_forward(self,state):raise ValueError('cert does not expose PPO/critic forwarding')
    def value(self,state):raise ValueError('cert has no critic')

    def episode(self,inst,num_samples,decode,override,return_trace,theta_patch=None,prefix_delta=None):
        if theta_patch is not None:raise ValueError('cert uses simplex overrides, not v6 field patches')
        if prefix_delta is not None:return prefix_episode(self,inst,decode,prefix_delta,override,None,return_trace,num_samples)
        if decode not in ('greedy','sample') or num_samples<1:raise ValueError('Invalid cert decode')
        p=inst.p.double().repeat_interleave(num_samples,0);w=inst.r.transpose(1,2).double().repeat_interleave(num_samples,0);c=inst.c.double().repeat_interleave(num_samples,0)
        env=MKPEnv(p,w,c);state=env.state;root=lp_relaxation(state,self.exec_cfg);state._root=(root.dual,root.bound)
        logp=p.new_zeros(len(p),dtype=torch.float32);entropy=logp.clone();count=logp.clone()
        total=logp.new_zeros(6);observations=0;sequence=[];trace=[]
        previous=None
        for step in range(inst.N):
            active=state.active
            if not bool(active.any()):break
            current=lp_relaxation(state,self.exec_cfg)
            if override is None:policy,theta,_=self.policy(state)
            else:
                theta=self.default_theta(state) if isinstance(override,str) and override=='fixed' else injected_theta(override,len(p),self.cfg.model.cert_temperature,p.device)
                policy=self.decoder(state,theta)
            action=policy.choose(decode=='sample')
            logp=logp+policy.log_prob(action)*active;entropy=entropy+policy.entropy()*active;count=count+active
            with torch.no_grad():total+=(theta.statistics()*active[:,None]).sum(0);observations+=int(active.sum())
            if return_trace:
                from mkp_cert_basis import cert_basis
                basis=cert_basis(state,self.exec_cfg);owner=torch.arange(len(p),device=p.device);picked=action.clamp_min(0)
                if previous is not None:
                    trace[-1]['dual_reoptimization_gain']=(previous-current.dual_value).cpu().tolist()
                trace.append({'step':step,'tau_cap':(1-(state.remaining/state.capacities).mean(1)).cpu().tolist(),
                    'theta':{'weights':theta.weights.detach().cpu().tolist(),'temperature':theta.temperature.detach().cpu().tolist()},
                    'own_regret':(basis.values[owner,picked,1]*basis.scale*active).cpu().tolist(),
                    'blocked_regret':(basis.values[owner,picked,2]*basis.scale*active).cpu().tolist()})
                previous=current.dual_value-p[owner,picked]*active-trace_value(basis,owner,picked,active)
            sequence.append(action);env.step(action)
        if return_trace and trace:trace[-1]['dual_reoptimization_gain']=previous.cpu().tolist()
        return dict(solution=state.selected,profit=env.objective,reward=(env.objective/root.bound.clamp_min(1e-8)).float(),ub=root.bound,
            log_likelihood=logp,entropy=entropy/count.clamp_min(1),n_reject=torch.zeros_like(logp),steps=len(sequence),
            sequence=torch.stack(sequence,1) if sequence else torch.empty(len(p),0,device=p.device,dtype=torch.long),
            trace=trace,theta_statistics=dict(zip(STATISTICS,(total/max(1,observations)).tolist())))


def trace_value(basis,owners,items,active):
    return (basis.values[owners,items,1]+basis.values[owners,items,2])*basis.scale*active


@torch.no_grad()
def prefix_episode(engine,inst,decode,delta,override,patch,return_trace,num_samples):
    """One trajectory; LP-driven prefix then permanently learned core phase."""
    if inst.B!=1 or num_samples!=1 or decode!='greedy' or delta<0:
        raise ValueError('prefix requires B1, one greedy trajectory and nonnegative delta')
    env=MKPEnv(inst.p.double(),inst.r.transpose(1,2).double(),inst.c.double());state=env.state
    began=lp_module.LP_SOLVE_BATCHES;root=lp_relaxation(state,engine.exec_cfg);state._root=(root.dual,root.bound)
    sequence=[];trace=[];core=False;calls=0;prefix_items=0
    while bool(state.active.any()):
        lp=lp_relaxation(state,engine.exec_cfg)
        if not core:
            eligible=state.feasible & (lp.primal>=.99) & (lp.reduced/(state.profits/lp.scale[:,None]).clamp_min(engine.exec_cfg.eps)>=delta)
            ids=eligible[0].nonzero(as_tuple=True)[0]
            if len(ids):
                ids=ids[lp.reduced[0,ids].argsort(descending=True,stable=True)]
                for item in ids:
                    if bool(state.feasible[0,item]):env.step(item.reshape(1));sequence.append(int(item));prefix_items+=1
                continue
            core=True
        if override is None:
            policy,theta,_=engine.policy(state);calls+=1
        else:
            if isinstance(engine,CertPolicy):theta=engine.default_theta(state) if isinstance(override,str) and override=='fixed' else injected_theta(override,1,engine.cfg.model.cert_temperature,state.profits.device)
            else:
                from mkp_v6exec import StrategyTheta
                theta=engine.default_theta(state) if isinstance(override,str) else StrategyTheta(**{n:torch.as_tensor(v,device=state.profits.device,dtype=torch.float32) for n,v in override.items()})
            policy=engine.decoder(state,theta)
        if patch is not None:theta=patch_theta(theta,patch);policy=engine.decoder(state,theta)
        action=policy.choose(False);sequence.append(int(action[0]));env.step(action)
    return dict(solution=state.selected,profit=env.objective,reward=(env.objective/root.bound).float(),ub=root.bound,
        sequence=torch.tensor([sequence],device=state.profits.device),steps=len(sequence),trace=trace,
        controller_calls=calls,lp_solves=lp_module.LP_SOLVE_BATCHES-began,prefix_items=prefix_items)
