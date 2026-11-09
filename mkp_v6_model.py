"""New-project dual-axis controller with the faithfully ported v6 executor.

Activation defaults/LP features: RL_ModelInhance/mkp_Model.py, commit 4f93555.
The encoder/controller backbone is this project's StateEncoder/StrategyController.
"""
import math
import torch
from torch import nn
from mkp_config import ModelConfig
from mkp_env import MKPEnv as FeatureEnv, MKPInstances, MKPState as FeatureState
from mkp_v6_config import ModelConfig as ExecutorConfig
from mkp_v6_env import MKPEnv, MKPState
from mkp_v6_lp import lp_relaxation
from mkp_v6_basis import CombinationControlHead
from mkp_v6exec import StrategyTheta, WeakDecoder, THETA_STATISTICS


def executor_config(cfg):
    return ExecutorConfig(embed_dim=cfg.embed_dim,num_heads=cfg.num_heads,price_floor=cfg.price_floor)


class V6Heads(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.config=executor_config(cfg)
        self.scalar=nn.Linear(cfg.embed_dim,8)
        self.combination=CombinationControlHead(self.config)
        nn.init.zeros_(self.scalar.weight)
        c=self.config
        self.ranges=((c.profit_exponent_min,c.profit_exponent_max),(0,c.scarcity_exponent_max),(0,c.bottleneck_weight_max),
            (c.temperature_min,c.temperature_max),(0,c.opportunity_weight_max),(0,c.reserve_weight_max),
            (0,c.lp_primal_weight_max),(0,c.lp_dual_weight_max))
        with torch.no_grad():
            for j,(v,(lo,hi)) in enumerate(zip((1.,1.,.1,.2,.1,.1,c.lp_primal_initial,c.lp_dual_initial),self.ranges)):
                fraction=min(.999,max(.001,(v-lo)/(hi-lo))) if hi>lo else .5
                self.scalar.bias[j]=math.log(fraction/(1-fraction))

    def forward(self,logits,resources,pooled):
        c=self.config;m=resources.shape[1]
        prices=c.price_floor+(1-c.price_floor)*m*logits.softmax(-1)
        raw=self.scalar(pooled).sigmoid()
        values=[lo+(hi-lo)*raw[:,i] for i,(lo,hi) in enumerate(self.ranges)]
        interaction,companion=self.combination(resources,pooled)
        return StrategyTheta(prices,*values,interaction,companion)


class TokenEncoder(nn.Module):
    def __init__(self,cfg,env_cfg):
        super().__init__()
        from mkp_Model import StateEncoder
        self.encoder=StateEncoder(cfg);self.feature_env=FeatureEnv(env_cfg);self.cfg=cfg

    def inputs(self,state,exec_cfg):
        lp=lp_relaxation(state,exec_cfg)
        if state._root is None:
            if bool(state.selected.any()):
                root=MKPEnv(state.profits,state.weights,state.capacities).state
                initial=lp_relaxation(root,exec_cfg)
            else:initial=lp
            state._root=(initial.dual.detach(),initial.bound.detach())
        mu,ub=state._root
        inst=MKPInstances(state.profits.float(),state.weights.transpose(1,2).float(),state.capacities.float())
        status=state.selected.to(torch.int8)
        bridge=FeatureState(inst,mu,ub.float(),state.remaining.float(),status,
            (state.profits*state.selected).sum(-1).float(),torch.zeros_like(ub,dtype=torch.float32),
            ~state.active,state.feasible)
        versions=tuple(p._version for p in self.encoder.parameters())
        signature=(id(self),versions,torch.is_grad_enabled(),self.training,torch.is_autocast_enabled())
        cached=getattr(state,"_new_static",{})
        if id(self) in cached and cached[id(self)][0]==signature:
            xs,ys,edge_s=cached[id(self)][1]
        else:
            item_s,res_s,edge_s=self.feature_env.static_features(inst,mu,ub.float())
            xs,ys=self.encoder.encode_static(item_s,res_s,edge_s)
            cached[id(self)]=(signature,(xs,ys,edge_s))
            state._new_static=cached
        item_d,res_d,edge_d,glob=self.feature_env.dynamic_features(bridge)
        if self.cfg.ctrl_residual_lp_features:
            eps,clip=exec_cfg.eps,exec_cfg.feature_clip
            dual_cost=lp.dual_share.sum(-1)
            lp_items=torch.stack((lp.primal,lp.reduced.clamp(-clip,clip),
                dual_cost.clamp(1/clip,clip).log()*state.feasible,4*lp.primal*(1-lp.primal)),-1)
            lp_resources=torch.stack((lp.dual.clamp(0,clip).log1p(),lp.slack,
                lp.dual/lp.dual.sum(-1,keepdim=True).clamp_min(eps)),-1)
            lp_global=torch.stack(((lp.bound/state.profits.sum(-1)).float(),
                                  (lp.bound/lp.feasible_mass.clamp_min(eps)).float()),-1)
            item_d=torch.cat((item_d,lp_items),-1);res_d=torch.cat((res_d,lp_resources),-1);glob=torch.cat((glob,lp_global),-1)
        return xs,ys,edge_s,item_d,res_d,edge_d,glob,~state.selected

    def forward(self,state,exec_cfg):
        xs,ys,edge_s,item_d,res_d,edge_d,glob,available=self.inputs(state,exec_cfg)
        x,y,g=self.encoder.fuse_dynamic(xs,ys,item_d,res_d,glob)
        return x,y,g,torch.cat((edge_s,edge_d),-1),available


class V6Actor(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from mkp_Model import StrategyController
        self.tokens=TokenEncoder(cfg.model,cfg.env);self.controller=StrategyController(cfg.model)
        self.feasibility=nn.Linear(cfg.model.embed_dim,1) if cfg.model.v6_critic else None
        self.exec_cfg=executor_config(cfg.model);self.cfg=cfg

    def forward(self,state):
        inputs=self.tokens.inputs(state,self.exec_cfg)
        if self.training and torch.is_grad_enabled() and self.cfg.model.grad_checkpoint:
            from torch.utils.checkpoint import checkpoint
            output=checkpoint(self.core,*inputs,use_reentrant=False)
        else:output=self.core(*inputs)
        return StrategyTheta(*output[:-1]),output[-1]

    def core(self,xs,ys,edge_s,item_d,res_d,edge_d,glob,available):
        x,y,g=self.tokens.encoder.fuse_dynamic(xs,ys,item_d,res_d,glob)
        edge=torch.cat((edge_s,edge_d),-1)
        # Exactly the new StrategyController representation; only heads differ.
        x,y,ctx=self.controller.representation(x,y,g,edge,available)
        with torch.autocast(device_type=x.device.type,enabled=False):
            m=y.shape[1]
            logits=self.controller.price_head(torch.cat((y.float(),ctx.float()[:,None].expand(-1,m,-1)),-1)).squeeze(-1)
            theta=self.controller.v6_heads(logits,y.float(),ctx.float())
            feasibility=(self.feasibility(x.float()).squeeze(-1) if self.feasibility is not None and self.training else
                         torch.zeros(x.shape[:2],device=x.device,dtype=torch.float32))
        return tuple(getattr(theta,n) for n in theta.__dataclass_fields__)+(feasibility,)


class V6Critic(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from mkp_Model import DualAxisBlock
        from copy import deepcopy
        critic_cfg=deepcopy(cfg.model);critic_cfg.ctrl_residual_lp_features=True
        self.tokens=TokenEncoder(critic_cfg,cfg.env)
        c=cfg.model;d=c.embed_dim
        self.blocks=nn.ModuleList([DualAxisBlock(d,c.num_heads,c.ff_mult,c.dropout,4,c.edge_hidden) for _ in range(c.controller_layers)])
        self.ln_x,self.ln_y=nn.LayerNorm(d),nn.LayerNorm(d)
        self.ctx=nn.Sequential(nn.Linear(4*d,d),nn.GELU(),nn.Linear(d,d))
        self.value_head=nn.Linear(d,1);self.exec_cfg=executor_config(c)

    def forward(self,state):
        from mkp_Model import StrategyController
        x,y,g,edge,available=self.tokens(state,self.exec_cfg)
        _,_,ctx=StrategyController.representation(self,x,y,g,edge,available)
        return self.value_head(ctx.float()).squeeze(-1)


class V6Policy(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        if cfg.model.theta_family!="v6" or cfg.env.feasibility_mode!="hard_mask" or cfg.model.theta_update!="step" or cfg.model.theta_interval!=1:
            raise ValueError("T8 requires v6 theta, hard mask and every-step controls")
        self.cfg=cfg;self.exec_cfg=executor_config(cfg.model)
        self.actor=V6Actor(cfg);self.critic=V6Critic(cfg) if cfg.model.v6_critic else None
        self.decoder=WeakDecoder(self.exec_cfg)

    def default_theta(self,state):
        head=self.actor.controller.v6_heads;m=state.capacities.shape[1];b=state.profits.shape[0]
        # Functional zero readout: trained head parameters cannot change D_port.
        device=state.profits.device
        # Preserve the original activation order, including softmax/gate rounding.
        ranges=head.ranges
        raw=state.profits.new_tensor([math.log(min(.999,max(.001,(v-lo)/(hi-lo)))/(1-min(.999,max(.001,(v-lo)/(hi-lo)))))
            for v,(lo,hi) in zip((1.,1.,.1,.2,.1,.1,.5,.5),ranges)],dtype=torch.float32).sigmoid()
        values=[(lo+(hi-lo)*raw[i]).expand(b) for i,(lo,hi) in enumerate(ranges)]
        cfg=self.exec_cfg
        mass=cfg.interaction_weight_max*torch.sigmoid(torch.tensor(math.log(cfg.combination_initial_weight/cfg.interaction_weight_max/(1-cfg.combination_initial_weight/cfg.interaction_weight_max)),device=device))
        mask=torch.ones(m,m,device=device,dtype=torch.bool).triu()
        upper=torch.zeros(b,m,m,device=device).masked_fill(~mask,-torch.inf).flatten(1).softmax(-1).view(b,m,m)*mass
        interaction=upper+upper.transpose(-1,-2)-torch.diag_embed(upper.diagonal(dim1=-2,dim2=-1))
        k=len(cfg.companion_scales)
        gate=torch.sigmoid(torch.tensor(math.log(cfg.combination_initial_weight/cfg.companion_weight_max/(1-cfg.combination_initial_weight/cfg.companion_weight_max)),device=device))
        companion=torch.zeros(b,k,device=device).softmax(-1)*cfg.companion_weight_max*gate
        prices=cfg.price_floor+(1-cfg.price_floor)*m*torch.zeros(b,m,device=device).softmax(-1)
        return StrategyTheta(prices,*values,interaction,companion)

    def policy(self,state):
        theta,feasibility=self.actor(state)
        return self.decoder(state,theta),theta,feasibility

    def value(self,state):
        if self.critic is None:raise ValueError("T8-A has no critic")
        return self.critic(state)

    def step_forward(self,state):
        policy,theta,feasibility=self.policy(state)
        return policy,self.value(state),theta,feasibility

    def episode(self,inst,num_samples,decode,override,return_trace):
        if decode not in ("sample","greedy") or num_samples<1:raise ValueError("Invalid v6 decode")
        p=inst.p.double().repeat_interleave(num_samples,0);w=inst.r.transpose(1,2).double().repeat_interleave(num_samples,0)
        c=inst.c.double().repeat_interleave(num_samples,0)
        env=MKPEnv(p,w,c);state=env.state;root=lp_relaxation(state,self.exec_cfg);state._root=(root.dual,root.bound)
        logp=p.new_zeros(len(p),dtype=torch.float32);entropy=logp.clone();decisions=logp.clone();sequence=[];trace=[]
        stat_sum=logp.new_zeros(len(THETA_STATISTICS));stat_count=0
        for step in range(inst.N):
            active=state.active
            if not bool(active.any()):break
            if override is None:policy,theta,_=self.policy(state)
            else:
                if isinstance(override,str):
                    if override!="fixed":raise ValueError("Only fixed v6 override is supported")
                    theta=self.default_theta(state)
                else:
                    values={n:torch.as_tensor(v,device=p.device,dtype=torch.float32) for n,v in override.items()}
                    theta=StrategyTheta(**{n:v.expand(len(p),*v.shape[1:]) if len(v)==1 else v.repeat_interleave(num_samples,0) for n,v in values.items()})
                policy=self.decoder(state,theta)
            action=policy.choose(stochastic=decode=="sample")
            logp=logp+policy.log_prob(action)*active
            # A follows the new project's item-PG raw categorical entropy;
            # B's ported collect_rollout keeps v6 normalized entropy unchanged.
            entropy=entropy+policy.entropy()*active
            decisions=decisions+active
            with torch.no_grad():
                stat_sum+=(theta.statistics()*active[:,None]).sum(0);stat_count+=int(active.sum())
            if return_trace:
                trace.append({"step":step,"tau_cap":(1-(state.remaining/state.capacities).mean(1)).detach().cpu().tolist(),
                              "theta":{n:getattr(theta,n).detach().cpu().tolist() for n in theta.__dataclass_fields__}})
            sequence.append(action);env.step(action)
        if bool(state.active.any()):raise RuntimeError("v6 construction exceeded N")
        return {"solution":state.selected,"profit":env.objective,"reward":(env.objective/root.bound.clamp_min(1e-8)).float(),
            "ub":root.bound,"log_likelihood":logp,"entropy":entropy/decisions.clamp_min(1),
            "n_reject":torch.zeros(len(p),device=p.device),"steps":len(sequence),
            "sequence":torch.stack(sequence,1) if sequence else torch.empty(len(p),0,dtype=torch.long,device=p.device),"trace":trace,
            "theta_statistics":dict(zip(THETA_STATISTICS,(stat_sum/max(stat_count,1)).tolist()))}
