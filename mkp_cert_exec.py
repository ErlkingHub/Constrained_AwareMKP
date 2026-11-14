"""Six-dimensional certified theta; fixed criteria and v6 categorical semantics."""
from dataclasses import dataclass
import torch
from mkp_cert_basis import cert_basis,NAMES,SIGNS
from mkp_v6exec import CandidatePolicy

CONSTANTS={'K-rc':(1,1,0,0,0),'K-bound':(.05,1,1,0,0),
           'K-profit':(1,1,0,1,0),'K-pair':(1,1,0,1,1)}
STATISTICS=tuple('pi_'+n for n in NAMES)+('temperature',)


@dataclass
class CertTheta:
    weights: torch.Tensor
    temperature: torch.Tensor

    def statistics(self):return torch.cat((self.weights,self.temperature[:,None]),-1)
    def detach(self):return CertTheta(self.weights.detach(),self.temperature.detach())


def constant_theta(name,batch=1,temperature=.2,device='cpu'):
    values=torch.as_tensor(CONSTANTS[name] if isinstance(name,str) else name,device=device,dtype=torch.float32)
    if values.shape!=(5,) or not bool(torch.isfinite(values).all()) or bool((values<0).any()) or float(values.sum())<=0:
        raise ValueError('cert weights require five finite nonnegative values and positive mass')
    return CertTheta((values/values.sum()).expand(batch,-1),torch.full((batch,),temperature,device=device))


def injected_theta(value,batch,temperature,device):
    if not isinstance(value,(dict,CertTheta)):return constant_theta(value,batch,temperature,device)
    weights=torch.as_tensor(value['weights'] if isinstance(value,dict) else value.weights,device=device,dtype=torch.float32)
    temp=torch.as_tensor(value.get('temperature',temperature) if isinstance(value,dict) else value.temperature,device=device,dtype=torch.float32).reshape(-1)
    if weights.ndim==1:weights=weights[None]
    if weights.ndim!=2 or weights.shape[1]!=5 or batch%len(weights) or batch%len(temp):raise ValueError('Invalid cert override batch/shape')
    weights=weights.repeat_interleave(batch//len(weights),0);temp=temp.repeat_interleave(batch//len(temp),0)
    if not bool(torch.isfinite(weights).all()) or bool((weights<0).any()) or not torch.allclose(weights.sum(-1),torch.ones(batch,device=device),rtol=0,atol=1e-5):
        raise ValueError('Injected cert weights must lie on the simplex')
    if not bool(torch.isfinite(temp).all()) or bool((temp<=0).any()):raise ValueError('Invalid cert temperature')
    return CertTheta(weights,temp)


class CertDecoder:
    def __init__(self,config):self.config=config
    def __call__(self,state,theta):
        basis=cert_basis(state,self.config)
        owners,items=state.feasible.nonzero(as_tuple=True)
        signed=basis.values[owners,items].float()*basis.values.new_tensor(SIGNS).float()
        scores=(signed*theta.weights[owners]).sum(-1)
        return CandidatePolicy(owners,items,scores/theta.temperature[owners],len(state.profits),
            score_terms={'score':scores,'basis':signed})
