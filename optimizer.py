import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from typing import Dict, Tuple, Optional, List


@torch.jit.script
def gradient(y:torch.Tensor, x:torch.Tensor, grad_outputs:Optional[List[Optional[torch.Tensor]]]=None)->torch.Tensor:
    if grad_outputs is None:
        grad_outputs  = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(y)])
    grad = torch.autograd.grad([y], [x], grad_outputs=grad_outputs, create_graph=True)[0]
    if grad is None:
        raise ValueError("Gradient is of type None")
    return grad

@torch.jit.script
def divergence(y:torch.Tensor, x:torch.Tensor)->torch.Tensor:
    div : torch.Tensor = torch.zeros_like(y[...,0:1])
    for i in range(y.shape[-1]):
        grad_outputs : Optional[List[Optional[torch.Tensor]]] = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(y[..., i])])
        temp: Optional[torch.Tensor] = torch.autograd.grad([y[..., i]], [x], grad_outputs, create_graph=True)[0]
        if temp is None:
            raise ValueError("Gradient is of type None")
        div += temp[..., i:i+1]
    return div

@torch.jit.script
def laplace(y:torch.Tensor, x:torch.Tensor, grad:Optional[torch.Tensor]) -> torch.Tensor:
    if grad is None:
        grad = gradient(y, x)
    return divergence(grad, x)

@torch.jit.script
def total_mse(gt:Dict[str, torch.Tensor], stan:Optional[torch.Tensor], grad:Optional[torch.Tensor]=None, lapl:Optional[torch.Tensor]=None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    loss1:Optional[torch.Tensor] = F.mse_loss(stan,    gt['output'])    if stan is not None else None
    loss2:Optional[torch.Tensor] = F.mse_loss(grad,    gt['gradient'])  if grad is not None else None
    loss3:Optional[torch.Tensor] = F.mse_loss(lapl,    gt['laplacian']) if lapl is not None else None
    return loss1, loss2, loss3

@torch.jit.script
class EMA_params:
    model_parameters: List[Parameter]
    ma_parameters: List[Parameter]
    alpha:float

    @torch.no_grad()
    def __init__(self, parameters:List[Parameter], alpha:float=0.01):
        self.model_parameters = []
        self.ma_parameters = []
        for p in parameters:
            if p.requires_grad:
                self.model_parameters.append(p.detach())
                self.ma_parameters.append(p.detach())
        self.alpha = alpha
            
    @torch.no_grad()
    def update(self):
        for ma_layer, model_layer in zip(self.ma_parameters, self.model_parameters):
            ma_layer += self.alpha * (model_layer - ma_layer)
        
    @torch.no_grad()
    def copy_from_ma(self):
        for ma_layer, model_layer in zip(self.ma_parameters, self.model_parameters):
            model_layer.copy_(ma_layer)

@torch.jit.script            
class Perturb_Parameters:
    perturb_parameters: List[Parameter]
    mean: List[torch.Tensor]
    std: float

    @torch.no_grad()
    def __init__(self, parameters:List[Parameter], mean:float=0., std:float=1.0e-5):
        self.perturb_parameters = []
        for p in parameters:
            if p.requires_grad:
                self.perturb_parameters.append(p.detach())
        self.mean = []
        self.std = std
        for p in self.perturb_parameters:
            self.mean.append(mean * torch.ones_like(p))
    
    @torch.no_grad()
    def perturb(self):
        for p, m in zip(self.perturb_parameters, self.mean):
            p += torch.normal(mean=m, std=self.std)
            
@torch.jit.script
class shift_loss_params:
    initial_parameters: List[Parameter]
    current_parameters: List[Parameter]
    _num_parameters: int
    
    @torch.no_grad()
    def __init__(self, parameters:List[Parameter]):
        self.initial_parameters = []
        self.current_parameters = []
        self._num_parameters = 0
        for p in parameters:
            if p.requires_grad:
                self.initial_parameters.append(p.clone().detach())
                self.current_parameters.append(p.detach())
                self._num_parameters += p.numel()
        
    @torch.no_grad()
    def get_loss(self) -> float:
        ret:float = 0.
        for initial, current in zip(self.initial_parameters, self.current_parameters):
            ret += F.mse_loss(initial, current, reduction='sum')
        return ret

    def num_params(self) -> int:
        return self._num_parameters