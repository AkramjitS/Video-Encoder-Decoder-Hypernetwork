import torch
import torch.nn as nn
import torch.nn.functional as F


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class EMA_params:
    @torch.no_grad()
    def __init__(self, params, alpha=0.001, only_grad_params=False):
        if only_grad_params:
            self.model = [p.detach() for p in params if p.requires_grad_(True)]
            self.ma = [p.clone() for p in params if p.requires_grad_(True)]
        else:
            self.model = [p.detach() for p in params]
            self.ma = [p.clone() for p in params]
        self.alpha = alpha
            
    @torch.no_grad()
    def update(self):
        for ma_layer, model_layer in zip(self.ma, self.model):
            ma_layer += self.alpha * (model_layer - ma_layer)
        
    @torch.no_grad()
    def copy_from_ma(self):
        for ma_layer, model_layer in zip(self.ma, self.model):
            model_layer.copy_(ma_layer)
            
class perturb_params:
    @torch.no_grad()
    def __init__(self, params, mean=0., std=1.0e-5):
        self.params = [layer.detach() for layer in params]
        self.mean = []
        self.std = std
        for layer in self.params:
            self.mean.append(mean * torch.ones(layer.size(), device='cuda'))
    
    @torch.no_grad()
    def perturb(self):
        for i, layer in enumerate(self.params):
            layer += torch.normal(mean=self.mean[i], std=self.std)
            
class shift_loss_params:
    
    @torch.no_grad()
    def __init__(self, params):
        self.initial = [layer.clone() for layer in params]
        self.params = [layer.detach() for layer in params]
        self._num_params = sum(p.numel() for p in self.initial)
        
    @torch.no_grad()
    def get_loss(self):
        return sum([F.mse_loss(layer, self.initial[i], reduction='sum') for i, layer in enumerate(self.params)])
    
    def num_params(self):
        return self._num_params