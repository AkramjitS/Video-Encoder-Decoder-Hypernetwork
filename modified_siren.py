import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from optimizer import gradient, divergence, laplace


class Expanded_SineLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30., linear_output = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear_output = linear_output
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                half_range = 1. / self.in_features
                self.linear.weight.uniform_(-half_range, half_range)
            else:
                half_range = np.sqrt(6. / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-half_range, half_range)
        
    def forward(self, input):
        linear = self.omega_0 * self.linear(input)
        if self.linear_output:
            self.linear_output_vector = linear
        return torch.sin(self.omega_0 * self.linear(input))
    
    def get_linear_output(self):
        if self.linear_output:
            return self.linear_output_vector.squeeze()
        return None
    
class Expanded_Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers=[], outermost_linear=False, omega_0=30., linear_output=False, linear_output_included = False):
        super().__init__()
        self.omega_0 = omega_0
        
        self.net = []
        old = in_features
        is_first = True
        self.linear_output = linear_output
        
        for out in hidden_layers:
            self.net.append(Expanded_SineLayer(old, out, is_first=is_first, omega_0 = omega_0, linear_output = linear_output))
            is_first = False
            old = out

        if outermost_linear:
            final_linear = nn.Linear(old, out_features)
            
            with torch.no_grad():
                half_range = np.sqrt(6. / in_features) / self.omega_0
                final_linear.weight.uniform_(-half_range, half_range)
                
            self.net.append(final_linear)
        else:
            self.net.append(Expanded_SineLayer(old, out_features,
                                      is_first=False, omega_0=omega_0, linear_output = linear_output))
        
        if linear_output and linear_output_included:
            self.inner_layers = self.net
        elif linear_output:
            self.inner_layers = self.net[:-1]
        else:
            self.inner_layers = None
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, get_gradient=False, get_laplacian=False):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output= self.net(coords).squeeze()
        
        if get_gradient:
            grad = gradient(output, coords).squeeze()
        else:
            grad = None
        
        if get_laplacian:
            if get_gradient:
                lapl = divergence(grad, coords).squeeze()
            else:
                lapl = laplace(output, coords).squeeze()
        else:
            lapl = None
        
        if not self.linear_output:
            return output, grad, lapl
        return output, grad, lapl, [layer.get_linear_output() for layer in self.inner_layers]
            