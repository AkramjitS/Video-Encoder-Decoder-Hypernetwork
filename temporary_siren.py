import torch
import torch.nn.functional as F
from torch.tensor import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from typing import List, Tuple, Optional

from math import tau

class SineLayer(Module):
    weight:Parameter
    bias:Parameter
    def __init__(self, in_features:int, out_features:int, half_range_weight:float, half_range_bias:float):
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)).uniform_(-half_range_weight, half_range_weight), True)
        self.bias = Parameter(torch.empty((out_features)).uniform_(-half_range_bias, half_range_bias), True)

    def forward(self, input:Tensor)->Tensor:
        return F.linear(input, self.weight, self.bias).sin()

class JumpNetwork(Module):
    modules:List[Module]
    fourier:Optional[Tensor]
    def __init__(self, modules:List[Module], fourier:Optional[Tensor]):
        super().__init__()
        self.modules = modules
        self.fourier = fourier

        for index, module in enumerate(self.modules):
            self.add_module('module-{}'.format(index), module)

    def forward(self, input:Tensor)->Tuple[Tensor, Tensor]:
        '''
        # temp8Siren
        # 0 -> 1 |-> 3 |-> 2 -> 3 -> 4
        #        |->   | 
        input = self.modules[0](input)
        input = self.modules[1](input)
        skip = input.clone()
        input = self.modules[3](input)
        input = input + skip
        input = self.modules[2](input)
        input = self.modules[3](input)
        return self.modules[4](input), None
        '''

        '''
        # temp7Siren
        # 0 -> 1 |-> 3      |-> 4 ?
        #        |-> 2 -> 3 |
        input = self.modules[0](input)
        input = self.modules[1](input)
        jump = input.clone()
        jump = self.modules[2](jump)
        jump = self.modules[3](jump)
        input = self.modules[3](input)
        input = input + jump
        return self.modules[4](input), None
        '''

        '''
        # temp6Siren
        # 0 -> 1 -> 3 -> 2 -> 3 -> 4
        input = self.modules[0](input)
        input = self.modules[1](input)
        input = self.modules[3](input)
        input = self.modules[2](input)
        input = self.modules[3](input)
        return self.modules[4](input), None
        '''

        '''
        # temp5Siren
        # 0 |-> 1 ->   |-> 3 -> 4  ?
        #   |-> 2 -> 1 |
        input = self.modules[0](input)
        input_jump = input.clone()
        input_jump = self.modules[2](input_jump)
        input_jump = self.modules[1](input_jump)
        input = self.modules[1](input)
        input = input + input_jump
        input = self.modules[3](input)
        return self.modules[4](input), None
        '''

        '''
        # temp4Siren
        # 0 -> 1 |-> 2 -> 1 |-> 3 -> 4 ?
        #        |->        |
        input = self.modules[0](input)
        input = self.modules[1](input)
        skip = input.clone()
        input = self.modules[2](input)
        input = self.modules[1](input)
        input = input + skip
        input = self.modules[3](input)
        return self.modules[4](input), None
        '''

        '''
        # temp3Siren
        # 0 -> 1 -> 2 -> 1 -> 3 -> 4 ?
        input = self.modules[0](input)
        input = self.modules[1](input)
        input = self.modules[2](input)
        input = self.modules[1](input)
        input = self.modules[3](input)
        return self.modules[4](input), None
        '''
        
        '''
        # temp2Siren
        # 0 |-> 1 ->   |-> 3 -> 3 -> 4  ?
        #   |-> 2 -> 1 |
        input = self.modules[0](input)
        input_jump = input.clone()
        input_jump = self.modules[2](input_jump)
        input_jump = self.modules[1](input_jump)
        input = self.modules[1](input)
        input = input + input_jump
        input = self.modules[3](input)
        input = self.modules[3](input)
        return self.modules[4](input), None
        '''

        '''# tempSiren
        input = self.modules[0](input)
        input = self.modules[1](input)
        input = self.modules[1](input)
        input = self.modules[2](input)
        input = self.modules[2](input)
        input = self.modules[3](input)
        input = self.modules[3](input)
        return self.modules[4](input), None
        '''

        '''
        # origSiren
        input = self.modules[0](input)
        input = self.modules[1](input)
        input = self.modules[2](input)
        input = self.modules[3](input)
        return self.modules[4](input), None
        '''

        
        # OrigExpandedSiren
        return self.modules[0](input), None
        

        '''
        # fourierOrigSiren
        if self.fourier is None:
            raise ValueError("Fourier tensor expected, got None")
        input = tau * torch.mm(input, self.fourier)
        input = torch.cat((input.cos(), input.sin()), dim=1)
        return self.modules[0](input), None
        '''
        

        '''
        # fourierOrigExpandedSiren
        if self.fourier is None:
            raise ValueError("Fourier tensor expected, got None")
        input = tau * torch.mm(input, self.fourier)
        input = torch.cat((input.cos(), input.sin()), dim=1)
        return self.modules[0](input), None
        '''

        '''
        input = self.modules[0](input)
        #skip1 = input.clone()
        input = self.modules[1](input)
        input = self.modules[2](input)
        #input = input + skip1
        input = self.modules[1](input)
        input = self.modules[2](input)
        #skip2 = input.clone()
        input = self.modules[3](input)
        input = self.modules[2](input)
        #input = input + skip2
        input = self.modules[3](input)
        return self.modules[4](input), None
        '''
