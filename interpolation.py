import torch
from math import pow, log

@torch.jit.script
def interpolate_lin_lin(percent: float, start:float, end:float)->float:
    return (1 - percent) * start + percent * end

@torch.jit.script
def interpolate_log_lin(percent:float, start:float, end:float, base:float=10.)->float:
    m = log(end, base) - log(start, base)
    b = log(start, base)
    return pow(base, (m * percent + b))