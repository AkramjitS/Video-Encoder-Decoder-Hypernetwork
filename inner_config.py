import torch

from typing import Dict, Optional, List, Tuple

config_model = (2, 3, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    None, False, False
)
config_image = ('forest/forest.jpg', (1271, 1920))
#config_image = ('forest/forest.jpg', (256, 387))
config_optimizer = ('Adam', {'lr':1.0e-4})
config_training_parameters:Dict[str, Optional[int]] = {
    'total_steps':5_000, 
    'steps_between_summary':100,
    'batch_size': 512 * 512,
    'steps_between_save':100}
config_alpha = {'output':1., 'hidden_linear':1.0e-2}
