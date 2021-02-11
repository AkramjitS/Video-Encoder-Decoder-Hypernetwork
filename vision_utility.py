import torch
import torchvision
from torchvision.transforms.functional import resize, convert_image_dtype

from typing import Tuple, Optional

@torch.jit.script
def get_mgrid(sidelen:Tuple[int, int], start:float=0., end:float=1.)->torch.Tensor:
    '''
    Generates a flattened grid of (x,y,...) coordinates in a range of start to end, inclusive.
    sidelength: height and width(in that order) of the image
    start: start value for grid accross all dimensions
    end: end value for grid accross all dimensions
    '''
    tensors:List[torch.Tensor] = []
    for slen in sidelen:
        tensors.append(torch.linspace(start, end, steps=slen))
    mgrid = torch.stack(torch.meshgrid(tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid

@torch.jit.script
def get_image_tensor(image_path:str, sidelength:Tuple[int, int])->torch.Tensor:
    '''
    image_path: local path to image
    sidelength: height and width(in that order) of the image
    '''
    image: torch.Tensor = torchvision.io.read_image(image_path)
    image = resize(image, sidelength)
    #image = normalize(image, [.5], [.5])
    image = convert_image_dtype(image, torch.float)
    image = image.permute(1, 2, 0).view(-1, 3)
    return image

'''
# Example usage:
# 
# get image from location `forest/forest.jpg` of height 1271 and width 1920
# assign the corresponding images tensor to image and print its shape
# show the image by converting it to PIL

from PIL.ImageShow import show

image = get_image_tensor('forest/forest.jpg', (1271, 1920))
print(image.shape)
show(torchvision.transforms.functional.to_pil_image(image))
'''