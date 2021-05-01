import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, SGD
from torchvision.transforms.functional import to_pil_image, resize

from vision_utility import get_mgrid, get_image_tensor
from inner_config import *
from modified_siren import Siren
#from optimizer import Perturb_Parameters

from temporary_siren import SineLayer, JumpNetwork

from typing import Tuple, List, Optional, Dict
from typing_extensions import Final

from math import tau, sqrt

from PIL.ImageShow import show

# Replace alpha with a scheduler or something else that could do its job
def train_image(training_parameters:Dict[str, Optional[str]], model:torch.nn.Module, grid:Tensor, image:Tensor, optimizer:torch.optim.Optimizer, alpha:Dict[str, float])\
    -> Tuple[torch.nn.Module, List[float], List[float]]:
    '''
    Train a model to fit to the given image

    Parameters:

    - training_parameters: ['total_steps', 'steps_between_summary', 'batch_size']

    - model: the model to train. This model must already be `jit`ed. Must return two outputs: 1. The model output, 2. Optional: The preactivation outputs of the hidden layers

    - grid: The coordinates of the image

    - image: The image as a tensor

    - optimizer: optimizer hooked on the models parameters

    - alpha: ['output', 'hidden_linear']: weight parameters for mixing multiple losses

    Return:
        model, losses unweighted, losses weighted

    '''

    if training_parameters['total_steps'] is None:
        raise ValueError("training_parameters['total_steps'] is None")
    if training_parameters['batch_size'] is None:
        raise ValueError("training_parameters['batch_size'] is None")

    # Setup model, grid, and image for cuda
    model = model.train()
    model = model.cuda()
    grid = grid.requires_grad_(False).cuda()
    image = image.requires_grad_(False).cuda()

    # Setup pertubation noise to training
    #pertubation = Perturb_Parameters(model.parameters())

    # Keep track of loss over training
    running_loss_unweighted:List[float] = []
    running_loss_weighted:List[float] = []

    image_size:int = 1
    for e in image.shape:
        image_size *= e

    # Loop over training iterations
    for iteration_step in range(0, training_parameters['total_steps'] + 1):

        # Keep track of loss over iteration
        iteration_loss_unweighted:float = 0.
        iteration_loss_weighted:float = 0.

        iteration_image_loss:float = 0.
        iteration_hidden_linear_loss:float = 0.

        # Loop over batches'
        batch_size = training_parameters['batch_size']
        for batch_index in range(grid.shape[0] // batch_size):
            # Setup batch input and correct output
            batch_grid:Tensor = grid[batch_index * batch_size : (batch_index + 1) * batch_size, :]
            batch_image:Tensor = image[batch_index * batch_size : (batch_index + 1) * batch_size, :]

            # Reset model gradients
            optimizer.zero_grad()

            # Get outputs of model
            batch_model_image, batch_model_hidden_linears = model(batch_grid)

            # Calculate losses
            batch_image_loss:Tensor = F.mse_loss(batch_model_image, batch_image)
            batch_hidden_linear_loss:Tensor = torch.zeros((1,), device='cuda')
            if batch_model_hidden_linears is not None:
                for batch_hidden_linear in batch_model_hidden_linears:
                    batch_hidden_linear_loss += batch_hidden_linear.square().sum()
            batch_loss_unweighted = batch_image_loss + batch_hidden_linear_loss
            batch_loss_weighted = alpha['output'] * batch_image_loss + alpha['hidden_linear'] * batch_hidden_linear_loss

            # Backprop using optimizer
            batch_loss_weighted.backward()
            optimizer.step()

            # Update pertubations
            #pertubation.perturb()

            # Keep track of losses for summary
            iteration_loss_unweighted += batch_loss_unweighted.detach().item()
            iteration_loss_weighted += batch_loss_weighted.detach().item()
            iteration_image_loss += batch_image_loss.detach().item()
            iteration_hidden_linear_loss += batch_hidden_linear_loss.detach().item()

        # Loop over remaining elements
        if grid.shape[0] % batch_size != 0:
            # Setup batch input and correct output
            batch_grid:Tensor = grid[-(grid.shape[0] % batch_size):, :]
            batch_image:Tensor = image[-(grid.shape[0] % batch_size):, :]

            # Reset model gradients
            optimizer.zero_grad()

            # Get outputs of model
            batch_model_image, batch_model_hidden_linears = model(batch_grid)

            # Calculate losses
            batch_image_loss:Tensor = F.mse_loss(batch_model_image, batch_image)
            batch_hidden_linear_loss:Tensor = torch.zeros((1,), device='cuda')
            if batch_model_hidden_linears is not None:
                for batch_hidden_linear in batch_model_hidden_linears:
                    batch_hidden_linear_loss += batch_hidden_linear.square().sum()
            batch_loss_unweighted = batch_image_loss + batch_hidden_linear_loss
            batch_loss_weighted = alpha['output'] * batch_image_loss + alpha['hidden_linear'] * batch_hidden_linear_loss

            # Backprop using optimizer
            batch_loss_weighted.backward()
            optimizer.step()

            # Update pertubations
            #pertubation.perturb()

            # Keep track of losses for summary
            iteration_loss_unweighted += batch_loss_unweighted.detach().item()
            iteration_loss_weighted += batch_loss_weighted.detach().item()
            iteration_image_loss += batch_image_loss.detach().item()
            iteration_hidden_linear_loss += batch_hidden_linear_loss.detach().item()

        # Update running weighted and unweigthted loss
        running_loss_unweighted.append(iteration_loss_unweighted)
        running_loss_weighted.append(iteration_loss_weighted)

        # Display summary if requested and is summary iteration
        if (training_parameters['steps_between_summary'] is not None) and (iteration_step % training_parameters['steps_between_summary'] == 0):
            print("Step: %d" % (iteration_step))
            print("Loss[image]{unweighted: %f, weighted: %f}" % (iteration_image_loss, iteration_image_loss * alpha['output']))
            print("Loss[hidden_linears]{unweighted: %f, weighted: %f}" % (iteration_hidden_linear_loss, iteration_hidden_linear_loss * alpha['hidden_linear']))
            print("Loss[total]{unweighted: %f, weighted:%f}" % (iteration_loss_unweighted, iteration_loss_weighted))
            print("Per pixel per channel image loss: %2.8E" % (iteration_image_loss / image_size))

        # Save model if requested and is save iteration
        if (training_parameters['steps_between_save'] is not None) and (iteration_step % training_parameters['steps_between_save'] == 0):
            #model.save('.model_saves/Siren/%2.8E_%d.pt' % (iteration_image_loss / image_size, iteration_step))
            torch.save(model, '.model_saves/SirenLarge14Layer/%2.8E_%d.pt' % (iteration_image_loss / image_size, iteration_step))

    return model, running_loss_unweighted, running_loss_weighted


# Use wandb tools to do hyperparameter search
if __name__ == '__main__':
    #model = Siren(*config_model)
    #model = torch.jit.script(model)

    # test different scaling constants
    #fourier = torch.normal(torch.zeros((2, 256)), 1.)# * 10.
    #fourier = fourier.cuda()
    fourier = None

    '''
    section0 = torch.nn.Sequential(
        SineLayer(64, 64, 30./64, 1.),
        #SineLayer(2, 64, 30./2, 1.),
        SineLayer(64, 64, 30./64, 1.),
        SineLayer(64, 64, 30./64, 1.),
    )
    section1 = torch.nn.Sequential(
        SineLayer(64, 64, 30./64, 1.),
        SineLayer(64, 64, 30./64, 1.),
        SineLayer(64, 64, 30./64, 1.),
        SineLayer(64, 64, 30./64, 1.),
    )
    section2 = torch.nn.Sequential(
        SineLayer(64, 64, 30./64, 1.),
        SineLayer(64, 64, 30./64, 1.),
        SineLayer(64, 64, 30./64, 1.),
    )
    section3 = torch.nn.Sequential(
        SineLayer(64, 64, 30./64, 1.),
        SineLayer(64, 64, 30./64, 1.),
        SineLayer(64, 64, 30./64, 1.),
    )
    section4 = torch.nn.Sequential(
        SineLayer(64, 3, 30./64, 1.),
    )
    '''
    sectionfourier = torch.nn.Sequential(
        #nn.Linear(512, 512, True), nn.ReLU(), # 1
        #nn.Linear(512, 256, True), nn.ReLU(), # 2
        #nn.Linear(256, 256, True), nn.ReLU(), # 3
        #nn.Linear(256, 256, True), nn.ReLU(), # 4
        #nn.Linear(256, 128, True), nn.ReLU(), # 5
        #nn.Linear(128, 128, True), nn.ReLU(), # 6
        #nn.Linear(128, 128, True), nn.ReLU(), # 7
        #nn.Linear(128, 128, True), nn.ReLU(), # 8
        #nn.Linear(128, 64, True), nn.ReLU(), # 9
        #nn.Linear(64, 64, True), nn.ReLU(), # 10
        #nn.Linear(64, 64, True), nn.ReLU(), # 11
        #nn.Linear(64, 64, True), nn.ReLU(), # 12
        #nn.Linear(64, 64, True), nn.ReLU(), # 13
        #nn.Linear(64, 3, True), #nn.Sigmoid(), # 14

        SineLayer(2, 512, 30./2, sqrt(1./2)),            # 1
        SineLayer(512, 512, sqrt(6./512), sqrt(1./512)), # 2
        SineLayer(512, 256, sqrt(6./512), sqrt(1./512)), # 3
        SineLayer(256, 256, sqrt(6./256), sqrt(1./256)), # 4
        SineLayer(256, 256, sqrt(6./256), sqrt(1./256)), # 5
        SineLayer(256, 128, sqrt(6./256), sqrt(1./256)), # 6
        SineLayer(128, 128, sqrt(6./128), sqrt(1./128)), # 7
        SineLayer(128, 128, sqrt(6./128), sqrt(1./128)), # 8
        SineLayer(128, 128, sqrt(6./128), sqrt(1./128)), # 9
        SineLayer(128, 64, sqrt(6./128), sqrt(1./128)),  # 10
        SineLayer(64, 64, sqrt(6./64), sqrt(1./64)),     # 11
        SineLayer(64, 64, sqrt(6./64), sqrt(1./64)),     # 12
        SineLayer(64, 64, sqrt(6./64), sqrt(1./64)),     # 13
        SineLayer(64, 3, sqrt(6./64), sqrt(1./64)),      # 14
        
        #SineLayer(2,  64,       30./2, sqrt(1./2) ), # 1
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 2
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 3
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 4
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 5
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 6
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 7
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 8
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 9
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 10
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 11
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 12
        #SineLayer(64, 64, sqrt(6./64), sqrt(1./64)), # 13
        #SineLayer(64, 3,  sqrt(6./64), sqrt(1./64))  # 14
        
    )

    section_list:List[torch.nn.Module] = [sectionfourier]#[section0, section1, section2, section3, section4]

    model = JumpNetwork(section_list, fourier)


    if config_optimizer[0] == 'Adam':
        optimizer = Adam(model.parameters(), **config_optimizer[1])
    elif config_optimizer[0] == 'SGD':
        optimizer = SGD(model.parameters(), **config_optimizer[1])
    else:
        raise ValueError("Optimizer[0] not found. Got %s" % (config_optimizer[0]))

    grid = get_mgrid(config_image[1])
    image = get_image_tensor(config_image[0], config_image[1])

    model, _, _ = train_image(config_training_parameters, model, grid, image, optimizer, config_alpha)
    #model = torch.jit.load('.model_saves/Siren/1.48716288E-10_100000.pt')

    with torch.no_grad():
        grid = grid.cuda()
        model = model.cuda().eval()

        model_image, _ = model(grid)


        show(to_pil_image(model_image.view(*config_image[1], 3).permute(2, 0, 1)))