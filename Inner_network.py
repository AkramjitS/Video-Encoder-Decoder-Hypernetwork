import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, SGD
from torchvision.transforms.functional import to_pil_image, resize

from vision_utility import get_mgrid, get_image_tensor
from inner_config import *
from modified_siren import Siren
#from optimizer import Perturb_Parameters

from typing import Tuple, List, Optional, Dict
from typing_extensions import Final

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
            print("Per pixel per channel image loss: %2.8E" % (image_size))

        # Save model if requested and is save iteration
        if (training_parameters['steps_between_save'] is not None) and (iteration_step % training_parameters['steps_between_save'] == 0):
            model.save('.model_saves/Siren/%2.8E_%d.pt' % (image_size, iteration_step))

    return model, running_loss_unweighted, running_loss_weighted

if __name__ == '__main__':
    model = Siren(*config_model)
    model = torch.jit.script(model)
    if config_optimizer[0] == 'Adam':
        optimizer = Adam(model.parameters(), **config_optimizer[1])
    elif config_optimizer[0] == 'SGD':
        optimizer = SGD(model.parameters(), **config_optimizer[1])
    else:
        raise ValueError("Optimizer[0] not found. Got %" % (config_optimizer[0]))

    grid = get_mgrid(config_image[1])
    image = get_image_tensor(config_image[0], config_image[1])

    model, _, _ = train_image(config_training_parameters, model, grid, image, optimizer, config_alpha)

    with torch.no_grad():
        grid = grid.cuda()
        model = model.cuda().eval()

        model_image, _ = model(grid)


        show(to_pil_image(model_image.view(*config_image[1], 3).permute(2, 0, 1)))