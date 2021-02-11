import torch
from torch.nn import Parameter
import torch.nn.functional as F
#from torch.nn.parameter import Parameter
from torch.tensor import Tensor

from typing import List, Optional, Tuple
from typing_extensions import Final

# Cannot currently `jit` classes that inherit from Module directly. Have to `jit` an instance
class Siren(torch.nn.Module):
    intermediate_output:Final[bool]
    linear_output:Final[bool]
    layer_parameters:List[Tuple[Parameter, Parameter]]

    def __init__(self, in_features:int, out_features:int, hidden_layers:Optional[List[int]], \
                parameters:Optional[List[Tuple[Parameter, Parameter]]], intermediate_output:bool, linear_output:bool):
        super().__init__()
        
        # xor check, `!=` is xor for bool type, we want to check that the xor is false to raise error
        if (hidden_layers is None) == (parameters is None):
            raise ValueError("(`hidden_layers` is None) xor (`parameters` is None) must return False")

        omega_0:Final[float] = 30.
        self.intermediate_output = intermediate_output
        self.linear_output = linear_output
        self.layer_parameters = []
        
        if parameters is None:
            previous_layer_size:int = in_features
            first:bool = True
            hidden_layers.append(out_features)
            for layer_size in hidden_layers:
                half_range:float = 1./previous_layer_size
                if first:
                    self.layer_parameters.append((
                        Parameter(torch.empty((layer_size, previous_layer_size)).uniform_(-half_range * omega_0, half_range * omega_0), True),
                        Parameter(torch.empty((layer_size)).uniform_(-half_range, half_range), True)
                    ))
                else:
                    self.layer_parameters.append((
                        Parameter(torch.empty((layer_size, previous_layer_size)).uniform_(-half_range, half_range), True),
                        Parameter(torch.empty((layer_size)).uniform_(-half_range, half_range), True)
                    ))
                
                previous_layer_size = layer_size
                first = False
        else:
            for parameter_tuple in parameters:
                self.layer_parameters.append((
                    parameter_tuple[0].requires_grad_(True),
                    parameter_tuple[1].requires_grad_(True)
                ))

        # Manually add layer_parameters:List[Tuple[Parameter, Parameter]] to _parameters due to
        # Pytorch not automatically adding nested Parameter to _parameters
        for index, (weight, bias) in enumerate(self.layer_parameters):
            self.register_parameter('weights-{}'.format(index), weight)
            self.register_parameter('biases-{}'.format(index), bias)

    def forward(self, input:Tensor)->Tuple[Tensor, Optional[List[Tensor]]]:
        # TEST: Would not multipling by omega_0 during initialization help performance or multiplying by omega_0 for more layers?
        input:Tensor = input.clone().detach().requires_grad_(True)
        if self.intermediate_output:
            intermediate_output:List[Tensor] = []
            for weight, bias in self.layer_parameters[:-1]:
                input = F.linear(input, weight, bias).sin()
                intermediate_output.append(input.clone())
            input = F.linear(input, self.layer_parameters[-1][0], self.layer_parameters[-1][1])
            if self.linear_output:
                return input, intermediate_output
            return input.sin(), intermediate_output

        for weight, bias in self.layer_parameters[:-1]:
            input = F.linear(input, weight, bias).sin()
        if self.linear_output:
            return F.linear(input, self.layer_parameters[-1][0], self.layer_parameters[-1][1]), None
        return F.linear(input, self.layer_parameters[-1][0], self.layer_parameters[-1][1]).sin(), None

    @torch.jit.export
    def length_parameters(self)->int:
        return len(self.layer_parameters)

    @torch.jit.export
    def update_parameters(self, updated_parameters:List[Tuple[Parameter, Parameter]]):
        # TODO: Check to make sure updated parameters are being updated in `_parameters`
        if len(self.layer_parameters) != len(updated_parameters):
            raise ValueError(("Length of input parameter list not equal to length of stored parameter list\n"
                              "Length of input parameter list is {}\n"
                              "Length of stored parameter list is {}\n"
            ).format(len(updated_parameters), len(self.layer_parameters)))
        
        for index, ((m_weight, m_bias), (u_weight, u_bias)) in enumerate(zip(self.layer_parameters, updated_parameters)):
            if (m_weight.shape != u_weight.shape) or (m_bias.shape != u_bias.shape):
                raise ValueError(("Shapes of weight or bias at index {} of input weights and biases does not match stored weights and biases shapes\n"
                                  "Shape of input weight and bias are {}, {} respectively\n"
                                  "Shape of stored weight and bias are {}, {} respectively\n"
                ).format(index, u_weight.shape, u_bias.shape, m_weight.shape, m_bias.shape))

            m_weight = u_weight.clone().detach().requires_grad_(True)
            m_bias = u_bias.clone().detach().requires_grad_(True)

class Recurrent_Siren(torch.nn.Module):
    intermediate_output:bool
    linear_output:bool
    segment_parameters:List[List[Tuple[Parameter, Parameter]]]
    outgoing_segment_jumps:List[Tuple[int, List[int]]]
    incoming_segment_jumps:List[Tuple[int, List[int]]]

    # TODO: allow for sending in either only outgoing_segment_jumps or incoming_segment_jumps and deduce the other as needed

    # TODO: Fix init and make the forward pass so that it actually properly represents an unrolled recurrent siren with skip and feedback connections

    def __init__(self, in_parameters:int, out_parameters:int, segment_descriptions:Optional[List[List[int]]], parameters:Optional[List[List[Tuple[Parameter, Parameter]]]], \
                incoming_segment_jumps:Optional[List[Tuple[int, List[int]]]], outgoing_segment_jumps:Optional[List[Tuple[int, List[int]]]], \
                intermediate_output:bool, linear_output:bool):
        '''
        # TODO description

        Recurrent extension of Siren that allows both skip and feedback connections by describing the unrolled network. Recurrent connections are added together

        Parameters:

        - in_parameters:

        - out_parameter:

        - segment_descriptions:

        - parameters:

        - outgoing_segment_jumps: list of segments that the current segment indexed by its index in the outer list can forward its output to. 
        If all segments only forward to one other segment, then this is a feedforward network. \n
         - Note: No segment can forward to the first segment and the last segment cannot forward to anything. The incoming segment to the
            first segment is the input and the last segment forwards to the output(Thus, the last element is required to be an empty list if there are any segments)
         - Note: All segments beside the first and last segment must have both atleast one segment that forwards into it and atleast one segment that it forwards to
         - Note: All incoming segment connections to a segment have to be the same size but they don't have to be the same size as the segment
         - Note: Forward targets must be unique for a segment and in ascending order

        - incoming_segment_jumps

        - intermediate_output:

        - linear_output:
        '''
        super().__init__()

        # xor check to ensure that only one of these is None
        if (segment_descriptions is None) == (parameters is None):
            raise ValueError("(`segement_description` is None) xor (`parameters` is None) must return False")

        # xor check for None
        if (incoming_segment_jumps is None) == (outgoing_segment_jumps is None):
            raise ValueError("(incoming_segment_jumps is None) xor (outgoing_segment_jumps) must return False")

        # TODO Rewrite this description
        # check for any forwards to the initial segment
        # check if the last segment has any outbound jumps
        # check if that outbound jumps are unique for a segment and in ascending order
        # check if all segments beyond the first and last have both incoming and outgoing connections
        # check if the boundries between segments match up in size
        # Create list that keeps track of incoming jumps to current segment indexed by outer index
        if incoming_segment_jumps is None:
            incoming_segment_jumps = []
            num_segment_jumps:int = len(outgoing_segment_jumps)
            num_segments:int = 0
            if parameters is None:
                num_segments = len(segment_descriptions)
            else:
                num_segments = len(parameters)

            for current_jump_index, (current_segment_index, destinations) in enumerate(outgoing_segment_jumps):
                if current_segment_index == num_segments-1:
                    if current_jump_index != num_segment_jumps-1:
                        raise ValueError("Outgoing segment jump index {} corresponds to last segment without being last outgoing segment jump index".format(current_jump_index))
                    if destinations != []:
                        raise ValueError("Last outgoing segment jump's destinations is not an empty list")
                elif destinations == []:
                    raise ValueError("Outgoing segment jump index {} has an empty destinations list without being last outgoing segment jump index".format(current_jump_index))
                incoming_segment_jumps.append((current_segment_index, []))

            for current_jump_index, (_, destinations) in enumerate(outgoing_segment_jumps[:-1]):
                last_destination:int = current_jump_index
                for destination in destinations:
                    if destination <= last_destination:
                        raise ValueError("Outgoing segment jump index {0} must have strictly increasing destination jumps indices that are greater than {0}\ndestinations: {1}".format(
                            current_jump_index, destinations
                        ))
                    if destination > num_segment_jumps-1:
                        raise ValueError("Outgoing segment jump index {0} has destination greater than last segment jump index: {1}\ndestinations: {2}".format(
                            current_jump_index, num_segment_jumps-1, destinations
                        ))

                    incoming_segment_jumps[destination][1].append(current_jump_index)
                    last_destination = destination
            
            for current_jump_index, (current_segment_index, sources) in enumerate(incoming_segment_jumps):
                if current_segment_index == 0:
                    if current_jump_index != 0:
                        raise ValueError("Incoming segment jump index {} corresponds to first segment without being first incoming segment jump index".format(current_jump_index))
                    if sources != []:
                        raise ValueError("First incoming segment jump's sources is not an empty list")
                    continue
                if sources == []:
                    raise ValueError("Incoming segment jump index {} has an empty sources list without being first incoming segment jump index".format(current_jump_index))
                source_size:int = 0
                other_source_size:int = 0
                if parameters is None:
                    source_size = segment_descriptions[incoming_segment_jumps[sources[0]][0]][-1]
                else:
                    source_size = parameters[incoming_segment_jumps[sources[0]][0]][-1][1].shape[0]
                for other_source in sources[1:]:
                    if parameters is None:
                        other_source_size = segment_descriptions[incoming_segment_jumps[other_source][0]][-1]
                    else:
                        other_source_size = parameters[incoming_segment_jumps[other_source][0]][-1][1].shape[0]
                    if source_size != other_source_size:
                        raise ValueError(("Incoming segment jump index {} has sources of different outgoing sizes\nsources: {}\n"
                            "Size of segment output corresponding to source jump index {} is {}\n"
                            "Size of segment output corresponding to source jump index {} is {}"
                        ).format(current_jump_index, sources, sources[0], source_size, other_source, other_source_size))
        else:
            outgoing_segment_jumps = []
            num_segment_jumps:int = len(incoming_segment_jumps)
            num_segments:int = 0
            if parameters is None:
                num_segments = len(segment_descriptions)
            else:
                num_segments = len(parameters)

            for current_jump_index, (current_segment_index, sources) in enumerate(incoming_segment_jumps):
                if current_segment_index == 0:
                    if current_jump_index != 0:
                        raise ValueError("Incoming segment jump index {} corresponds to first segment without being first incoming segment jump index".format(current_jump_index))
                    if sources != []:
                        raise ValueError("First incoming segment jump's sources is not an empty list")
                elif sources == []:
                    raise ValueError("Incoming segment jump index {} has an empty sources list without being first incoming segment jump index".format(current_jump_index))
                outgoing_segment_jumps.append((current_segment_index, []))

            for current_jump_index, (_, sources) in enumerate(incoming_segment_jumps[1:]):
                last_source:int = current_jump_index
                for source in sources:
                    if source >= last_source:
                        raise ValueError("Incoming segment jump index {0} must have strictly increasing source jumps indices that are less than {0}\nsources: {1}".format(
                            current_jump_index, sources
                        ))
                    if source < 0:
                        raise ValueError("Incoming segment jump index {0} has source less than first segment jump index: {1}\nsources: {2}".format(
                            current_jump_index, 0, sources
                        ))

                    outgoing_segment_jumps[source][1].append(current_jump_index)
                    last_source = source
            
            for current_jump_index, (current_segment_index, destinations) in enumerate(outgoing_segment_jumps):
                if current_segment_index == num_segment_jumps-1:
                    if current_jump_index != num_segments-1:
                        raise ValueError("Outgoing segment jump index {} corresponds to last segment without being last outgoing segment jump index".format(current_jump_index))
                    if destinations != []:
                        raise ValueError("Last outgoing segment jump's destinations is not an empty list")
                    continue
                if destinations == []:
                    raise ValueError("Outgoing segment jump index {} has an empty destination list without being last outgoing segment jump index".format(current_jump_index))
                destination_size:int = 0
                other_destination_size:int = 0
                if parameters is None:
                    destination_size = segment_descriptions[incoming_segment_jumps[destinations[0]][0]][0]
                else:
                    destination_size = parameters[incoming_segment_jumps[destinations[0]][0]][0][1].shape[0]
                for other_destination in destinations[1:]:
                    if parameters is None:
                        other_destination_size = segment_descriptions[incoming_segment_jumps[other_source][0]][0]
                    else:
                        other_destination_size = parameters[incoming_segment_jumps[other_source][0]][0][1].shape[0]
                    if destination_size != other_destination_size:
                        raise ValueError(("Outgoing segment jump index {} has destinations of different incoming sizes\ndestinations: {}\n"
                            "Size of segment output corresponding to destination jump index {} is {}\n"
                            "Size of segment output corresponding to destination jump index {} is {}"
                        ).format(current_jump_index, destination, destinations[0], destination_size, other_destination, other_destination_size))

        
        omega_0:Final[float] = 30.
        self.intermediate_output = intermediate_output
        self.linear_output = linear_output
        self.segment_parameters = []
        self.outgoing_segment_jumps = outgoing_segment_jumps
        self.incoming_segment_jumps = incoming_segment_jumps

        if parameters is None:
            previous_layer_size = in_parameters
            for segment_index, segment in enumerate(segment_descriptions[:-1]):
                self.segment_parameters.append([])
                if segment_index != 0:
                    previous_layer_size = segment_descriptions[self.incoming_segment_jumps[segment_index][0]][-1]
                for layer_index, layer_size in enumerate(segment):
                    half_range:float = 1./previous_layer_size
                    if (segment_index == 0) and (layer_index == 0):
                        self.segment_parameters[-1].append((
                            Parameter(torch.empty((layer_size, previous_layer_size)).uniform_(-half_range * omega_0, half_range * omega_0), True),
                            Parameter(torch.empty((layer_size)).uniform_(-half_range, half_range), True)
                        ))
                    else:
                        self.segment_parameters[-1].append((
                            Parameter(torch.empty((layer_size, previous_layer_size)).uniform_(-half_range, half_range), True),
                            Parameter(torch.empty((layer_size)).uniform_(-half_range, half_range), True)
                        ))
                    
                    previous_layer_size = layer_size
        else:
            for segment in parameters:
                self.segment_parameters.append([])
                for parameter_tuple in segment:
                    self.segment_parameters[-1].append((
                        parameter_tuple[0].requires_grad_(True),
                        parameter_tuple[1].requires_grad_(True)
                    ))

        # Due to parameters being stored in List[List[Tuple[Parameter, Parameter]]], we have to manually add them to _parameter because pytorch cannot find them
        for segment_index, segment in enumerate(self.segment_parameters):
            for layer_index, parameter_tuple in enumerate(segment):
                self.register_parameter('weight-{}-{}'.format(segment_index, layer_index), parameter_tuple[0])
                self.register_parameter('bias-{}-{}'.format(segment_index, layer_index), parameter_tuple[1])

    def forward(self, input:Tensor)->Tuple[Tensor, Optional[List[Tensor]]]:
        input:Tensor = input.clone().detach().requires_grad_(True)
        # TODO complete