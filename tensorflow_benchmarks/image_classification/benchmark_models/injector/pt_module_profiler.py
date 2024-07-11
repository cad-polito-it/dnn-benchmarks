from typing import Optional, Sequence, Dict, List

import torch
import torch.nn as nn

from torch.utils.hooks import RemovableHandle



DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def profile_weight_shape(
    network: nn.Module, 
    device=DEFAULT_DEVICE,
)-> Dict[str, List[int]]:
    shape_index = {}

    for name, module in network.named_modules():
        if hasattr(module, 'weight'):
            shape_index[name] = list(module.weight.size())

    return shape_index


def profile_output_shape(
    network: nn.Module,
    input_data: Optional[torch.Tensor] = None,
    input_shape: Optional[Sequence[int]] = None,
    device=DEFAULT_DEVICE,
) -> Dict[str, List[int]]:

    shape_index = {}

    def _make_shape_profile_hook(name):
        def _shape_profile_hook(module, input, output):
            shape_index[name] = list(output.shape)
            # Do not modify the output
            return output

        return _shape_profile_hook


    if input_data and not input_shape:
        input_shape = input_data.shape
        input_data = input_data.to(device)
    elif not input_data and input_shape:
        input_data = torch.normal(0.0, 1.0, input_shape).to(device)
    else:
        raise ValueError("One and only one between input_data and input_shape must be specified.")
    
    network.to(device)

    hook_handles: List[RemovableHandle] = []

    try:
        for name, module in network.named_modules():
            handle = module.register_forward_hook(_make_shape_profile_hook(name))
            hook_handles.append(handle)
        network(input_data)
    finally:
        for handle in hook_handles:
            handle.remove()

    return shape_index
