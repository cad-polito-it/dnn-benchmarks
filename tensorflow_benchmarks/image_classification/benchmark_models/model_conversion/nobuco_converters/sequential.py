from benchmark_models.tf_utils import create_manipulated_model

import torch

import tensorflow as tf
from tensorflow import keras

import nobuco
from nobuco import ChannelOrder, converter, ChannelOrderingStrategy


@converter(
    torch.nn.Sequential,
    channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER,
)
def convert_Sequential(self, x):
    """
    Convert a `nn.Sequential` layer to a `keras.Model` where the layers belonging
    to pytorch model are concatenated.

    We avoid to use `keras.Sequential` since it is less flexible to modify models with
    that layer for adding fault injectors.
    """
    # Arguments for converting submodules
    kwargs = {"return_outputs_pt": True, "trace_shape": True}
    # Variable used for tracking pytorch intermediate output inside the Sequential module
    temp_out = x
    # Collect the converted layers of the sequential
    tf_layer_list = []

    for i, module in enumerate(self.children()):
        # In order, One by one convert the members of the sequential
        seq_layer, seq_out = nobuco.pytorch_to_keras(
            module,
            input_shapes={input: tuple([None] + list(temp_out.shape[1:]))},
            args=(temp_out,),
            inputs_channel_order=ChannelOrder.TENSORFLOW,
            outputs_channel_order=ChannelOrder.TENSORFLOW,
            **kwargs,
        )
        # Update the intermediate output of the current pytorch layer, to be used in the next
        temp_out = seq_out
        tf_layer_list.append(seq_layer)  # add the converted layer

    def func(inp):
        # Build the graph in keras by iterating all the layers in the list
        # inp is a keras tensor, input of the sequential block
        temp = inp
        for layer in tf_layer_list:
            temp = layer(temp)
        return temp

    return func
