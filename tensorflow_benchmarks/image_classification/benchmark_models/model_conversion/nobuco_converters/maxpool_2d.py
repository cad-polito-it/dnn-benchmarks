import torch

import tensorflow as tf
from tensorflow import keras

import numpy as np

from nobuco import converter, ChannelOrderingStrategy


@converter(
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.functional.max_pool2d,
    channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS,
)
def convert_max_pool2d(self: torch.nn.modules.pooling.MaxPool2d, x):

    if not self.dilation in [1, (1, 1)]:
        raise NotImplementedError(
            f"Conversion with dilation ({self.dilation}) different than (1,1) is not supported"
        )

    def func(x):
        if isinstance(self.padding, tuple):
            pad_h, pad_w = self.padding
        elif isinstance(self.padding, int):
            pad_h = self.padding
            pad_w = self.padding
        else:
            raise ValueError(f"Padding format ({self.padding}) not supported")
        # Pad simmetrically as in PyTorch
        x_padded = tf.pad(
            x,
            [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]],
            constant_values=-np.infty,
        )
        # Apply valid padding in pre-padded layer
        pooling = keras.layers.MaxPooling2D(
            self.kernel_size,
            strides=self.stride,
            padding="valid",
            data_format="channels_last",
        )

        x = pooling(x_padded)

        return x

    return func
