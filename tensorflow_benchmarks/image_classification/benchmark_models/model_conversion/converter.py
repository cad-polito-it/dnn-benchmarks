import os
from benchmark_models.inference_tools.metric_evaluators import TopKAccuracy
from benchmark_models.model_conversion.args import parse_args
from benchmark_models.tf_utils import create_manipulated_model

from benchmark_models.utils import (
    get_device,
    get_loader,
    load_network,
)
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow import keras

import nobuco
from nobuco import ChannelOrder

# Keep this even if unused
from benchmark_models.model_conversion.nobuco_converters import maxpool_2d, sequential

from benchmark_models.inference_tools.pytorch_inference_manager import (
    PTInferenceManager,
)
from benchmark_models.inference_tools.tf_inference_manager import TFInferenceManager


class PrintShapeLayer(keras.layers.Layer):
    """
    Test layer to check if the output model can be manipulate in "Manipulation Testing"
    Prints the output shape of the layer in stdout
    """

    def __init__(self, prev_name="unknown", **kwargs):
        super(PrintShapeLayer, self).__init__(**kwargs)
        self.prev_name = prev_name

    def call(self, inputs):
        if hasattr(inputs, "shape"):
            # Print the desired message with the input shape
            print(f"Hello I'm {self.prev_name}, my shape is {inputs.shape}")
        else:
            # Print the desired message with the input shape
            print(f"Hello I'm {self.prev_name}, I have no shape")
        return inputs


def convert_pt_to_tf(network: Module, loader: DataLoader):
    sample_image, _ = next(iter(loader))

    print("Starting conversion from PyTorch to Keras")
    print(f"Using dataset sample image of shape: {sample_image.shape}")

    b, h, w, c = sample_image.shape
    with torch.no_grad():
        keras_model = nobuco.pytorch_to_keras(
            network,
            args=(sample_image,),
            kwargs=None,
            input_shapes={input: (None, h, w, c)},
            trace_shape=True,
            inputs_channel_order=ChannelOrder.TENSORFLOW,
            outputs_channel_order=ChannelOrder.TENSORFLOW,
        )
        print("Conversion from PyTorch to Keras complete.")
    return keras_model


def fake_layer_factory(layer):
    print(f"Layer Name: {layer.name} Type:{type(layer)}")
    if isinstance(layer, keras.Model):
        cloned_submodel = keras.models.clone_model(
            layer, clone_function=fake_layer_factory
        )
        cloned_submodel.set_weights(layer.get_weights())
        return cloned_submodel
    if isinstance(layer, keras.layers.ReLU):
        return keras.Sequential([keras.layers.ReLU(), PrintShapeLayer(layer.name)])

    cloned_layer = layer.__class__.from_config(layer.get_config())
    return cloned_layer


def main(args):
    print("Running nobuco converter")
    print(f"Converting network {args.network_name} to PyTorch")

    output_path = args.output_path or os.path.join(
        "models", "converted-tf", args.dataset, f"{args.network_name}.keras"
    )

    if args.output_path is None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    skip_conversion = os.path.exists(output_path) and not args.overwrite

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda, use_cuda=args.use_cuda)

    print(f"Using device {device}")

    # Load the network
    network = load_network(
        network_name=args.network_name, dataset_name=args.dataset, device=device
    )
    network.eval()

    _, conversion_loader = get_loader(
        dataset_name=args.dataset, batch_size=args.batch_size, permute_tf=False
    )

    if not skip_conversion:
        print(
            "STEP 1. [STARTING] Conversion from PyTorch to Keras using nobuco convertor"
        )
        with torch.no_grad():
            keras_model = convert_pt_to_tf(network, conversion_loader)
            print(
                "STEP 1. [COMPLETED] Conversion from PyTorch to Keras using nobuco convertor"
            )
        print("STEP 2. [STARTING] Saving converted model to .keras")
        keras_model.save(output_path)
        print("STEP 2. [COMPLETED] Saving converted model to .keras")
        print(
            f"Converted model saved to {output_path}. It can be loaded using keras.models.load_model('path/to/model')"
        )
    else:
        print(
            "STEP 1. [SKIPPED] Conversion from PyTorch to Keras using nobuco convertor. Model already present, to avoid skipping use --overwrite flag."
        )
        print(
            "STEP 2. [SKIPPED] Saving converted model to .keras. Model already present, to avoid skipping use --overwrite flag. "
        )

    print(f"STEP 3. [STARTED] Reloading Keras model from {output_path}.")
    reloaded_keras_model = keras.models.load_model(output_path)

    reloaded_keras_model.summary(expand_nested=True)

    _, validation_loader_tf = get_loader(
        dataset_name=args.dataset, batch_size=args.batch_size, permute_tf=True
    )

    _, validation_loader_pt = get_loader(
        dataset_name=args.dataset, batch_size=args.batch_size, permute_tf=False
    )

    print(f"STEP 3. [COMPLETED] Reloading Keras model from {output_path}.")
    if not args.skip_validation:
        if not args.skip_pt_validation:
            print("STEP 4. [STARTING] Validating PyTorch Model.")
            inference_executor = PTInferenceManager(
                network=network,
                network_name=args.network_name,
                device=device,
                loader=validation_loader_pt,
            )
            inference_executor.run_inference(save_outputs=False)
            metric = TopKAccuracy(k=1)
            accuracy = inference_executor.evaluate_metric(metric)
            print(f"Pytorch accuracy: {accuracy}")
            print("STEP 4. [COMPLETED] Validating PyTorch Model.")
        else:
            print("STEP 4. [SKIPPED] Validating PyTorch Model.")
        # Run inference for the TF converted network to compare the value
        print("STEP 5. [STARTING] Validating converted Keras Model.")

        tf_inference_executor = TFInferenceManager(
            network=reloaded_keras_model,
            network_name=args.network_name,
            loader=validation_loader_tf,
        )
        tf_inference_executor.run_inference(save_outputs=False)
        metric = TopKAccuracy(k=1)
        accuracy = tf_inference_executor.evaluate_metric(metric)
        print(f"Tensorflow accuracy: {accuracy}")
        print("STEP 5. [COMPLETED] Validating converted Keras Model.")
    else:
        print("STEP 4. [SKIPPED] Validating PyTorch Model.")
        print("STEP 5. [SKIPPED] Validating converted Keras Model.")

    if not args.skip_manipulation:
        print("STEP 6. [STARTING] Keras Model cloning and manipulation test.")

        def fake_injection_clone_function(layer, old_layer):
            if isinstance(
                layer,
                (
                    keras.layers.Conv2D,
                    keras.layers.BatchNormalization,
                    keras.layers.Dense,
                    keras.layers.Add,
                    keras.layers.Concatenate,
                    keras.layers.MaxPool2D,
                ),
            ):
                return keras.Sequential([layer, PrintShapeLayer(layer.name)])
            else:
                return None

        cloned_model = create_manipulated_model(
            reloaded_keras_model, fake_injection_clone_function, copy_weights=True
        )

        cloned_model.summary(expand_nested=True)
        tf_cloned_inference_executor = TFInferenceManager(
            network=cloned_model,
            network_name=args.network_name,
            loader=validation_loader_tf,
        )
        tf_cloned_inference_executor.run_inference()
        print("STEP 6. [COMPLETED] Keras Model cloning and manipulation test.")
    else:
        print("STEP 6. [SKIPPED] Keras Model cloning and manipulation test.")

    print("All done.")


if __name__ == "__main__":
    main(parse_args())
