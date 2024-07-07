import torch

from models.utils import load_ImageNet_validation_set, load_CIFAR10_datasets

from utils import get_loader, load_network, get_device, parse_args
from classes_core.error_simulator_keras import (
    create_injection_sites_layer_simulator,
    ErrorSimulator,
)


def main(args):
    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=False)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda, use_cuda=args.use_cuda)

    print(f"Using device {device}")

    _, loader = get_loader(
        network_name=args.network_name,
        batch_size=args.batch_size,
        permute_tf=args.tensorflow,
    )

    _, _, loader = load_CIFAR10_datasets(
        test_batch_size=args.batch_size, permute_tf=args.tensorflow
    )
    print(f"Using dataset: CIFAR10")

    if args.tensorflow:
        # Import inference manager only here to avoid importing tensorflow for pytorch users
        from benchmark_models.inference_tools.tf_inference_manager import (
            TFInferenceManager,
        )
        from tf_utils import load_converted_tf_network, create_manipulated_model
        import keras

        tf_network = load_converted_tf_network(args.network_name)

        # Execute the fault injection campaign with the smart network
        inference_executor = TFInferenceManager(
            network=tf_network, network_name=args.network_name, loader=loader
        )
        tf_network.summary(expand_nested=True)
        inference_executor.run_inference(max_inferences=2)

        def classes_factory(layer, old_layer):
            if layer.name == "re_lu_5":
                print(old_layer.output_shape)
                n, h, w, c = (None, 4, 4, 512)
                (
                    available_injection_sites,
                    masks,
                ) = create_injection_sites_layer_simulator(
                    5,
                    "relu",
                    str((1, c, h, w)),
                    str((1, h, w, c)),
                    models_folder="classes_models",
                )

                sim = ErrorSimulator(
                    available_injection_sites,
                    masks,
                    len(available_injection_sites),
                    [0],
                )
                return keras.Sequential([layer, sim])
            else:
                return None

        cloned_model = create_manipulated_model(tf_network, classes_factory)

        cloned_model.summary(expand_nested=True)

        # Execute the fault injection campaign with the smart network
        inference_executor = TFInferenceManager(
            network=cloned_model, network_name=args.network_name, loader=loader
        )

    else:
        # Import inference manager only here to avoid importing pytorch for tensorflow users

        from benchmark_models.inference_tools.pytorch_inference_manager import (
            PTInferenceManager,
        )

        # Load the network
        network = load_network(network_name=args.network_name, device=device)

        print(f"Using network: {args.network_name}")

        network.eval()
        # Execute the fault injection campaign with the smart network
        inference_executor = PTInferenceManager(
            network=network,
            device=device,
            network_name=args.network_name,
            loader=loader,
        )

    # This function runs clean inferences on the golden dataset
    inference_executor.run_inference()


if __name__ == "__main__":
    main(args=parse_args())
