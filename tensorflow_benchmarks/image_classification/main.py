import torch

from benchmark_models.utils import load_network, get_device, parse_args, get_loader


def main(args):
    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=False)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda, use_cuda=args.use_cuda)

    print(f"Using device {device}")

    # Load the network
    network = load_network(
        network_name=args.network_name, device=device, dataset_name=args.dataset
    )

    print(f"Using network: {args.network_name}")

    _, loader = get_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        permute_tf=args.tensorflow,
    )

    # Import inference manager only here to avoid importing tensorflow for pytorch users
    from benchmark_models.inference_tools.tf_inference_manager import (
        TFInferenceManager,
    )
    from benchmark_models.tf_utils import load_converted_tf_network

    tf_network = load_converted_tf_network(args.network_name, args.dataset)

    # Execute the fault injection campaign with the smart network
    inference_executor = TFInferenceManager(
        network=tf_network, network_name=args.network_name, loader=loader
    )

    inference_executor.run_clean()

if __name__ == '__main__':
    main(parse_args())