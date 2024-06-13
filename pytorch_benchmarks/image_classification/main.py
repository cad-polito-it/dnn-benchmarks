import torch
from utils import load_network, get_device, parse_args, get_loader


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
    )


    # Import inference manager only here to avoid importing pytorch for tensorflow users
    from utils import (
        PTInferenceManager
    )

    # Execute the fault injection campaign with the smart network
    inference_executor = PTInferenceManager(
        network=network,
        device=device,
        network_name=args.network_name,
        loader=loader,
    )

    # This function runs clean inferences on the golden dataset 
    inference_executor.run_inference(verbose=True)


if __name__ == "__main__":
    main(args=parse_args())