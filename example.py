import torch
from fann.benchmark_models.utils import load_network, get_device, parse_args, get_loader

def tensorflow_clean_inference(tf_network, args, loader):
    print("Going with tensorflow")
    from fann.tf import TFInferenceManager

    from fann.tf.utils import load_converted_tf_network

    tf_network = load_converted_tf_network(args.network_name, args.dataset)

    # Execute the fault injection campaign with the smart network
    inference_executor = TFInferenceManager(
        network=tf_network, network_name=args.network_name, loader=loader
    )

    # This function runs clean inferences on the golden dataset
    inference_executor.run_inference()

def pytorch_clean_inference(network, args, loader):
    print("Going with pytorch")
    from fann.pt import PTInferenceManager

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
    args=parse_args()

    print("Starting")
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

    if args.tensorflow:
        tensorflow_clean_inference(network, args, loader)
    else:
        pytorch_clean_inference(network, args, loader)