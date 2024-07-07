import torch
import torch.nn as nn
import argparse

import numpy as np

from benchmark_models.utils import (
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    get_loader,
    load_network,
)
import csv


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAYERS = (nn.Conv2d, nn.Linear)
BITWIDTH = 32


def parse_args():
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(
        description="Run Inferences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Dataset to use",
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--network-name",
        "-n",
        type=str,
        required=True,
        help="Target network",
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--injections-per-layer",
        type=int,
        default=100,
        help="Injections per layer",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Output Path",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=512,
        help="Batch Size",
    )
    parser.add_argument(
        "-w",
        "--bitwidth",
        type=int,
        default=32,
        help="Bitwidth",
    )
    # parser.add_argument(
    #    "--tensorflow",
    #    "--tf",
    #    help="Execute the network in TensorFlow. Convert it a conversion does not exists",
    #    action="store_true",
    # )
    parsed_args = parser.parse_args()

    return parsed_args


def random_weight(shape):
    coord = []
    for dim in shape:
        coord.append(np.random.randint(dim))
    return tuple(coord)


def main(args):
    device = torch.device(DEVICE)

    network = load_network(
        args.network_name,
        device,
        args.dataset,
    )
    network.eval()

    inj_id = 0
    injections = []
    # 27,layer1.0.conv1,"(5, 3, 2, 1)",12
    for name, module in network.named_modules():
        if isinstance(module, LAYERS):
            for _ in range(args.injections_per_layer):
                coords = random_weight(module.weight.shape)
                # bitpos = np.random.randint(args.bitwidth)
                bitpos = 30
                injections.append([inj_id, name, str(coords), bitpos])
                inj_id += 1
    with open(args.output_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(injections)


if __name__ == "__main__":
    main(parse_args())
