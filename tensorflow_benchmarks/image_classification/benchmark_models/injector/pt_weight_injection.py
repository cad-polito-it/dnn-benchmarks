from dataclasses import dataclass
import csv
from typing import Tuple
from benchmark_models.inference_tools.inference_manager import InferenceManager
from benchmark_models.inference_tools.metric_evaluators import TopKAccuracy
from benchmark_models.inference_tools.pytorch_inference_manager import (
    PTInferenceManager,
)
from benchmark_models.inference_tools.tf_inference_manager import TFInferenceManager
from benchmark_models.injector.faultlist_loader import load_fault_list
from benchmark_models.utils import (
    SUPPORTED_MODELS,
    SUPPORTED_DATASETS,
    get_loader,
    load_network,
)
import torch
import torch.nn as nn
import argparse
from contextlib import contextmanager
from tabulate import tabulate
from tqdm import tqdm
import numpy as np
import csv
import os
from datetime import datetime

from utils import float32_to_int, int_to_float32

DEFAULT_REPORT_FOLDER = "reports"


@contextmanager
def weight_bit_flip_applied(
    pt_model: nn.Module,
    layer: nn.Module,
    weight_coord: Tuple[int],
    bit: int,
):
    """
    Applies a bit flip to the weight when inside the context. The modifications are made in place.
    When exiting the context the weight is restored to the original version (even after an exception).

    Usage Example
    ----
    ```
    model = ... # Load PyTorch model
    layer = 'conv1'
    weight_coord = (1,2,0,1)
    bit = 27


    with weight_bit_flip_applied(model, layer, weight_coord, bit):
        # Execute inside here inference with fault active
        ...
    # Here outside of the context, the fault is not active and the models works as before the context
    ```

    Args
    ---
    * ``pt_model: nn.Module``
        The PyTorch model to inject
    * ``layer : nn.Module``
        The Module object  that contains the target layer. It must have a .weight attribute.
    * ``weight_coord: Tuple[int]``
        The coordinate of the weight that needs to bit flipped. The weight_coord must be a valid index in ``layer.weight``
    * ``bit: int``
        The bit position to flip.

    Returns
    ---
    ``model``. (Not used since the modifications are made in place to the model)
    """
    with torch.no_grad():
        weights = layer.weight
        selected_weight = float(weights[tuple(weight_coord)])
        selected_weight_int = float32_to_int(selected_weight)
        selected_weight_int ^= 1 << bit
        new_weight = int_to_float32(selected_weight_int)
        weights[tuple(weight_coord)] = new_weight
        try:
            yield pt_model
        finally:
            weights[tuple(weight_coord)] = selected_weight


def main(args):
    _, loader = get_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        permute_tf=False,
        dataset_path="../datasets",
    )

    device = torch.device(args.device)

    network = load_network(
        args.network_name,
        device,
        args.dataset,
    )
    network.eval()
    network.to(device)

    target_layers_list, injections = load_fault_list(
        args.fault_list, convert_faults_pt_to_tf=False
    )

    layer_shapes = [layer.weight.shape for layer in target_layers_list]
    target_layers_list = zip(target_layers_list, layer_shapes)

    top_1_accuracy = TopKAccuracy(k=1)
    top_5_accuracy = TopKAccuracy(k=5)

    inf_manager = PTInferenceManager(network, args.network_name, device, loader)
    inf_manager.run_clean()
    top_1_gold, top_1_acc = inf_manager.evaluate_metric(
        top_1_accuracy, use_faulty_outputs=False
    )
    top_5_gold, top_5_acc = inf_manager.evaluate_metric(
        top_5_accuracy, use_faulty_outputs=False
    )
    print(f"Accuracy TOP 1: {top_1_gold} {top_1_acc * 100}%")
    print(f"Accuracy TOP 5: {top_5_gold} {top_5_acc * 100}%")
    golden_labels = np.argmax(inf_manager.clean_output_scores, axis=-1)

    csv_headers = [
        "inj_id",
        "layer",
        "weigths",
        "bit_pos",
        "n_injections",
        "top_1_correct",
        "top_5_correct",
        "top_1_robust",
        "top_5_robust",
        "masked",
        "non_critical",
        "critical",
    ]

    gold_row = [
        "GOLDEN",
        None,
        None,
        None,
        inf_manager.clean_inference_counts,
        top_1_gold,
        top_5_gold,
        None,
        None,
        None,
        None,
    ]

    if args.output_path is None:
        report_folder = os.path.join(
            DEFAULT_REPORT_FOLDER, args.dataset, args.network_name
        )
        os.makedirs(report_folder, exist_ok=True)
        report_file_path = os.path.join(
            report_folder,
            f"{args.dataset}_{args.network_name}_{datetime.now().strftime('%y%m%d_%H%M')}.csv",
        )
    else:
        report_file_path = args.output_path

    if args.save_scores:
        report_folder_base = os.path.join(
            os.path.dirname(report_file_path), datetime.now().strftime("%y%m%d_%H%M")
        )
        os.makedirs(report_folder_base, exist_ok=True)
        np.save(
            os.path.join(report_folder_base, "clean.npy"),
            np.array(inf_manager.clean_output_scores),
        )

    write_header = not os.path.exists(report_file_path)
    with open(report_file_path, "a") as f:
        report_writer = csv.writer(f)
        if write_header:
            report_writer.writerow(csv_headers)
            report_writer.writerow(gold_row)

        for inj_num, injection in enumerate(
            tqdm(injections[args.resume_from :], leave=False)
        ):
            inj_id, target_layer_name, weight_coords, bit_pos = injection
            target_layer = network.get_submodule(target_layer_name)
            with weight_bit_flip_applied(network, target_layer, weight_coords, bit_pos):
                inf_manager.run_faulty(network)
                inf_count = inf_manager.faulty_inference_counts

                top_1_count, _ = inf_manager.evaluate_metric(
                    top_1_accuracy, use_faulty_outputs=True
                )
                top_5_count, _ = inf_manager.evaluate_metric(
                    top_5_accuracy, use_faulty_outputs=True
                )

                clean_out_scores = np.array(inf_manager.clean_output_scores)
                faulty_out_scores = np.array(inf_manager.faulty_output_scores)

                if args.save_scores:
                    np.save(
                        os.path.join(report_folder_base, f"inj_{inj_id}.npy"),
                        faulty_out_scores,
                    )

                masked_count = (clean_out_scores == faulty_out_scores).all(axis=1).sum()

                top_1_robust_count, top_1_robustness = top_1_accuracy(
                    inf_count, golden_labels, faulty_out_scores
                )
                top_5_robust_count, top_5_robustness = top_5_accuracy(
                    inf_count, golden_labels, faulty_out_scores
                )

                non_critical_count = top_1_robust_count - masked_count

                critical_count = inf_count - top_1_robust_count

                report_data = [
                    inj_id,
                    target_layer_name,
                    weight_coords,
                    bit_pos,
                    inf_count,
                    top_1_count,
                    top_5_count,
                    top_1_robust_count,
                    top_5_robust_count,
                    masked_count,
                    non_critical_count,
                    critical_count,
                ]
                report_writer.writerow(report_data)
                if inj_num % 10 == 0:
                    f.flush()
                inf_manager.reset_faulty_run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform fault injections to weigths of a Tensorflow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset to use",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=512, help="Test set batch size"
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
        "--fault-list",
        "-f",
        type=str,
        required=True,
        help="Path to Fault list",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        required=False,
        help="Path to the generated output",
    )
    parser.add_argument(
        "--resume-from",
        "-r",
        type=int,
        required=False,
        default=0,
        help="Resume experiment from a certain injection id",
    )
    parser.add_argument(
        "--save-scores",
        "-s",
        action="store_true",
        help="Save Injection Data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device used for inference execution",
    )

    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    main(parse_args())
    # load_fault_list('test.csv')
