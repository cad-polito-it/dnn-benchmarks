from dataclasses import dataclass
import csv
import torch
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from benchmark_models.inference_tools.metric_evaluators import TopKAccuracy
from benchmark_models.inference_tools.tf_inference_manager import TFInferenceManager
from benchmark_models.injector.faultlist_loader import convert_weights_coords_from_pt_to_tf, load_fault_list
from benchmark_models.injector.pt_module_profiler import profile_weight_shape
from benchmark_models.injector.utils import float32_to_int, int_to_float32
from benchmark_models.utils import SUPPORTED_MODELS, SUPPORTED_DATASETS, get_loader, load_network
from benchmark_models.tf_utils import load_converted_tf_network
import argparse
from contextlib import contextmanager
from operator import attrgetter
from tabulate import tabulate
import struct
from tqdm import tqdm
import numpy as np
import csv
import os
from datetime import datetime
from natsort import natsorted

DEFAULT_REPORT_FOLDER = "reports"
INJECTED_LAYERS_TYPES_KERAS = (keras.layers.Conv2D, keras.layers.Dense)


@contextmanager
def weight_bit_flip_applied(
    keras_model: keras.Model,
    layer: keras.layers.Layer,
    weight_coord: Tuple[int],
    bit: int,
):
    """
    Applies a bit flip to the weight when inside the context. The modifications are made in place.
    When exiting the context the weight is restored to the original version (even after an exception).

    Usage Example
    ----
    ```
    model = ... # Load Keras model
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
    * ``pt_model: keras.Model``
        The Keras model to inject
    * ``layer : keras.layers.Layer``
        The Keras layer object  that contains the target layer. It must have a .weight attribute.
    * ``weight_coord: Tuple[int]``
        The coordinate of the weight that needs to bit flipped. The weight_coord must be a valid index in ``layer.weight``
    * ``bit: int``
        The bit position to flip.

    Returns
    ---
    ``model``. (Not needed since the modifications are made in place to the model)
    """
    weights = layer.get_weights()
    selected_weight = weights[0][weight_coord]
    selected_weight_int = float32_to_int(selected_weight)
    selected_weight_int ^= 1 << bit
    new_weight = int_to_float32(selected_weight_int)
    weights[0][weight_coord] = new_weight
    layer.set_weights(weights)
    try:
        yield keras_model
    finally:
        weights[0][weight_coord] = selected_weight
        layer.set_weights(weights)


def main(args):
    _, loader = get_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        permute_tf=True,
        dataset_path="../datasets",
    )
    pt_network = load_network(
        args.network_name,
        torch.device("cpu"),
        args.dataset,
    )
    tf_network = load_converted_tf_network(
        args.network_name,
        args.dataset,
        models_path="models/converted-tf",
    )
    faulty_network = keras.models.clone_model(tf_network)
    faulty_network.set_weights(tf_network.get_weights())

    tf_network.summary(expand_nested=True)

    target_layers_names, injections = load_fault_list(
        args.fault_list, convert_faults_pt_to_tf=True
    )

    # Filter layers
    keras_conv_layers = [
        layer
        for layer in faulty_network._flatten_layers(include_self=False, recursive=True)
        if isinstance(layer, INJECTED_LAYERS_TYPES_KERAS)
    ]

    keras_conv_layers_names = list(map(attrgetter("name"), keras_conv_layers))
    weights_shapes = [[w.shape for w in l.get_weights()] for l in keras_conv_layers]

    print(
        tabulate(
            zip(target_layers_names, keras_conv_layers_names, weights_shapes),
            headers=["PyTorch", "Keras", "Keras Weight Shape"],
        )
    )


    print(f"Number of target layers: {len(target_layers_names)}")
    print(f"Number of conv layers in model: {len(keras_conv_layers)}")

    if len(target_layers_names) != len(keras_conv_layers):
        raise ValueError("Mismatching layer mapping")

    top_1_accuracy = TopKAccuracy(k=1)
    top_5_accuracy = TopKAccuracy(k=5)
    if args.sort_tf_layers:
        print("Reordering layers")
        keras_conv_layers = natsorted(keras_conv_layers, key=lambda l: l.name)

    
    data, label = next(iter(loader))
    target_layers_shapes = profile_weight_shape(pt_network)
    target_layer_mapping = dict(zip(target_layers_names, keras_conv_layers))

    for pt_layer_name, tf_layer in target_layer_mapping.items():
        pt_shape = target_layers_shapes[pt_layer_name]
        tf_shape = tf_layer.get_weights()[0].shape
        pt_shape_converted = convert_weights_coords_from_pt_to_tf(pt_shape)
        if not all(a == b for a,b in  zip(pt_shape_converted, tf_shape)):
            print(f'TF weight shape {pt_shape_converted} at layer {pt_layer_name} and PT weight {tf_shape} at layer {tf_layer.name} do not match')

    inf_manager = TFInferenceManager(tf_network, "ResNet20", loader)
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
            target_layer = target_layer_mapping[target_layer_name]
            with weight_bit_flip_applied(
                faulty_network, target_layer, weight_coords, bit_pos
            ):
                inf_manager.run_faulty(faulty_network)
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
        required=False,
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
        "--sort-tf-layers",
        action="store_true",
        help="(Nat)Sort TF layers by name to make them match with their layer",
    )

    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    main(parse_args())
    # load_fault_list('test.csv')
