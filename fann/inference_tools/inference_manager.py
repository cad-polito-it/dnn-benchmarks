import os
from typing import List
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
from torch.utils.data import DataLoader


from fann.inference_tools.metric_evaluators import MetricEvaluator


class InferenceManager(ABC):
    def __init__(
        self,
        network,
        network_name: str,
        loader: DataLoader,
    ):
        self.network = network
        self.network_name = network_name
        self.loader = loader

        # The clean output of the network after the first run
        self.clean_output_scores = list()
        self.clean_output_indices = list()
        self.faulty_output_scores = list()
        self.faulty_output_indices = list()
        self.clean_labels = list()
        self.clean_inference_counts = 0
        self.faulty_inference_counts = 0

        # TODO: Change format for saving data (if needed)
        # The output dir
        self.label_output_dir = os.path.join(
            "output",
            self.network_name,
            "pt",
            "label",
            f"batch_size_{self.loader.batch_size}",
        )

        self.clean_output_dir = os.path.join(
            "output",
            self.network_name,
            "pt",
            "clean",
            f"batch_size_{self.loader.batch_size}",
        )

        self.clean_faulty_dir = os.path.join(
            "output",
            self.network_name,
            "pt",
            "clean",
            f"batch_size_{self.loader.batch_size}",
        )

        # Create the output dir
        os.makedirs(self.label_output_dir, exist_ok=True)
        os.makedirs(self.clean_output_dir, exist_ok=True)

    def get_metrics_names(self) -> List[str]:
        return list(self.evaluators.keys())

    def evaluate_metric(self, metric: MetricEvaluator, use_faulty_outputs=False):
        """
        Run evaluation of a metric
        """
        if not use_faulty_outputs:
            output = metric(
                self.clean_inference_counts, self.clean_labels, self.clean_output_scores
            )
        else:
            output = metric(
                self.faulty_inference_counts,
                self.clean_labels,
                self.faulty_output_scores,
            )

        return output

    def reset(self):
        self.reset_clean_run()
        self.reset_faulty_run()

    def reset_clean_run(self):
        self.clean_output_scores = list()
        self.clean_output_indices = list()
        self.clean_labels = list()
        self.clean_inference_counts = 0

    def reset_faulty_run(self):
        self.faulty_output_scores = list()
        self.faulty_output_indices = list()
        self.faulty_inference_counts = 0

    @abstractmethod
    def run_inference(self, faulty=False, verbose=False, save_outputs=False):
        pass

    def run_faulty(self, faulty_network, save_outputs=False):
        gold_network = self.network
        self.network = faulty_network
        try:
            self.run_inference(save_outputs=save_outputs, faulty=True)
        finally:
            self.network = gold_network

    def run_clean(self, verbose=True, save_outputs=False):
        self.run_inference(faulty=False, verbose=verbose, save_outputs=save_outputs)
