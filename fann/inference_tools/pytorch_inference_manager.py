import os
import shutil
import time
import math
from datetime import timedelta
from typing import Callable, Dict, List, Union
from abc import ABC

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from benchmark_models.inference_tools.inference_manager import InferenceManager


class PTInferenceManager(InferenceManager):
    def __init__(
        self,
        network: Module,
        network_name: str,
        device: torch.device,
        loader: DataLoader,
    ):
        super(PTInferenceManager, self).__init__(network, network_name, loader)
        self.device = device

    def run_inference(self, faulty=False, verbose=False, save_outputs=True):
        """
        Run a clean inference of the network
        :return: A string containing the formatted time elapsed from the beginning to the end of the fault injection
        campaign
        """

        with torch.no_grad():
            # Start measuring the time elapsed
            start_time = time.time()

            # Cycle all the batches in the data loader
            pbar = tqdm(
                self.loader,
                colour="red" if faulty else "green",
                desc="Faulty Run" if faulty else "Clean Run",
                ncols=shutil.get_terminal_size().columns,
            )

            dataset_size = 0

            for batch_id, batch in enumerate(pbar):
                # print(batch_id)
                batch_data, batch_labels = batch
                # print(len(label)) #total of 10000 images
                # print(label)
                dataset_size = dataset_size + len(batch_labels)
                batch_data = batch_data.to(self.device)

                # Run inference on the current batch
                batch_scores, batch_indices = self.__run_inference_on_batch(
                    data=batch_data
                )
                if not faulty:
                    self.clean_inference_counts += len(batch_labels)
                else:
                    self.faulty_inference_counts += len(batch_labels)

                if save_outputs:
                    # Save the output
                    torch.save(
                        batch_scores, f"{self.clean_output_dir}/batch_{batch_id}.pt"
                    )
                    torch.save(
                        batch_labels, f"{self.label_output_dir}/batch_{batch_id}.pt"
                    )

                if not faulty:
                    # Append the results to a list
                    self.clean_output_scores += batch_scores
                    self.clean_output_indices += batch_indices
                    self.clean_labels += batch_labels
                else:
                    # Append the results to a list
                    self.faulty_output_scores += batch_scores
                    self.faulty_output_indices += batch_indices

        # COMPUTE THE ACCURACY OF THE NEURAL NETWORK
        # Element-wise comparison
        elementwise_comparison = [
            label != index
            for label, index in zip(self.clean_labels, self.clean_output_indices)
        ]

        # COMPUTE THE ACCURACY OF THE NEURAL NETWORK
        # Element-wise comparison
        if not faulty:
            elementwise_comparison = [
                label != index
                for label, index in zip(self.clean_labels, self.clean_output_indices)
            ]
        else:
            elementwise_comparison = [
                label != index
                for label, index in zip(self.clean_labels, self.faulty_output_indices)
            ]
        # Count the number of different elements
        if verbose:
            num_different_elements = elementwise_comparison.count(True)
            print(f"The DNN wrong predicions are: {num_different_elements}")
            accuracy = (1 - num_different_elements / dataset_size) * 100
            print(f"The final accuracy is: {accuracy}%")

        # Stop measuring the time
        elapsed = math.ceil(time.time() - start_time)

        return str(timedelta(seconds=elapsed))

    def __run_inference_on_batch(self, data: torch.Tensor):
        """
        Rim a fault injection on a single batch
        :param data: The input data from the batch
        :return: a tuple (scores, indices) where the scores are the vector score of each element in the batch and the
        indices are the argmax of the vector score
        """

        # Execute the network on the batch
        network_output = self.network(
            data
        )  # it is a vector of output elements (one vector for each image). The size is num_batches * num_outputs
        # print(network_output)
        prediction = torch.topk(
            network_output, k=1
        )  # it returns two lists : values with the top1 values and indices with the indices
        # print(prediction.indices)

        # Get the score and the indices of the predictions
        prediction_scores = network_output.cpu()

        prediction_indices = [int(fault) for fault in prediction.indices]
        return prediction_scores, prediction_indices
