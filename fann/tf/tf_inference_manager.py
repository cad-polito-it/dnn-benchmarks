import os
import shutil
import time
import math
from datetime import timedelta

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow import keras

import numpy as np
from tqdm import tqdm

from fann.inference_tools.inference_manager import InferenceManager


class TFInferenceManager(InferenceManager):
    def __init__(self, network: keras.Model, network_name: str, loader: DataLoader):
        super(TFInferenceManager, self).__init__(network, network_name, loader)

    def run_inference(self, faulty=False, verbose=False, save_outputs=True):
        """
        Run a clean inference of the network
        :return: A string containing the formatted time elapsed from the beginning to the end of the fault injection
        campaign
        """
        # Start measuring the time elapsed
        start_time = time.time()

        # Cycle all the batches in the data loader
        pbar = tqdm(
            self.loader,
            colour="red" if faulty else "green",
            desc="Faulty Run" if faulty else "Clean Run",
            ncols=shutil.get_terminal_size().columns,
        )

        if not faulty:
            self.clean_inference_counts = 0
            self.clean_labels = []
            self.clean_output_indices = []
            self.clean_output_scores = []
        else:
            self.faulty_inference_counts = 0
            self.faulty_output_scores = []

        dataset_size = len(self.loader.dataset)
        inferences_count = 0

        for batch_id, batch in enumerate(pbar):
            # print(batch_id)
            data, labels = batch
            # print(len(label)) #total of 10000 images
            # print(label)
            inferences_count += len(labels)
            data = tf.convert_to_tensor(data.numpy())

            # Run inference on the current batch
            scores = self.__run_inference_on_batch(data=data)
            if not faulty:
                # Append the results to a list
                self.clean_inference_counts = inferences_count
                self.clean_output_scores += scores.tolist()
                self.clean_labels += labels
            else:
                # Append the results to a list
                self.faulty_inference_counts = inferences_count
                self.faulty_output_scores += scores.tolist()

            if save_outputs:
                # Save the output
                torch.save(scores, f"{self.clean_output_dir}/batch_{batch_id}.pt")
                torch.save(labels, f"{self.label_output_dir}/batch_{batch_id}.pt")

        # Stop measuring the time
        elapsed = math.ceil(time.time() - start_time)

        return str(timedelta(seconds=elapsed))

    def __run_inference_on_batch(self, data):
        """
        Rim a fault injection on a single batch
        :param data: The input data from the batch
        :return: a tuple (scores, indices) where the scores are the vector score of each element in the batch and the
        indices are the argmax of the vector score
        """

        # Execute the network on the batch
        network_output = self.network(
            data
        ).numpy()  # it is a vector of output elements (one vector for each image). The size is num_batches * num_outputs

        return network_output