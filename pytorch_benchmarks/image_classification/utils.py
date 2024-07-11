import os
import argparse

import torch

from models.CIFAR10 import inception_cifar10
from models.CIFAR10 import mobilenetv2_cifar10
from models.CIFAR10 import googlenet_cifar10
from models.CIFAR10 import mobilenetv2_cifar10
from models.CIFAR10 import vgg_cifar10
from models.CIFAR10 import resnet_cifar10
from models.CIFAR10 import densenet_cifar10

from models.CIFAR100 import resnet_cifar100
from models.CIFAR100 import densenet_cifar100
from models.CIFAR100 import googlenet_cifar100

from models.GTSRB import vgg_GTSRB
from models.GTSRB import resnet_GTSRB
from models.GTSRB import densenet_GTSRB

# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader

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


import os

from tqdm import tqdm

from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, GTSRB
from torchvision.transforms.v2 import (
    ToTensor,
    Resize,
    Compose,
    ColorJitter,
    RandomRotation,
    AugMix,
    GaussianBlur,
    RandomEqualize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

from abc import ABC, abstractmethod


class UnknownNetworkException(Exception):
    pass


SUPPORTED_DATASETS = ["CIFAR10", "CIFAR100", "GTSRB"]

SUPPORTED_MODELS = [
    "ResNet18",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "DenseNet121",
    "DenseNet161",
    "GoogLeNet",
    "MobileNetV2",
    "InceptionV3",
    "Vgg11_bn",
    "Vgg13_bn",
    "Vgg16_bn",
    
]


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
        "--forbid-cuda",
        action="store_true",
        help="Completely disable the usage of CUDA. This command overrides any other gpu options.",
    )
    parser.add_argument(
        "--use-cuda", action="store_true", help="Use the gpu if available."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Test set batch size"
    )
    parser.add_argument(
        "--network-name",
        "-n",
        type=str,
        required=True,
        help="Target network",
        choices=SUPPORTED_MODELS,
    )
    parsed_args = parser.parse_args()

    return parsed_args


class Identity:
    """Rotate by one of the given angles."""

    def __call__(self, x):
        return x



def load_CIFAR10_datasets(
    train_batch_size=32,
    train_split=0.8,
    test_batch_size=1,
    test_image_per_class=None,
    dataset_path="datasets",
):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # Crop the image to 32x32
            transforms.RandomHorizontalFlip(),  # Data Augmentation
            transforms.ToTensor(),  # Transform from image to pytorch tensor
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ) # Normalize the data (stability for training)
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.CenterCrop(32),  # Crop the image to 32x32
            transforms.ToTensor(),  # Transform from image to pytorch tensor
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )  # Normalize the data (stability for training)
        ]
    )

    train_dataset = CIFAR10(
        dataset_path, train=True, transform=transform_train, download=True
    )
    test_dataset = CIFAR10(
        dataset_path, train=False, transform=transform_test, download=True
    )

    # If only a number of images is required per class, modify the test set
    if test_image_per_class is not None:
        image_tensors = list()
        label_tensors = list()
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                image_tensors.append(test_image[0])
                label_tensors.append(test_image[1])
                image_class_counter[test_image[1]] += 1
        test_dataset = TensorDataset(
            torch.stack(image_tensors), torch.tensor(label_tensors)
        )

    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        lengths=[train_split_length, val_split_length],
        generator=torch.Generator().manual_seed(1234),
    )
    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(
        dataset=train_subset, batch_size=train_batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_subset, batch_size=train_batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=False
    )

    print("CIFAR10 Dataset loaded")

    return train_loader, val_loader, test_loader


def load_CIFAR100_datasets(
    train_batch_size=32,
    train_split=0.8,
    test_batch_size=1,
    test_image_per_class=None,
    dataset_path="datasets",
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            )
        ]
    )

    train_dataset = CIFAR100(
        dataset_path, train=True, transform=transform, download=True
    )
    test_dataset = CIFAR100(
        dataset_path, train=False, transform=transform, download=True
    )

    train_split = 0.8
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        lengths=[train_split_length, val_split_length],
        generator=torch.Generator(),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_subset, batch_size=train_batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_subset, batch_size=train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=False
    )

    print("CIFAR100 Dataset loaded")
    return train_loader, val_loader, test_loader


def load_GTSRB_datasets(
    train_batch_size=32,
    train_split=0.8,
    test_batch_size=1,
    test_image_per_class=None,
    dataset_path="datasets",
):
    train_transforms = Compose(
        [
            ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
            RandomEqualize(0.4),
            AugMix(),
            RandomHorizontalFlip(0.3),
            RandomVerticalFlip(0.3),
            GaussianBlur((3, 3)),
            RandomRotation(30),
            Resize([50, 50]),
            ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
        ]
    )

    test_transforms = Compose(
        [
            Resize([50, 50]),
            ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
        ]
    )

    train_dataset = GTSRB(
        root=dataset_path, split="train", download=True, transform=train_transforms
    )
    test_dataset = GTSRB(
        root=dataset_path, split="test", download=True, transform=test_transforms
    )

    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * 0.8)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        lengths=[train_split_length, val_split_length],
        generator=torch.Generator().manual_seed(1234),
    )
    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(
        dataset=train_subset, batch_size=train_batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_subset, batch_size=train_batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=False
    )

    print("GTSRB Dataset loaded")

    return train_loader, val_loader, test_loader


def load_from_dict(network, device, path, function=None):
    if ".th" in path:
        state_dict = torch.load(path, map_location=device)["state_dict"]
        print("Loaded from .th file")
    else:
        state_dict = torch.load(path, map_location=device)
        print("state_dict loaded")

    if function is None:
        clean_state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }
    else:
        clean_state_dict = {
            key.replace("module.", ""): (
                function(value) if not (("bn" in key) and ("weight" in key)) else value
            )
            for key, value in state_dict.items()
        }

    network.load_state_dict(clean_state_dict, strict=False)
    print("state_dict loaded into network")
    

def get_loader(
    dataset_name: str,
    batch_size: int,
    image_per_class: int = None,
    network: torch.nn.Module = None,
    dataset_path="datasets",
) -> DataLoader:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param network_name: The name of the network
    :param batch_size: The batch size
    :param image_per_class: How many images to load for each class
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :return: The DataLoader
    """
    if dataset_name == "CIFAR10":
        print("Loading CIFAR10 dataset")
        train_loader, _, loader = load_CIFAR10_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class,
        )


    elif dataset_name == "CIFAR100":
        print("Loading CIFAR100 dataset")
        train_loader, _, loader = load_CIFAR100_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class,
        )

    elif dataset_name == "GTSRB":
        print("Loading GTSRB dataset")
        train_loader, _, loader = load_GTSRB_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class,
        )

    else:
        raise UnknownNetworkException(f"ERROR: unknown dataset: {dataset_name}")

    print(f"Batch size:\t\t{batch_size} \nNumber of batches:\t{len(loader)}")

    return train_loader, loader


def load_network(
    network_name: str, device: torch.device, dataset_name: str
) -> torch.nn.Module:
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :param device: the device where to load the network
    :return: The loaded network
    """
    network_path = f'pretrained_weights/{dataset_name}/{network_name}_{dataset_name}'
    if dataset_name == "CIFAR10":
        if "ResNet" in network_name:
            if network_name == "ResNet20":
                network_function = resnet_cifar10.resnet20()
            elif network_name == "ResNet32":
                network_function = resnet_cifar10.resnet32()
            elif network_name == "ResNet44":
                network_function = resnet_cifar10.resnet44()
            elif network_name == "ResNet56":
                network_function = resnet_cifar10.resnet56()
            elif network_name == "ResNet110":
                network_function = resnet_cifar10.resnet110()
            elif network_name == "ResNet1202":
                network_function = resnet_cifar10.resnet1202()
            else:
                raise UnknownNetworkException(
                    f"ERROR: unknown version of ResNet: {network_name}"
                )

            network = network_function
            network_path += ".th"
            # Load the weights
            load_from_dict(network=network, device=device, path=network_path)


        elif "DenseNet" in network_name:
            if network_name == "DenseNet121":
                network = densenet_cifar10.densenet121()
            elif network_name == "DenseNet161":
                network = densenet_cifar10.densenet161()
            elif network_name == "DenseNet169":
                network = densenet_cifar10.densenet169()
            else:
                raise UnknownNetworkException(
                    f"ERROR: unknown version of ResNet: {network_name}"
                )
            network_path += ".pt"
            load_from_dict(network=network, device=device, path=network_path)

        elif "Vgg" in network_name:
            if network_name == "Vgg11_bn":
                network = vgg_cifar10.vgg11_bn()
            elif network_name == "Vgg13_bn":
                network = vgg_cifar10.vgg13_bn()
            elif network_name == "Vgg16_bn":
                network = vgg_cifar10.vgg16_bn()
            elif network_name == "Vgg19_bn":
                network = vgg_cifar10.vgg19_bn()
            else:
                raise UnknownNetworkException(
                    f"ERROR: unknown version of ResNet: {network_name}"
                )
            network_path += ".pt"
            load_from_dict(network=network, device=device, path=network_path)

        elif "GoogLeNet" in network_name:
            network = googlenet_cifar10.GoogLeNet()
            network_path += ".pt"
            load_from_dict(network=network, device=device, path=network_path)

        elif "MobileNetV2" in network_name:
            network = mobilenetv2_cifar10.MobileNetV2()

            state_dict = torch.load(network_path, map_location=device)["net"]
            function = None
            if function is None:
                clean_state_dict = {
                    key.replace("module.", ""): value
                    for key, value in state_dict.items()
                }
            else:
                clean_state_dict = {
                    key.replace("module.", ""): (
                        function(value)
                        if not (("bn" in key) and ("weight" in key))
                        else value
                    )
                    for key, value in state_dict.items()
                }
            network_path += ".pt"
            network.load_state_dict(clean_state_dict, strict=False)

        elif "InceptionV3" in network_name:
            network = inception_cifar10.Inception3()
            network_path += ".pt"
            load_from_dict(network=network, device=device, path=network_path)
        else:
            raise UnknownNetworkException(f"ERROR: unknown network: {network_name}")

    elif dataset_name == "CIFAR100":
        print(f"Loading network {network_name}")
        if "ResNet18" in network_name:
            network = resnet_cifar100.resnet18()
            print("resnet18 loaded")
        elif "DenseNet121" in network_name:
            network = densenet_cifar100.densenet121()
            print("densenet121 loaded")
        elif "GoogLeNet" in network_name:
            network = googlenet_cifar100.googlenet()
            print("googlenet loaded")
        else:
            raise UnknownNetworkException(
                f"ERROR: unknown version of the model: {network_name}"
            )
        network_path += ".pth"
        load_from_dict(network=network, device=device, path=network_path)

    elif dataset_name == "GTSRB":
        print(f"Loading network {network_name}")
        if "ResNet20" in network_name:
            network = resnet_GTSRB.resnet20()
            print("resnet20 loaded")
        elif "DenseNet121" in network_name:
            network = densenet_GTSRB.densenet121()
            print("densenet121 loaded")
        elif "Vgg11_bn" in network_name:
            network = vgg_GTSRB.vgg11_bn()
            print("vgg11_bn loaded")
        else:
            raise UnknownNetworkException(
                f"ERROR: unknown version of the model: {network_name}"
            )
        network_path += ".pt"

        load_from_dict(network=network, device=device, path=network_path)

    else:
        raise UnknownNetworkException(f"ERROR: unknown dataset: {dataset_name}")

    network.to(device)
    network.eval()

    # Send network to device and set for inference

    return network


def get_device(forbid_cuda: bool, use_cuda: bool) -> torch.device:
    """
    Get the device where to perform the fault injection
    :param forbid_cuda: Forbids the usage of cuda. Overrides use_cuda
    :param use_cuda: Whether to use the cuda device or the cpu
    :return: The device where to perform the fault injection
    """

    # Disable gpu if set
    if forbid_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
        if use_cuda:
            print("WARNING: cuda forcibly disabled even if set_cuda is set")
    # Otherwise, use the appropriate device
    else:
        if use_cuda:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = ""
                print("ERROR: cuda not available even if use-cuda is set")
                exit(-1)
        else:
            device = "cpu"

    return torch.device(device)




# TODO Implement metrics


class MetricEvaluator(ABC):
    @abstractmethod
    def __call__(self, inferences_count: int, labels, outputs):
        pass


class TopKAccuracy(MetricEvaluator):
    def __init__(self, k):
        if k < 1:
            raise ValueError("k must be greather than 0")
        self.k = k

    def __call__(self, inferences_count, labels, outputs):
        flat_labels = np.expand_dims(
            np.array(labels), axis=1
        )  # SHAPE (inferences_count) CONTENT (correct_label)
        flat_outputs = np.array(
            outputs
        )  # SHAPE (inferences_count, n_classes) CONTENT (image, class_score)
        top_k_indexes = flat_outputs.argpartition(-self.k, axis=-1)[
            :, -self.k :
        ]  # SHAPE (inferences_count, self.k) CONTENT (image, top_k_idxs)
        correct_elements = flat_labels == top_k_indexes
        correct = correct_elements.any(axis=1).sum()
        return correct, correct / inferences_count

class InferenceManager(ABC):
    def __init__(
        self,
        network,
        network_name: str,
        loader: DataLoader,
        dataset_name: str
    ):
        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.dataset_name = dataset_name
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
        self.label_output_dir = f'output/{self.dataset_name}/{self.network_name}/batch_size_{self.loader.batch_size}/labels'

        self.clean_output_dir = f'output/{self.dataset_name}/{self.network_name}/batch_size_{self.loader.batch_size}/clean'

        self.clean_faulty_dir = f'output/{self.dataset_name}/{self.network_name}/batch_size_{self.loader.batch_size}/faulty'
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


class PTInferenceManager(InferenceManager):
    def __init__(
        self,
        network: Module,
        network_name: str,
        device: torch.device,
        loader: DataLoader,
        dataset_name: str
    ):
        super(PTInferenceManager, self).__init__(network, network_name, loader, dataset_name)
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