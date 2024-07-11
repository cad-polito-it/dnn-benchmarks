import os

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageNet, CIFAR100, GTSRB
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


class PermuteToTensorFlow:
    """Rotate by one of the given angles."""

    def __call__(self, x):
        return x.permute(1, 2, 0).contiguous()


class Identity:
    """Rotate by one of the given angles."""

    def __call__(self, x):
        return x


def load_ImageNet_validation_set(
    batch_size,
    image_per_class=None,
    imagenet_folder="~/Datasets/ImageNet",
    permute_tf=False,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_validation = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_validation_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            PermuteToTensorFlow(),
        ]
    )

    validation_dataset_folder = "tmp"
    validation_dataset_path = (
        f"{validation_dataset_folder}/imagenet_{image_per_class}.pt"
    )
    validation_dataset_path_tf = (
        f"{validation_dataset_folder}/imagenet_{image_per_class}_tf.pt"
    )

    try:
        if image_per_class is None:
            raise FileNotFoundError
        if not permute_tf:
            validation_dataset = torch.load(validation_dataset_path)
        else:
            validation_dataset_tf = torch.load(validation_dataset_path_tf)
        print("Resized Imagenet loaded from disk")

    except FileNotFoundError:
        validation_dataset = ImageNet(
            root=imagenet_folder, split="val", transform=transform_validation
        )

        validation_dataset_tf = ImageNet(
            root=imagenet_folder, split="val", transform=transform_validation_tf
        )

        if image_per_class is not None:
            selected_validation_list = []
            image_class_counter = [0] * 1000
            for validation_image in tqdm(
                validation_dataset, desc="Resizing Imagenet Dataset", colour="Yellow"
            ):
                if image_class_counter[validation_image[1]] < image_per_class:
                    selected_validation_list.append(validation_image)
                    image_class_counter[validation_image[1]] += 1
            validation_dataset = selected_validation_list

            selected_validation_list = []
            image_class_counter = [0] * 1000
            for validation_image in tqdm(
                validation_dataset_tf,
                desc="Resizing Imagenet Dataset TF",
                colour="Yellow",
            ):
                if image_class_counter[validation_image[1]] < image_per_class:
                    selected_validation_list.append(validation_image)
                    image_class_counter[validation_image[1]] += 1
            validation_dataset_tf = selected_validation_list

        os.makedirs(validation_dataset_folder, exist_ok=True)
        torch.save(validation_dataset, validation_dataset_path)
        torch.save(validation_dataset_tf, validation_dataset_path_tf)
    # DataLoader is used to load the dataset
    # for training
    if permute_tf:
        val_loader = torch.utils.data.DataLoader(
            dataset=validation_dataset_tf,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        val_loader = torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    print("Dataset loaded")

    return val_loader


def load_CIFAR10_datasets(
    train_batch_size=32,
    train_split=0.8,
    test_batch_size=1,
    test_image_per_class=None,
    permute_tf=False,
    dataset_path="datasets",
):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # Crop the image to 32x32
            transforms.RandomHorizontalFlip(),  # Data Augmentation
            transforms.ToTensor(),  # Transform from image to pytorch tensor
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),  # Normalize the data (stability for training)
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.CenterCrop(32),  # Crop the image to 32x32
            transforms.ToTensor(),  # Transform from image to pytorch tensor
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),  # Normalize the data (stability for training)
            PermuteToTensorFlow() if permute_tf else Identity(),
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
    permute_tf=False,
    dataset_path="datasets",
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            ),
            PermuteToTensorFlow() if permute_tf else Identity(),
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
    permute_tf=False,
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
            PermuteToTensorFlow() if permute_tf else Identity(),
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
