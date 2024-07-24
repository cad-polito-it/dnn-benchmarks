import torch
import tensorflow
import tqdm
from tensorflow import keras
from torchvision import transforms
from torchvision.datasets import CIFAR10

class PermuteToTensorFlow:
    """Rotate by one of the given angles."""

    def __call__(self, x):
        return x.permute(1, 2, 0).contiguous()

# Load the dataset
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
        PermuteToTensorFlow(), # Tensorflow specific
    ]
)

    # Download the dataset
train_dataset = CIFAR10(
    ".", train=True, transform=transform_train, download=True
)
test_dataset = CIFAR10(
    ".", train=False, transform=transform_test, download=True
)
    # Prepare the loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=32, shuffle=True
)
test_loader = test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=32, shuffle=False
)

# Load the model
model = keras.models.load_model("./Vgg19_bn.keras")

# Prepare accuracy
total = 0
correct = 0

# Perform the inference
bar = tqdm.tqdm(test_loader)
for batch in bar:
    data, labels = batch

    total += len(labels)
    data = tensorflow.convert_to_tensor(data.numpy())

    scores = model(data).numpy()
    predicted = scores.argmax(axis=1)
    correct += (predicted == labels.numpy()).sum()    
    bar.set_description_str(f"Cumulative accuracy: {correct/total*100:.2f}")
print(f"Model accuracy is {correct/total*100:.2f}%")
