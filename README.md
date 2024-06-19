# Deep Neural Network Models for Reliability Studies
Welcome to the repository containing state-of-the-art Deep Neural Network (DNN) models implemented in both PyTorch and TensorFlow for conducting reliability studies. 

## Project Collaboration

This project is a collaboration between the following institutions:

- [Politecnico di Torino](https://www.polito.it/)
- [Politecnico di Milano](https://www.polimi.it/)
- [Ecole Centrale de Lyon](https://www.ec-lyon.fr/en)

## Installation

1. Create a virtual environment

```
python -m venv .venv
```

2. Activate the environment

```
source .venv/bin/activate
```

3. Install the dependencies from the requirements
You can find a requirements.txt from which you can install all dependencies using

```
pip install -r requirements.txt
```

4. Download the pretrained networks by running ?
```
./download_models.sh
```

## Getting Started

### Pytorch

A clean Pytorch inference can be executed with the following programm:
```
python main.py -n network-name -b batch-size -d dataset-name 
```

It is possible to execute inferences with available GPUs sepcifing the argument ```--use-cuda```.

By default, results are saved in ```.pt``` files in the ```output/dataset-name/network-name/batch-size/``` folder. 

In addition, the accuracy of the models will be displayed on the terminal after the inferences of the selected test dataset are completed.


### Tensorflow

DARIO

## Available Models (so far)

The Keras versions of the models, when available, are obtained using the [nobuco](https://github.com/AlexanderLutsenko/nobuco) PyTorch to Keras converter.
The Keras versions of all models share the same structure and weigths, and have similar accuracies to their PyTorch counterpart.

### CIFAR-10 Models
Here is a list of models trained for CIFAR10 dataset, that has images belonging to 10 classes.
All the models are validated using the CIFAR10 validation set, that cointains 10000 images.

|    Model    |    PyTorch TOP-1 Accuracy     |     Keras TOP-1 Accuracy     |
| :---------: | :---------------------------: | :--------------------------: |
|  ResNet20   | <div align="right">91.5 %</div> | <div align="right">91.5 %</div> |
|  ResNet32   | <div align="right">92.3 %</div> | <div align="right">92.3 %</div> |
|  ResNet44   | <div align="right">92.8 %</div> | <div align="right">92.8 %</div> |
|  ResNet56   | <div align="right">93.3 %</div> | <div align="right">93.3 %</div> |
|  ResNet110  | <div align="right">93.5 %</div> | <div align="right">93.5 %</div> |
| MobileNetV2 | <div align="right">91.7 %</div> | <div align="right">91.7 %</div> |
|  Vgg19_bn   | <div align="right">93.2 %</div> | <div align="right">93.2 %</div> |
|  Vgg16_bn   | <div align="right">93.5 %</div> | <div align="right">93.5 %</div> |
|  Vgg13_bn   | <div align="right">93.8 %</div> | <div align="right">93.8 %</div> |
|  Vgg11_bn   | <div align="right">91.3 %</div> | <div align="right">91.3 %</div> |
| DenseNet121 | <div align="right">93.2 %</div> | <div align="right">93.1 %</div> |
| DenseNet161 | <div align="right">93.1 %</div> | <div align="right">93.1 %</div> |
|  GoogLeNet  | <div align="right">92.2 %</div> | <div align="right">92.2 %</div> |

### CIFAR-100 Models
Here is a list of models trained for CIFAR100 dataset, that has images belonging to 100 classes.
All the models are validated using the CIFAR100 validation set, that cointains 10000 images.

|    Model    | PyTorch TOP-1 Accuracy | Keras TOP-1 Accuracy |
| :---------: | :--------------------: | :------------------: |
|  ResNet18   | <div align="right">76.2 %</div> | <div align="right">76.2 %</div> |
| DenseNet121 | <div align="right">78.7 %</div> | <div align="right">78.7 %</div> |
|  GoogLeNet  | <div align="right">76.3 %</div> | <div align="right">76.3 %</div> |


### GTSRB Models
Here is a list of models trained for GTSRB dataset, containing 43 classes of German Traffic signals.
All the models are validated using the GTSRB validation set, that cointains 12640 images.

|    Model    | PyTorch TOP-1 Accuracy | Keras TOP-1 Accuracy |
| :---------: | :--------------------: | :------------------: |
|  ResNet20   | <div align="right">94.3%</div> | <div align="right">94.3%</div> |
| DenseNet121 | <div align="right">96.5%</div> | <div align="right">96.5%</div> |
|  Vgg11_bn   | <div align="right">95.5%</div> | <div align="right">95.5%</div> |


