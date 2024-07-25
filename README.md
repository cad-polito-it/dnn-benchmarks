# Deep Neural Network Models for Reliability Studies
Welcome to the repository containing state-of-the-art Deep Neural Network (DNN) models implemented in both PyTorch and TensorFlow for conducting reliability studies. 

## Project Collaboration

This project is a collaboration between the following institutions:

- [Politecnico di Torino](https://www.polito.it/)
- [Politecnico di Milano](https://www.polimi.it/)
- [Ecole Centrale de Lyon](https://www.ec-lyon.fr/en)


## Getting Started
The idea of this project is to share neural networks to conduct reliability studies. The goal is to make it easier and more accessible for different research groups to compare their results.

Within the repository, you will find the code and weights for some PyTorch models for image classification (so far), pre-trained on the CIFAR10, CIFAR100, and GTSRB datasets. Additionally, using [nobuco](https://github.com/AlexanderLutsenko/nobuco), we have converted the PyTorch models to their Keras counterparts, which share the same architecture, weights, and similar accuracy. A public backup fork of the converter is available also here [nobuco-fork](https://github.com/D4De/nobuco).

For each model, a fault list has been generated, and a fault injection campaign has been conducted to evaluate the reliability and comparability between the PyTorch and Keras versions. For further details, you can refer to the paper \cite{} submitted to (TCAD?). All fault lists are included in the repository, along with some of the results from the injection campaigns.

> In order to download the pretrained-weights, please execute the following command:
> `git submodule init`
> Nevertheless, this command will download all the available models with related trained weights.
> If you are interested in a specific subset, you need to initialize only the related modules. You find a comprehensive list of the avaiable models in the directory ./benchmark_models/models/ and you can download one of them with the following command:
> `git submodule init {model_name}` where you need to substitue `{model_name}` with the submodule name you find in the repo.

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


## Projects structure

The repository is organized into three main directories:
- ```pytorch_becnhark/```: contains folders for each type of task for each group of models, such as ```image_classification/```, each containing the code and weights of the PyTorch models
- ```tensorflow_benchmark/```: has the same structure as the previous directory but for the Keras models.
- ```fault_lists/```: contains the fault lists for each model and a portion of the results obtained from fault injection campaigns."

### Pytorch
Inside the ```pytorch_benchmarks/image_classification``` folder, you can run a test on all models.

A clean Pytorch inference can be executed with the following program:
```
python main.py -n network-name -b batch-size -d dataset-name 
```

It is possible to execute inferences with available GPUs specifying the argument ```--use-cuda```.

By default, results are saved in ```.pt``` files in the ```output/dataset-name/network-name/batch-size/``` folder. 

In addition, the accuracy of the models will be displayed on the terminal after the inferences of the selected test dataset are completed.


### Tensorflow
Inside the ```tensorflow_benchmarks/image_classification``` folder, you can a clean inference for each model, using the following command.

```
python main.py -n network-name -b batch-size -d dataset-name 
```

It is possible to execute inferences with available GPUs specifying the argument ```--use-cuda```.

The accuracy of the models will be displayed on the terminal after the inferences of the selected test dataset are completed.

## Fault list and FI

Inside the ```.fault_lists/```. directory, the fault lists have been generated for each model paired with a specific dataset to perform a statistical analysis of reliability. It is noted that the type of fault described is permanent and simulates a stuck-at fault in the memory where the model weights are stored. The files are in .csv format, and their structure is as follows:

PyTorch Fault List for a ResNet20 model trained on CIFAR10 example

| Injection | Layer |   TensorIndex  | Bit | n_injections | masked | non_critical | critical |
|:---------:|:-----:|:--------------:|:---:|:------------:|:------:|:------------:|:--------:|
|         0 | conv1 | "(3, 0, 2, 1)" |  15 |        10000 |     14 |         9985 |        1 |
|    ...    |  ...  |       ...      | ... |      ...     |   ...  |      ...     |    ...   |

- `Injection`: column indicating the injection number.
- `Layer`: the layer in which the fault is injected.        
- `TensorIndex`: coordinate of the weight where the fault is injected.
- `Bit`: corrupted bit that is flipped.
- `n_injections`: number of inferences made with the injected fault (matches the test set of the dataset).
- `masked`: inferences that mask the fault.
- `non_critical`: inferences where the fault alters the output but not the prediction.
- `critical`: inference where the fault is classified as SDC-1, meaning it alters the final prediction.

To perform the fault injection campaigns on the PyTorch models, we used [SFIadvancedmodels](https://github.com/cad-polito-it/SFIadvancedmodels), a fault injector developed by the CAD & Reliability group of the Department of Control and Computer Engineering (DAUIN) of Politecnico di Torino

SHOULD THIS PART BE COMPLETED BY ADDING TENSORFLOW FAULT LISTS AND THE LINK TO POLIMI FI?

## Available Models (so far)

### CIFAR-10 Models
Here is a list of models trained for CIFAR10 dataset, which has images belonging to 10 classes.
All the models are validated using the CIFAR10 validation set that contains 10000 images.

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
All the models are validated using the CIFAR100 validation set that contains 10000 images.

|    Model    | PyTorch TOP-1 Accuracy | Keras TOP-1 Accuracy |
| :---------: | :--------------------: | :------------------: |
|  ResNet18   | <div align="right">76.2 %</div> | <div align="right">76.2 %</div> |
| DenseNet121 | <div align="right">78.7 %</div> | <div align="right">78.7 %</div> |
|  GoogLeNet  | <div align="right">76.3 %</div> | <div align="right">76.3 %</div> |


### GTSRB Models
Here is a list of models trained for GTSRB dataset, containing 43 classes of German Traffic signals.
All the models are validated using the GTSRB validation set that contains 12640 images.

|    Model    | PyTorch TOP-1 Accuracy | Keras TOP-1 Accuracy |
| :---------: | :--------------------: | :------------------: |
|  ResNet20   | <div align="right">94.3%</div> | <div align="right">94.3%</div> |
| DenseNet121 | <div align="right">96.5%</div> | <div align="right">96.5%</div> |
|  Vgg11_bn   | <div align="right">95.5%</div> | <div align="right">95.5%</div> |


