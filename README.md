# Deep Neural Network Models for Reliability Studies
Welcome to the repository containing state-of-the-art Deep Neural Network (DNN) models implemented in both PyTorch and TensorFlow for conducting reliability studies. 

## Project Collaboration

This project is a collaboration between the following institutions:

- [Politecnico di Torino](https://www.polito.it/)
- [Politecnico di Milano](https://www.polimi.it/)
- [Ecole Centrale de Lyon](https://www.ec-lyon.fr/en)


## Getting Started
The idea of this project is to share neural networks to conduct reliability studies. The goal is to make it easier and more accessible for different research groups to compare their results.

Within the repository, you will find the code and weights for some PyTorch models for image classification (so far), pre-trained on the CIFAR10, CIFAR100, and GTSRB datasets. Additionally, we have converted the PyTorch models to their Keras counterparts, which share the same architecture, weights, and similar accuracy. You can find the scripts for converting in the following repository [dnn-benchmarks-converter](https://github.com/D4De/dnn-benchmarks-converter). Such scripts are based on [nobuco](https://github.com/AlexanderLutsenko/nobuco); a public backup fork of the converter is available also here [nobuco-fork](https://github.com/D4De/nobuco).

For each model, a fault list has been generated, and a fault injection campaign has been conducted to evaluate the reliability and comparability between the PyTorch and Keras versions. [dnn-benchmarks-converter](https://github.com/D4De/dnn-benchmarks-converter) repository also contains the scripts for converting the fault lists from PyTorch to Keras. For further details, you can refer to the paper \cite{} submitted to TCAD. All fault lists are included in the repository, along with some of the results from the injection campaigns.


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
4. Clone the repository

Given the great size of the checkpoints of the weights, this repo uses git-lfs. 
This means that to clone the repo **with the weights** you need to install git-lfs.
In general, you just need to install the git-lfs package on your distribution (something like `apt install git-lsf`)
and then install the plugin on git using the command `git lsf install`.
For more information [check this link](https://git-lfs.com/)
After doing that, simply clone the repo:

```bash
git clone https://github.com/cad-polito-it/dnn-benchmarks
```


## Projects structure

The repository is organized into two main directories:
- ```torch/```: This directory includes folders organized by hardware type and task, such as ```gpu/image_classification/```, with each containing subdirectories for different data representation and reference datasets. Each dataset-specific folder holds the code, the fault list and the pretrained weights for the corresponding PyTorch models
- ```tensorflow/```: has the same structure as the previous directory but for the Keras models.


## Dataset transformations description

The following transformations are applied for image preprocessing with each dataset in PyTorch and TensorFlow, respectively, ensuring the input data is appropriately augmented for training and prepared for testing.

### Transformations in PyTorch

**CIFAR10**
```
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       
        transforms.RandomHorizontalFlip(),                                          
        transforms.ToTensor(),                                                      
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   
])
transform_test = transforms.Compose([
    transforms.CenterCrop(32),                                                  
    transforms.ToTensor(),                                                      
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   
])
```
**CIFAR100**
```
transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
])
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
])
```
**GTSRB**
```
transform_train = Compose([
    ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
    RandomEqualize(0.4),
    AugMix(),
    RandomHorizontalFlip(0.3),
    RandomVerticalFlip(0.3),
    GaussianBlur((3,3)),
    RandomRotation(30),
    
    Resize([50,50]),
    ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))
    
])
transform_test = Compose([
    Resize([50, 50]),
    ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214), 
                            (0.2724, 0.2608, 0.2669)),
])
```
**PASCAL VOC 2012**
```

val_transform = transforms.Compose([
                transforms.Resize((520, 520)),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

])
    
t_transform = transforms.Compose([
ToImage(),
Resize((520, 520), interpolation = InterpolationMode.NEAREST),

])
```

### Transformations in TensorFlow

> [!NOTE] NumPy functions have been used in place of native TensorFlow ones to obtain exactly the same internal representation of the manipulated dataset in the two environments.

**CIFAR10**
```
image = image / np.float32(255.0)
image = (image - (0.4914, 0.4822, 0.4465)) / (0.2023, 0.1994, 0.2010)
```

**CIFAR100**
```
image = image / np.float32(255.0)
image = (image - (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)) / (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
```

**GTSRB**
```
image = tf.image.resize(image, [50, 50]).numpy()
image = image / np.float32(255.0)
image = (image - (0.3403, 0.3121, 0.3214)) / (0.2724, 0.2608, 0.2669)
```


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

To perform the fault injection campaigns on the Tensorflow framework, we used a [Keras weight Injector](https://github.com/D4De/keras_weight_injector), developed by research team on Dependable Computing Systems at Politecnico di Milano



## Available Models 

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

### COCO (restricted to only PASCAL VOC 2012 classes) Models
Here is a list of models trained for COCO dataset, but restricted to only the 20 PASCAL VOC 2012 segmentation dataset classes.
All the models are validated using the PASCAL VOC 2012 validation set that contains 1449 images.

|    Model    | PyTorch Pixel Accuracy | Keras Pixel Accuracy | PyTorch IoU Accuracy | Keras IoU Accuracy |
| :---------: | :--------------------: | :------------------: | :------------------: | :------------------: |
|  DeepLabV3   | <div align="right">89.2%</div> | <div align="right">89.2%</div> | <div align="right">71.1%</div> | <div align="right">71.1%</div> |

# Acknowledgments
This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions, neither the European Union nor the European Commission can be considered responsible for them.




