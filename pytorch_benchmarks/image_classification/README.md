
Download the weights 

```
python download_weights.py https://drive.google.com/drive/folders/1GaDyYLqR8zPio7zat2u-aBIr2uNQCf4t?usp=sharing

```

To run the test set on the network 

```
python3 main.py -d DATASET_NAME -n MODEL_NAME -b BATCH_SIZE 
```

to use GPU if available add

```
python3 main.py -d DATASET_NAME -n MODEL_NAME -b BATCH_SIZE --use-cuda
```