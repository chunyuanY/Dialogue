# Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots
This repository contains the source code and datasets for the EMNLP 2019 paper [Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots]. <br>


## Dependencies
Python 3.x <br>
Pytorch 1.1.0

## Datasets and Trained Models
Your can download the processed datasets and the checkpoints of trained models for reproduce the experimental results in the paper by the following url: <br>
https://drive.google.com/drive/folders/1pJKIppcbjuTZxbTc8ye5mfnC2ygR2xTo?usp=sharing

Unzip the dataset.rar file to the folder of ```dataset``` and unzip the checkpoint.rar file to the folder of ```checkpoint```. <br>
The ```log/``` directory already contains the training and evaluation logs of each dataset.

## Train a new model
```
cd ./Dialogue/
python ./run.py --task "ubuntu" --is_training True
python ./run.py --task "douban" --is_training True
python ./run.py --task "alime" --is_training True
```
The training process is recorded in ```log/[ubuntu|douban|alime].msn.log``` file.

## Test a trained model
```
python ./run.py --task "ubuntu"
python ./run.py --task "douban"
python ./run.py --task "alime"
```

