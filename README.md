# Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots
This repository contains the source code and datasets for the EMNLP 2019 paper [Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots]. <br>


## Dependencies
Python 3.x <br>
Pytorch 1.1.0

## Datasets and Trained Models
Your can download the processed datasets and model checkpoints for reproduce the experimental results in the paper. <br>
https://www.dropbox.com/sh/18gncqj296swzjg/AABtrQUpBi3JORai8zimNpNCa?dl=0

Unzip the dataset.rar file to the folder of ```dataset``` and unzip the checkpoint.rar file to the folder of ```checkpoint```.


## Train a new model
Take Ubuntu_V1 as an example.
```
cd scripts
bash ubuntu_train.sh
```
The training process is recorded in ```log_train_IMN_UbuntuV1.txt``` file.

## Test a trained model
```
bash ubuntu_test.sh
```
The testing process is recorded in ```log_test_IMN_UbuntuV1.txt``` file. And your can get a ```ubuntu_test_out.txt``` file which records scores for each context-response pair. Run the following command and you can compute the metric of Recall.
```
python compute_recall.py
```

## Cite
If you use the code and datasets, please cite the following paper:
**"Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots"**
Jia-Chen Gu, Zhen-Hua Ling, Quan Liu. _CIKM (2019)_

```
@inproceedings{gu2019interactive,
  title        = {Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots},
  author       = {Jia{-}Chen Gu and
                  Zhen{-}Hua Ling and
                  Quan Liu},
  booktitle    = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  year         = {2019},
  organization = {ACM}
}
```
