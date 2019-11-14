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

## Reproduce the experimental results by the pre-trained model
```
cd ./Dialogue/
python ./run.py --task "ubuntu"
python ./run.py --task "douban"
python ./run.py --task "alime"
```

## Train a new model
```
cd ./Dialogue/
python ./run.py --task "ubuntu" --is_training True
python ./run.py --task "douban" --is_training True
python ./run.py --task "alime" --is_training True
```
The training process is recorded in ```log/[ubuntu|douban|alime].msn.log``` file.


## Citation
If you find this code useful in your research, please cite our paper:
```
@inproceedings{yuan2019multi,
  title={Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots},
  author={Yuan, Chunyuan and Zhou, Wei and Li, Mingming and Lv, Shangwen and Zhu, Fuqing and Han, Jizhong and Hu, Songlin},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={111--120},
  year={2019}
}
```
