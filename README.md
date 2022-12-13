
## Social Network Analysis Project 2022: Exploring Matrix Factorization for initializing LightGCN to generate Customer-to-Customer Recommendations

This project is based of the the Pytorch implementation of LightGCN:

>SIGIR 2020. Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Author: Prof. Xiangnan He (staff.ustc.edu.cn/~hexn/)

GitHub: https://github.com/gusye1234/LightGCN-PyTorch

## Introduction

In this work, we aim to study the effect of intializing LightGCN with Matrix Factorization embeddings and create a recommendation system for Customer-to-Customer (C2C) Data.

## Enviroment Requirement

`pip install -r requirements.txt`



## Dataset

We provide six processed datasets:

Four Baseline from LightGCN: Gowalla, Yelp2018, Amazon-book and one small dataset LastFM.

Two C2C Datasets: Bonanza and Ebid

### Preprocessing C2C for LightGCN and Matrix Factorization

Preprocessed data is already included in the repo, but if you want to replicate our results fully to combine item and seller into (item, seller) pairs for use with the models run:

`python generate_model_data.py`

### Exploratory Data Analysis

To create degree distribution figures for both C2C datasets run (you might need to tweak the path for dataset if using different directory structure):

* For raw data
`python eda.py`

* For processed data
`python eda_transform.py`

## Train LightGCN
run LightGCN on datasets:

* Gowalla command

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64 --epochs=100`
* Bonanza Command

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="bonanza" --topks="[20]" --recdim=64 --epochs=1000`

* Ebid Command

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="ebid" --topks="[20]" --recdim=64 --epochs=1000`


## Pretrain and run Light GCN with pretrained weights for bonanza (same works for ebid)


### Matrix Factorization with BPR Loss

* Train with matrix factorization method

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="bonanza" --topks="[20]" --recdim=64 --model="mf" --epochs=100 `

* Train Light GCN with pretrained embeddings

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="bonanza" --topks="[20]" --recdim=64 --model="lgn" --pretrain=1 --epochs=900`

### Vanilla Matrix Factorization (Non-BPR):
```shell
usage: Script to train Matrix Factorization pre-training [-h] [--dataset DATASET] [--batch_size BATCH_SIZE] [--latent_space LATENT_SPACE]
                                                         [--epochs EPOCHS] [--logger_file LOGGER_FILE] [--path_to_model PATH_TO_MODEL]
                                                         [--extension EXTENSION] [--delimeter DELIMETER]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset
  --batch_size BATCH_SIZE
                        Batch Size to train the model
  --latent_space LATENT_SPACE
                        Latent Space of MF model
  --epochs EPOCHS       Epochs to run the code for
  --logger_file LOGGER_FILE
                        Path to save logger file
  --path_to_model PATH_TO_MODEL
                        Path to save model file
  --extension EXTENSION
                        file extension, default csv
  --delimeter DELIMETER
                        delimeter for seperating files
```
* Train Vanilla Matrix Factorization method
```shell
python train_embeddings.py --dataset data/ebid --logger_file logs/ebid_mf_100.csv --path_to_model code/code/checkpoints/pmf-ebid-64.pth.tar --delimeter , --extension csv
```

* Train LGCN with Vanilla Matrix Factorization
> You need to set `pretrain_modelname="pmf".
```
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="ebid" --topks="[20]" --recdim=64 --model="lgn" --pretrain=1 --epochs 900
```

## Files and Modules
Here are some important modules and scripts used for the project and a basic description of their usage.
```shell
├── analytics // code for all the analytics ops and utils
├── code // code dir for LightGCN
├── data // pre-processed data for the training ops
├── dataloader // dataloader model for pre-processing
├── eda.py // script to perform eda
├── eda_transform.py // script to perform eda
├── generate_model_data.py // script to pre-process dataset
├── logger.py // logger
├── logs // folder to store experiment logs
├── matrix_factorization //vanilla matrix factorization module
├── plots // folder to store and generate plots
├── requirements.txt
├── train_embeddings.py // script to train vanilla MF models
├── trained_model // folder to store trained models
└── utils.py // helper utils for the root level scripts

9 directories, 7 files
```
