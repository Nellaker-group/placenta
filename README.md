# A New Graph Node Classification Benchmark: Learning Structure from Histology Cell Graphs

## Overview

Accompanying repository for **A New Graph Node Classification Benchmark: 
Learning Structure from Histology Cell Graphs (Neurips 2022)**. 

This repo contains the PyTorch and PyTorch Geometric implementation for downloading 
and processing the dataset, *Placenta*, and running all main experiments.


Images of cell graph thumbnails and zoomed in histology -> overlay -> graph -> GT


## Installation

Our codebase is writen in python=3.7.2. 

We recommend installation from source using the MakeFile which will install all 
requirements:
```bash
git clone git@github.com:Nellaker-group/placenta.git
cd placenta
# Activate conda or venv environment with python installation:
# e.g. conda create -y -n placenta python=3.7.2
#      conda activate placenta
make environment_cu113
```

## Dataset

The dataset consists of two cell graphs constructed across two placenta histology
whole slide images.




## Training


### Training with Default Cell Graphs


### Training with Custom Cell Graphs


## Inference


## Experiments

Performance on test data of existing scalable architectures:

| Model Architecture  | Accuracy | Top 2 Accuracy | ROC AUC |
| ------------- | ------------- | ------------- | ------------- |
| MLP  | 49.50±0.00  | 76.21±0.00 | 0.811±0.000 |
| GraphSAGE-mean  | 64.88±0.43  | 88.94±0.38 | 0.883±0.005 |
| ClusterGCN  | 64.24±1.21  | 88.26±0.82 | 0.882±0.006 |
| GraphSAINT-rw  | 63.94±0.23  | 87.86±0.15 | 0.895±0.002 |
| SIGN  | 71.84±0.00  | 92.40±0.00 | 0.970±0.000 |
| ShaDow  | 63.04±0.77  | 86.88±0.74 | 0.863±0.008 |
| ClusterGAT  | 58.28±0.07  | 83.76±0.07 | 0.851±0.002 |
| ClusterGATv2  | 52.99±0.96  | 78.02±1.43 | 0.834±0.004 |


Training configurations:

```bash
python train.py --exp-name mlp_train --model-type mlp --batch-size 51200 --num-neighbours 0
python train.py --exp-name graphsage_train --model-type graphsage --batch-size 32000 --num-neighbours 10 --layers 12
python train.py --exp-name clustergcn_train --model-type clustergcn --batch-size 200 --num-neighbours 400
python train.py --exp-name graphsaint_train --model-type graphsaint --batch-size 32000 --num-neighbours 500
python train.py --exp-name sign_train --model-type sign --batch-size 51200 --num-neighbours 10
python train.py --exp-name shadow_train --model-type shadow --batch-size 4000 --num-neighbours 5 --layers 8
python train.py --exp-name gat_train --model-type gat --batch-size 200 --num-neighbours 400 --layers 2
python train.py --exp-name gatv2_train --model-type gatv2 --batch-size 200 --num-neighbours 400 --layers 2
```