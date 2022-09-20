# A New Graph Node Classification Benchmark: Learning Structure from Histology Cell Graphs

## Overview

Accompanying repository for **A New Graph Node Classification Benchmark: 
Learning Structure from Histology Cell Graphs (Neurips 2022)**. 

This repo contains the PyTorch and PyTorch Geometric implementation for downloading 
and processing the dataset, *Placenta*, and running all main paper experiments.


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
The make command will run the following:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install -r requirements.txt
pip install -e .
```
If you would rather install a different version of pytorch for your cuda version, 
please change the first two lines as per library instructions.

## Dataset

The dataset consists of two cell graphs constructed across two placenta histology
whole slide images, combined into one graph. The nodes of the graph represent cells 
and the edges represent interactions between cells. 

| Nodes     | Edges     | Labels  | Node Features | Classes |
|-----------|-----------|---------|---------------|---------|
| 2,395,747 | 5,486,089 | 799,745 | 64            | 9       |

The raw data can be downloaded, processed, and 
loaded into memory using `placenta.data.dataset.Placenta(root)`. 

The raw data will be downloaded to `root/raw`. It consists of two hdf5 files 
`wsi_1.hdf5` and `wsi_2.hdf5` containing the cell features and coordinates, 
two tsv files `wsi_1.tsv` and `wsi_2.tsv` containing the ground truth labels, 
and two csv files `val_patches.csv` and `test_patches.csv` containing the validation 
and test patch coordinates.

The raw data will be processed into two `torch_geometric.data.Data` objects and saved
into `root/processed`. When loading the Placenta dataset into memory, both processed 
graphs will be combined into a single `torch_geometric.data.DataBatch` object.

Nodes without ground truth labels are masked in `data.unlabelled_mask`, and the 
remaining train, validation, and test nodes are masked in `data.train_mask`, 
`data.val_mask`, and `data.test_mask` respectively.

### Custom Cell Graphs

We recommend graph construction as outlined in our paper, however, we offer some
alternative graph constructions. The `placenta.data.dataset.Placenta` class can be 
passed a `GraphConstructor` object which determines how the cell graph is made. 
Along with the `DefaultGraphConstructor`, we offer `DelaunyGraphConstructor` and 
`KNNGraphConstructor` which construct the edges using the Delauny triangulation and 
k-nearest neighbours algorithms respectively.

If you wish to define an entirely new graph from the raw data, you can create a
new `GraphConstructor`. This could be to define new edges, change the node features,
or change the data splits, and so on. 

When using a new `GraphConstructor`, you will need to remove (or rename) the old 
processed data.

## Training

Models can be trained for inductive node classification using `train.py`. This will
use the default graph construction and will train across both WSI 1 and WSI 2. During
the validation step, the model will be evaluated on validation data and training data
using the validation sampling alogrithm for that model type.

```bash
python train.py --exp-name graphsage_train --model-type graphsage --batch-size 32000 --num-neighbours 10 --layers 12
```

## Evaluation and Inference

Models can be evaluated on the test or validation set using `eval.py`. This will use the 
default graph construction for WSI 1, of which the these sets are defined. Along with
performance metrics, this will generate confusion matrix and precision-recall 
plots to see performance across classes.

```bash
python eval.py --exp-name graphsage_train --run-time-stamp 2022-09-20T16-00-53 --model-name 400_graph_model.pt --model-type graphsage --use-test-set
```

You may use `inference.py` to performance inference across an area of a WSI. This
will save a tsv of predictions per coordinate and produce a png to visualise these 
predictions.

```bash
python inference.py --exp-name graphsage_train --run-time-stamp 2022-09-20T16-00-53 --model-name 400_graph_model.pt --model-type graphsage --wsi-id 1 --x-min 93481 --y-min 8540 --width 5000 --height 5000
```

## Experiments

Performance on test data using existing scalable architectures:

| Model Architecture | Accuracy   | Top 2 Accuracy | ROC AUC     |
|--------------------|------------|----------------|-------------|
| MLP                | 47.98±0.79 | 75.22±0.92     | 0.750±0.003 |
| GraphSAGE-mean     | 64.88±0.43 | 88.94±0.38     | 0.883±0.005 |
| ClusterGCN         | 64.24±1.21 | 88.26±0.82     | 0.882±0.006 |
| GraphSAINT-rw      | 63.94±0.23 | 87.86±0.15     | 0.895±0.002 |
| SIGN               | 64.77±0.43 | 88.32±0.42     | 0.886±0.002 |
| ShaDow             | 63.04±0.77 | 86.88±0.74     | 0.863±0.008 |
| ClusterGAT         | 58.07±0.61 | 83.43±0.96     | 0.851±0.002 |
| ClusterGATv2       | 57.07±0.65 | 83.21±0.55     | 0.854±0.005 |


Training configurations for these experiments:

```bash
python train.py --exp-name mlp_train --model-type mlp --batch-size 51200 --num-neighbours 0
python train.py --exp-name graphsage_train --model-type graphsage --batch-size 32000 --num-neighbours 10 --layers 12
python train.py --exp-name clustergcn_train --model-type clustergcn --batch-size 200 --num-neighbours 400
python train.py --exp-name graphsaint_train --model-type graphsaint --batch-size 32000 --num-neighbours 500
python train.py --exp-name sign_train --model-type sign --batch-size 51200 --num-neighbours 10
python train.py --exp-name shadow_train --model-type shadow --batch-size 4000 --num-neighbours 5 --layers 8
python train.py --exp-name gat_train --model-type gat --batch-size 200 --num-neighbours 400 --layers 2 --dropout 0.25
python train.py --exp-name gatv2_train --model-type gatv2 --batch-size 200 --num-neighbours 400 --layers 2 --dropout 0.25
```

## Visualisation

Ground truth points for a region of a WSI can be plotted using 
`placenta/analysis/vis_groundtruth.py`. The plot will be saved to 
`visualisations/groundtruth/wsi_{wsi_id}/x{x_min}_y{y_min}_w{width}_h{height}.png`.
