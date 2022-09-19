import time
from datetime import datetime
import random
from pathlib import Path

import GPUtil
import torch
import numpy as np
import pandas as pd


def get_project_dir():
    return Path(__file__).absolute().parent.parent.parent

def setup_run(project_dir, exp_name):
    fmt = "%Y-%m-%dT%H-%M-%S"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = project_dir / "results" / exp_name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def get_device(get_cuda_device_num=False):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if get_cuda_device_num:
            return f"cuda:{set_gpu_device()}"
        else:
            return "cuda"
    else:
        return "cpu"


def set_gpu_device():
    print(GPUtil.showUtilization())
    device_ids = GPUtil.getAvailable(
        order="memory", limit=1, maxLoad=0.3, maxMemory=0.3
    )
    while not device_ids:
        print("No GPU avail.")
        time.sleep(10)
        device_ids = GPUtil.getAvailable(
            order="memory", limit=1, maxLoad=0.3, maxMemory=0.3
        )
    device_id = str(device_ids[0])
    print(f"Using GPU number {device_id}")
    return device_id


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def send_graph_to_device(data, device, groundtruth=None):
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    if not groundtruth is None:
        groundtruth = torch.Tensor(groundtruth).type(torch.LongTensor).to(device)


def get_feature(feature, predictions, embeddings, organ=None):
    if feature == "predictions":
        cell_classes = [cell.id for cell in organ.cells]
        preds = pd.Series(predictions)
        one_hot_preds = pd.get_dummies(preds)
        missing_cells = []
        for cell in cell_classes:
            if cell not in one_hot_preds.columns:
                missing_cells.append(cell)
        for cell in missing_cells:
            one_hot_preds[cell] = 0
        one_hot_preds = one_hot_preds[cell_classes]
        return one_hot_preds.to_numpy()
    elif feature == "embeddings":
        return embeddings
    else:
        raise ValueError(f"No such feature {feature}")
