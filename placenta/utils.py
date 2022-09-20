import time
from datetime import datetime
import random
from pathlib import Path

import GPUtil
import torch
import numpy as np


def get_project_dir():
    return Path(__file__).absolute().parent.parent

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
