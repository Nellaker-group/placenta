import os
import shutil

import pytest

from train import train
from eval import eval_model
from placenta.enums import ModelsArg
from placenta.utils import get_project_dir, setup_run, get_device, set_seed
from placenta.runners.train_runner import TrainParams, TrainRunner
from placenta.runners.eval_runner import EvalParams, EvalRunner
from placenta.logger.logger import Logger
from placenta.organs import Placenta as organ
from placenta.dataset import (
    Placenta,
    get_nodes_within_tiles,
    DelaunayGraphConstructor,
    KNNGraphConstructor,
    OneHotGraphConstructor,
)


@pytest.fixture
def logger():
    yield Logger(list(["train", "train_inf", "val"]), ["loss", "accuracy"], file=True)


def get_data_patch(dataset):
    data = dataset[0]
    mask = get_nodes_within_tiles(
        (93481, 8540), 5000, 5000, data.pos[:, 0], data.pos[:, 1]
    )
    return data.subgraph(mask)


def get_run_params(data):
    return TrainParams(
        data,
        get_device(),
        None,
        ModelsArg.graphsage,
        100,
        10,
        2,
        4,
        64,
        0.5,
        0.001,
        0,
        True,
        True,
        1,
        organ,
    )


def _eval_path(model_name, pretrained_path):
    model_epochs = (
        "model_final"
        if model_name == "graph_model.pt"
        else f"model_{model_name.split('_')[0]}"
    )
    save_path = pretrained_path / "eval" / model_epochs
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def train_and_eval(run_params, exp_name, dataset, logger):
    project_dir = get_project_dir()
    train_runner = TrainRunner.new(run_params)

    run_path = setup_run(project_dir, f"{run_params.model_type}/{exp_name}")
    dataset.save_params(run_path)
    train_runner.params.save(0, exp_name, run_path)

    train(train_runner, logger, run_path)

    assert os.path.exists(run_path / "1_graph_model.pt")
    assert os.path.exists(run_path / "final_graph_model.pt")
    assert os.path.exists(run_path / "graph_train_stats.csv")
    assert os.path.exists(run_path / "graph_params.csv")
    assert os.path.exists(run_path / "train_params.csv")

    pretrained_path = run_path / "final_graph_model.pt"
    save_path = _eval_path("final_graph_model.pt", run_path)

    eval_params = EvalParams(
        train_runner.params.data,
        train_runner.params.device,
        str(pretrained_path),
        run_params.model_type,
        512,
        organ,
    )
    eval_runner = EvalRunner.new(eval_params)

    eval_model(eval_runner, False, save_path)
    assert os.path.exists(run_path / "eval")
    shutil.rmtree(run_path)


def test_knn(logger):
    project_dir = get_project_dir()
    gc = KNNGraphConstructor(project_dir / "datasets" / "raw", k=5)
    dataset = Placenta(get_project_dir() / "datasets", graph_constructor=gc)
    data = get_data_patch(dataset)
    run_params = get_run_params(data)
    exp_name = "test"
    set_seed(0)
    train_and_eval(run_params, exp_name, dataset, logger)


def test_delaunay(logger):
    project_dir = get_project_dir()
    gc = DelaunayGraphConstructor(project_dir / "datasets" / "raw")
    dataset = Placenta(get_project_dir() / "datasets", graph_constructor=gc)
    data = get_data_patch(dataset)
    run_params = get_run_params(data)
    exp_name = "test"
    set_seed(0)
    train_and_eval(run_params, exp_name, dataset, logger)


def test_one_hot(logger):
    project_dir = get_project_dir()
    gc = OneHotGraphConstructor(project_dir / "datasets" / "raw")
    dataset = Placenta(get_project_dir() / "datasets", graph_constructor=gc)
    data = get_data_patch(dataset)
    run_params = get_run_params(data)
    exp_name = "test"
    set_seed(0)
    train_and_eval(run_params, exp_name, dataset, logger)
