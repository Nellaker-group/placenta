import os
import shutil

import pytest

from train import train
from eval import eval_model
from enums import *
from utils import get_project_dir, setup_run, get_device, set_seed
from placenta.runners.train_runner import TrainParams, TrainRunner
from placenta.runners.eval_runner import EvalParams, EvalRunner
from placenta.logger.logger import Logger
from organs import Placenta as organ
from dataset import Placenta, get_nodes_within_tiles


@pytest.fixture
def dataset():
    project_dir = get_project_dir()
    yield Placenta(project_dir / "datasets")


@pytest.fixture
def data(dataset):
    data = dataset[0]
    mask = get_nodes_within_tiles(
        (93481, 8540), 5000, 5000, data.pos[:, 0], data.pos[:, 1]
    )
    yield data.subgraph(mask)


@pytest.fixture
def device():
    yield get_device()


@pytest.fixture
def logger():
    yield Logger(list(["train", "train_inf", "val"]), ["loss", "accuracy"], file=True)


@pytest.fixture
def run_params(data, device):
    yield TrainParams(
        data,
        device,
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


def test_train_sage(dataset, logger, run_params):
    model_type = ModelsArg.graphsage
    exp_name = "test"
    set_seed(0)
    project_dir = get_project_dir()

    run_params.model_type = model_type
    train_runner = TrainRunner.new(run_params)

    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
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
        model_type,
        512,
        organ,
    )
    eval_runner = EvalRunner.new(eval_params)

    eval_model(eval_runner, False, save_path)
    assert os.path.exists(run_path / "eval")
    shutil.rmtree(run_path)


def test_train_clustergcn(dataset, logger, run_params):
    model_type = ModelsArg.clustergcn
    exp_name = "test"
    set_seed(0)
    project_dir = get_project_dir()

    run_params.model_type = model_type
    train_runner = TrainRunner.new(run_params)

    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
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
        model_type,
        512,
        organ,
    )
    eval_runner = EvalRunner.new(eval_params)

    eval_model(eval_runner, False, save_path)
    assert os.path.exists(run_path / "eval")
    shutil.rmtree(run_path)


def test_train_saint(dataset, logger, run_params):
    model_type = ModelsArg.graphsaint
    exp_name = "test"
    set_seed(0)
    project_dir = get_project_dir()

    run_params.model_type = model_type
    train_runner = TrainRunner.new(run_params)

    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
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
        model_type,
        512,
        organ,
    )
    eval_runner = EvalRunner.new(eval_params)

    eval_model(eval_runner, False, save_path)
    assert os.path.exists(run_path / "eval")
    shutil.rmtree(run_path)


def test_train_sign(dataset, logger, run_params):
    model_type = ModelsArg.sign
    exp_name = "test"
    set_seed(0)
    project_dir = get_project_dir()

    run_params.model_type = model_type
    train_runner = TrainRunner.new(run_params)

    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
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
        model_type,
        512,
        organ,
    )
    eval_runner = EvalRunner.new(eval_params)

    eval_model(eval_runner, False, save_path)
    assert os.path.exists(run_path / "eval")
    shutil.rmtree(run_path)


def test_train_mlp(dataset, logger, run_params):
    model_type = ModelsArg.mlp
    exp_name = "test"
    set_seed(0)
    project_dir = get_project_dir()

    run_params.model_type = model_type
    train_runner = TrainRunner.new(run_params)

    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
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
        model_type,
        512,
        organ,
    )
    eval_runner = EvalRunner.new(eval_params)

    eval_model(eval_runner, False, save_path)
    assert os.path.exists(run_path / "eval")
    shutil.rmtree(run_path)


def test_train_gat(dataset, logger, run_params):
    model_type = ModelsArg.gat
    exp_name = "test"
    set_seed(0)
    project_dir = get_project_dir()

    run_params.model_type = model_type
    train_runner = TrainRunner.new(run_params)

    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
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
        model_type,
        512,
        organ,
    )
    eval_runner = EvalRunner.new(eval_params)

    eval_model(eval_runner, False, save_path)
    assert os.path.exists(run_path / "eval")
    shutil.rmtree(run_path)


def test_train_gatv2(dataset, logger, run_params):
    model_type = ModelsArg.gatv2
    exp_name = "test"
    set_seed(0)
    project_dir = get_project_dir()

    run_params.model_type = model_type
    train_runner = TrainRunner.new(run_params)

    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
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
        model_type,
        512,
        organ,
    )
    eval_runner = EvalRunner.new(eval_params)

    eval_model(eval_runner, False, save_path)
    assert os.path.exists(run_path / "eval")
    shutil.rmtree(run_path)


def test_train_shadow(dataset, logger, run_params):
    model_type = ModelsArg.shadow
    exp_name = "test"
    set_seed(0)
    project_dir = get_project_dir()

    run_params.model_type = model_type
    train_runner = TrainRunner.new(run_params)

    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
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
        model_type,
        512,
        organ,
    )
    eval_runner = EvalRunner.new(eval_params)

    eval_model(eval_runner, False, save_path)
    assert os.path.exists(run_path / "eval")
    shutil.rmtree(run_path)
