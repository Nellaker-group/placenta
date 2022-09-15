import os
import shutil

import pytest

from graph_train import main as graph_train
from graph_eval import main as graph_eval
from placenta.graphs.enums import *
from placenta.utils.utils import get_project_dir


def get_run_dir(model_type):
    project_dir = get_project_dir()
    exp_dir = project_dir / "results" / model_type / "test"
    run_dir = os.listdir(exp_dir)[0]
    run_dir = exp_dir / run_dir
    return run_dir


def test_train_sage():
    model_type = SupervisedModelsArg.sup_graphsage
    graph_train(
        exp_name="test",
        wsi_ids=[1],
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        model_type=model_type,
        graph_method=MethodArg.intersection,
        batch_size=100,
        num_neighbours=10,
        epochs=2,
        layers=4,
        hidden_units=64,
        groundtruth_tsvs=["wsi_1.tsv"],
        validation_step=1,
    )
    run_dir = get_run_dir(model_type.value)
    assert os.path.exists(run_dir / "1_graph_model.pt")
    assert os.path.exists(run_dir / "final_graph_model.pt")
    assert os.path.exists(run_dir / "graph_train_stats.csv")
    assert os.path.exists(run_dir / "params.csv")

    timestamp_dir = run_dir.parts[-1]
    graph_eval(
        exp_name="test",
        model_weights_dir=timestamp_dir,
        model_type=model_type.value,
        model_name="final_graph_model.pt",
        wsi_id=1,
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        val_patch_files=["all_wsi.csv"],
        graph_method=MethodArg.intersection,
        groundtruth_tsv="wsi_1.tsv",
    )
    assert os.path.exists(run_dir / "eval")
    shutil.rmtree(run_dir)


def test_train_clustergcn():
    model_type = SupervisedModelsArg.sup_clustergcn
    graph_train(
        exp_name="test",
        wsi_ids=[1],
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        model_type=model_type,
        graph_method=MethodArg.intersection,
        batch_size=100,
        num_neighbours=10,
        epochs=2,
        layers=4,
        hidden_units=64,
        groundtruth_tsvs=["wsi_1.tsv"],
        validation_step=1,
    )
    run_dir = get_run_dir(model_type.value)
    assert os.path.exists(run_dir / "1_graph_model.pt")
    assert os.path.exists(run_dir / "final_graph_model.pt")
    assert os.path.exists(run_dir / "graph_train_stats.csv")
    assert os.path.exists(run_dir / "params.csv")

    timestamp_dir = run_dir.parts[-1]
    graph_eval(
        exp_name="test",
        model_weights_dir=timestamp_dir,
        model_type=model_type.value,
        model_name="final_graph_model.pt",
        wsi_id=1,
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        val_patch_files=["all_wsi.csv"],
        graph_method=MethodArg.intersection,
        groundtruth_tsv="wsi_1.tsv",
    )
    assert os.path.exists(run_dir / "eval")
    shutil.rmtree(run_dir)


def test_train_saint():
    model_type = SupervisedModelsArg.sup_graphsaint_rw
    graph_train(
        exp_name="test",
        wsi_ids=[1],
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        model_type=model_type,
        graph_method=MethodArg.intersection,
        batch_size=100,
        num_neighbours=10,
        epochs=2,
        layers=4,
        hidden_units=64,
        groundtruth_tsvs=["wsi_1.tsv"],
        validation_step=1,
    )

    run_dir = get_run_dir(model_type.value)
    assert os.path.exists(run_dir / "1_graph_model.pt")
    assert os.path.exists(run_dir / "final_graph_model.pt")
    assert os.path.exists(run_dir / "graph_train_stats.csv")
    assert os.path.exists(run_dir / "params.csv")

    timestamp_dir = run_dir.parts[-1]
    graph_eval(
        exp_name="test",
        model_weights_dir=timestamp_dir,
        model_type=model_type.value,
        model_name="final_graph_model.pt",
        wsi_id=1,
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        val_patch_files=["all_wsi.csv"],
        graph_method=MethodArg.intersection,
        groundtruth_tsv="wsi_1.tsv",
    )
    assert os.path.exists(run_dir / "eval")
    shutil.rmtree(run_dir)


def test_train_sign():
    model_type = SupervisedModelsArg.sup_sign
    graph_train(
        exp_name="test",
        wsi_ids=[1],
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        model_type=model_type,
        graph_method=MethodArg.intersection,
        batch_size=100,
        num_neighbours=10,
        epochs=2,
        layers=4,
        hidden_units=64,
        groundtruth_tsvs=["wsi_1.tsv"],
        validation_step=1,
    )

    run_dir = get_run_dir(model_type.value)
    assert os.path.exists(run_dir / "1_graph_model.pt")
    assert os.path.exists(run_dir / "final_graph_model.pt")
    assert os.path.exists(run_dir / "graph_train_stats.csv")
    assert os.path.exists(run_dir / "params.csv")

    timestamp_dir = run_dir.parts[-1]
    graph_eval(
        exp_name="test",
        model_weights_dir=timestamp_dir,
        model_type=model_type.value,
        model_name="final_graph_model.pt",
        wsi_id=1,
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        val_patch_files=["all_wsi.csv"],
        graph_method=MethodArg.intersection,
        groundtruth_tsv="wsi_1.tsv",
    )
    assert os.path.exists(run_dir / "eval")
    shutil.rmtree(run_dir)


def test_train_mlp():
    model_type = SupervisedModelsArg.sup_mlp
    graph_train(
        exp_name="test",
        wsi_ids=[1],
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        model_type=model_type,
        graph_method=MethodArg.intersection,
        batch_size=100,
        num_neighbours=10,
        epochs=2,
        layers=4,
        hidden_units=64,
        groundtruth_tsvs=["wsi_1.tsv"],
        validation_step=1,
    )

    run_dir = get_run_dir(model_type.value)
    assert os.path.exists(run_dir / "1_graph_model.pt")
    assert os.path.exists(run_dir / "final_graph_model.pt")
    assert os.path.exists(run_dir / "graph_train_stats.csv")
    assert os.path.exists(run_dir / "params.csv")

    timestamp_dir = run_dir.parts[-1]
    graph_eval(
        exp_name="test",
        model_weights_dir=timestamp_dir,
        model_type=model_type.value,
        model_name="final_graph_model.pt",
        wsi_id=1,
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        val_patch_files=["all_wsi.csv"],
        graph_method=MethodArg.intersection,
        groundtruth_tsv="wsi_1.tsv",
    )
    assert os.path.exists(run_dir / "eval")
    shutil.rmtree(run_dir)


def test_train_gat():
    model_type = SupervisedModelsArg.sup_gat
    graph_train(
        exp_name="test",
        wsi_ids=[1],
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        model_type=model_type,
        graph_method=MethodArg.intersection,
        batch_size=100,
        num_neighbours=10,
        epochs=2,
        layers=4,
        hidden_units=64,
        groundtruth_tsvs=["wsi_1.tsv"],
        validation_step=1,
    )

    run_dir = get_run_dir(model_type.value)
    assert os.path.exists(run_dir / "1_graph_model.pt")
    assert os.path.exists(run_dir / "final_graph_model.pt")
    assert os.path.exists(run_dir / "graph_train_stats.csv")
    assert os.path.exists(run_dir / "params.csv")

    timestamp_dir = run_dir.parts[-1]
    graph_eval(
        exp_name="test",
        model_weights_dir=timestamp_dir,
        model_type=model_type.value,
        model_name="final_graph_model.pt",
        wsi_id=1,
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        val_patch_files=["all_wsi.csv"],
        graph_method=MethodArg.intersection,
        groundtruth_tsv="wsi_1.tsv",
    )
    assert os.path.exists(run_dir / "eval")
    shutil.rmtree(run_dir)


def test_train_gatv2():
    model_type = SupervisedModelsArg.sup_gatv2
    graph_train(
        exp_name="test",
        wsi_ids=[1],
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        model_type=model_type,
        graph_method=MethodArg.intersection,
        batch_size=100,
        num_neighbours=10,
        epochs=2,
        layers=4,
        hidden_units=64,
        groundtruth_tsvs=["wsi_1.tsv"],
        validation_step=1,
    )

    run_dir = get_run_dir(model_type.value)
    assert os.path.exists(run_dir / "1_graph_model.pt")
    assert os.path.exists(run_dir / "final_graph_model.pt")
    assert os.path.exists(run_dir / "graph_train_stats.csv")
    assert os.path.exists(run_dir / "params.csv")

    timestamp_dir = run_dir.parts[-1]
    graph_eval(
        exp_name="test",
        model_weights_dir=timestamp_dir,
        model_type=model_type.value,
        model_name="final_graph_model.pt",
        wsi_id=1,
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        val_patch_files=["all_wsi.csv"],
        graph_method=MethodArg.intersection,
        groundtruth_tsv="wsi_1.tsv",
    )
    assert os.path.exists(run_dir / "eval")
    shutil.rmtree(run_dir)


def test_train_shadow():
    model_type = SupervisedModelsArg.sup_shadow
    graph_train(
        exp_name="test",
        wsi_ids=[1],
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        model_type=model_type,
        graph_method=MethodArg.intersection,
        batch_size=100,
        num_neighbours=10,
        epochs=2,
        layers=4,
        hidden_units=64,
        groundtruth_tsvs=["wsi_1.tsv"],
        validation_step=1,
    )

    run_dir = get_run_dir(model_type.value)
    assert os.path.exists(run_dir / "1_graph_model.pt")
    assert os.path.exists(run_dir / "final_graph_model.pt")
    assert os.path.exists(run_dir / "graph_train_stats.csv")
    assert os.path.exists(run_dir / "params.csv")

    timestamp_dir = run_dir.parts[-1]
    graph_eval(
        exp_name="test",
        model_weights_dir=timestamp_dir,
        model_type=model_type.value,
        model_name="final_graph_model.pt",
        wsi_id=1,
        x_min=93481,
        y_min=8540,
        width=5000,
        height=5000,
        val_patch_files=["all_wsi.csv"],
        graph_method=MethodArg.intersection,
        groundtruth_tsv="wsi_1.tsv",
    )
    assert os.path.exists(run_dir / "eval")
    shutil.rmtree(run_dir)
