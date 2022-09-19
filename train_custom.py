from typing import Optional, List

import typer
from torch_geometric.transforms import SIGN
from torch_geometric.data import Batch

from placenta.organs.organs import Placenta
from placenta.logger.logger import Logger
from placenta.utils.utils import setup_run, get_device
from placenta.graphs.enums import FeatureArg, MethodArg, ModelsArg
from placenta.graphs.graph_supervised import setup_node_splits, collect_params
from placenta.runners.train_runner import TrainParams, TrainRunner
from placenta.data.process_data import get_data
from placenta.utils.utils import send_graph_to_device, set_seed, get_project_dir


def main(
    seed: int = 0,
    exp_name: str = typer.Option(...),
    wsi_ids: List[int] = typer.Option([]),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 5,
    feature: FeatureArg = FeatureArg.embeddings,
    pretrained: Optional[str] = None,
    model_type: ModelsArg = typer.Option(...),
    graph_method: MethodArg = MethodArg.k,
    batch_size: int = typer.Option(...),
    num_neighbours: int = typer.Option(...),
    epochs: int = typer.Option(...),
    layers: int = typer.Option(...),
    hidden_units: int = 256,
    dropout: float = 0.5,
    learning_rate: float = 0.001,
    weighted_loss: bool = True,
    use_custom_weights: bool = True,
    groundtruth_tsvs: List[str] = typer.Option([]),
    val_patch_files: Optional[List[str]] = [],
    test_patch_files: Optional[List[str]] = [],
    validation_step: int = 100,
    verbose: bool = True,
):
    """
    Train a model on the placenta dataset using a custom cell graph construction.
    :param seed: set the random seed
    :param exp_name: name of the experiment (used for saving the model)
    :param wsi_ids: list of WSI ids to use for training
    :param x_min: x coordinate of the top left corner of the region to use in both WSIs
    :param y_min: y coordinate of the top left corner of the region to use in both WSIs
    :param width: width of the region to use in both WSIs (-1 for full width)
    :param height: height of the region to use in both WSIs (-1 for full height)
    :param k: number of neighbours per edge for knn and intersection graphs
    :param feature: node features, either embeddings vectors or one hot cell types
    :param pretrained: path to a pretrained model
    :param model_type: type of model to train, one of ModelsArg
    :param batch_size: batch size
    :param num_neighbours: neighbours per hop or cluster size or sample coverage
    :param epochs: max number of epochs to train for
    :param layers: number of model layers (and usually hops)
    :param hidden_units: number of hidden units per layer
    :param dropout: dropout rate on each layer and attention heads
    :param learning_rate: learning rate
    :param weighted_loss: whether to used weighted cross entropy
    :param use_custom_weights: whether to use custom weights for the loss
    :param groundtruth_tsvs: list groundtruth tsv file names, order matches wsi_ids
    :param val_patch_files: list of validation patch files
    :param test_patch_files: list of test patch files
    :param validation_step: number of steps between validation check
    :param verbose: whether to print graph setup
    """
    organ = Placenta
    graph_method = graph_method.value
    feature = feature.value
    device = get_device()
    set_seed(seed)
    project_dir = get_project_dir()
    pretrained_path = project_dir / pretrained if pretrained else None

    val_patch_files = [
        project_dir / "datasets" / "raw" / file for file in val_patch_files
    ]
    test_patch_files = [
        project_dir / "datasets" / "raw" / file for file in test_patch_files
    ]

    # Setup recording of stats per batch and epoch
    logger = Logger(
        list(["train", "train_inf", "val"]), ["loss", "accuracy"], file=True
    )

    datas = []
    for i, wsi_id in enumerate(wsi_ids):
        # Make graph data object
        data, groundtruth = get_data(
            project_dir,
            organ,
            wsi_id,
            x_min,
            y_min,
            width,
            height,
            groundtruth_tsvs,
            feature,
            graph_method,
            k,
            verbose,
        )

        # Split nodes into unlabelled, training and validation sets
        if wsi_id == 1:
            data = setup_node_splits(
                data,
                groundtruth,
                True,
                True,
                val_patch_files,
                test_patch_files,
            )
        else:
            if len(val_patch_files) == 0:
                data = setup_node_splits(data, groundtruth, True)
            else:
                data = setup_node_splits(data, groundtruth, True, False)
        datas.append(data)

    # Combine multiple graphs into a single graph
    data = Batch.from_data_list(datas)
    # Final data setup
    if model_type.value == "shadow":
        del data.batch  # bug in pyg when using shadow model and Batch
    elif model_type.value == "sign":
        data = SIGN(layers)(data)  # precompute SIGN fixed embeddings
    send_graph_to_device(data, device)

    # Setup training parameters, including dataloaders and models
    run_params = TrainParams(
        data,
        device,
        pretrained_path,
        model_type,
        batch_size,
        num_neighbours,
        epochs,
        layers,
        hidden_units,
        dropout,
        learning_rate,
        weighted_loss,
        use_custom_weights,
        organ,
    )
    train_runner = TrainRunner.new(run_params)

    # Saves each run by its timestamp and record params for the run
    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
    params = collect_params(
        seed,
        exp_name,
        wsi_ids,
        x_min,
        y_min,
        width,
        height,
        k,
        feature,
        graph_method,
        run_params,
    )
    params.to_csv(run_path / "params.csv", index=False)

    # Train!
    try:
        print("Training:")
        prev_best_val = 0
        for epoch in range(1, epochs + 1):
            loss, accuracy = train_runner.train()
            logger.log_loss("train", epoch - 1, loss)
            logger.log_accuracy("train", epoch - 1, accuracy)

            if epoch % validation_step == 0 or epoch == 1:
                train_accuracy, val_accuracy = train_runner.validate()
                logger.log_accuracy("train_inf", epoch - 1, train_accuracy)
                logger.log_accuracy("val", epoch - 1, val_accuracy)

                # Save new best model
                if val_accuracy >= prev_best_val:
                    train_runner.save_state(run_path, logger, epoch)
                    print("Saved best model")
                    prev_best_val = val_accuracy

    except KeyboardInterrupt:
        save_hp = input("Would you like to save anyway? y/n: ")
        if save_hp == "y":
            # Save the interrupted model
            train_runner.save_state(run_path, logger, epoch)

    # Save the fully trained model
    train_runner.save_state(run_path, logger, "final")
    print("Saved final model")


if __name__ == "__main__":
    typer.run(main)
