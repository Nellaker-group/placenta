from typing import Optional

import typer

from placenta.logger.logger import Logger
from utils import setup_run, get_device
from enums import ModelsArg
from placenta.runners.train_runner import TrainParams, TrainRunner
from utils import set_seed, get_project_dir
from organs import Placenta as organ
from dataset import Placenta


def main(
    seed: int = 0,
    exp_name: str = typer.Option(...),
    pretrained: Optional[str] = None,
    model_type: ModelsArg = typer.Option(...),
    batch_size: int = typer.Option(...),
    num_neighbours: int = typer.Option(...),
    epochs: int = 1000,
    layers: int = 16,
    hidden_units: int = 256,
    dropout: float = 0.5,
    learning_rate: float = 0.001,
    weighted_loss: bool = True,
    use_custom_weights: bool = True,
    validation_step: int = 100,
):
    """
    Train a model on the placenta dataset using the default cell graph construction.

    Args:
        seed: set the random seed
        exp_name: name of the experiment (used for saving the model)
        pretrained: path to a pretrained model
        model_type: type of model to train, one of ModelsArg
        batch_size: batch size
        num_neighbours: neighbours per hop or cluster size or sample coverage
        epochs: max number of epochs to train for
        layers: number of model layers (and usually hops)
        hidden_units: number of hidden units per layer
        dropout: dropout rate on each layer and attention heads (if applicable)
        learning_rate: learning rate
        weighted_loss: whether to used weighted cross entropy
        use_custom_weights: if weighted loss, whether to use custom weights
        validation_step: number of epochs between validation check
    """
    device = get_device()
    set_seed(seed)
    project_dir = get_project_dir()
    pretrained_path = project_dir / pretrained if pretrained else None

    # Setup recording of stats per batch and epoch
    logger = Logger(
        list(["train", "train_inf", "val"]), ["loss", "accuracy"], file=True
    )

    # Download, process, and load graph
    dataset = Placenta(project_dir / "datasets")
    data = dataset[0]

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
        12,
        weighted_loss,
        use_custom_weights,
        validation_step,
        organ,
    )
    train_runner = TrainRunner.new(run_params)

    # Saves each run by its timestamp and record params for the run
    run_path = setup_run(project_dir, f"{model_type}/{exp_name}")
    dataset.save_params(run_path)
    train_runner.params.save(seed, exp_name, run_path)

    # train!
    train(train_runner, logger, run_path)


def train(train_runner, logger, run_path):
    train_runner.prepare_data()
    epochs = train_runner.params.epochs
    validation_step = train_runner.params.validation_step

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
