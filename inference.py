from pathlib import Path

import typer
import numpy as np
import pandas as pd

from placenta.utils import get_device, set_seed, get_project_dir
from placenta.organs import Placenta as organ
from placenta.enums import ModelsArg
from placenta.analysis.vis_groundtruth import visualize_points
from placenta.dataset import Placenta, get_nodes_within_tiles
from placenta.runners.eval_runner import EvalParams, EvalRunner


def main(
    exp_name: str = typer.Option(...),
    run_time_stamp: str = typer.Option(...),
    model_name: str = typer.Option(...),
    model_type: ModelsArg = typer.Option(...),
    wsi_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    remove_unlabelled: bool = False,
):
    """
    Runs inference of a model for a region of a wsi. Will save the predictions
    as a tsv file and will plot the predictions.

    Args:
        exp_name: Name of experiment for which model was trained in.
        run_time_stamp: Time stamp of the run for which model was trained in.
        model_name: Name of model weights file to load and evaluate.
        model_type: Type of model to evaluate. One of ModelsArg
        wsi_id: ID of the wsi to run inference on.
        x_min: x coordinate of the top left corner of the region to run inference on.
        y_min: y coordinate of the top left corner of the region to run inference on.
        width: width of the region to run inference on. -1 means max width.
        height: height of the region to run inference on. -1 means max height.
        remove_unlabelled: Whether to remove unlabelled pixels from the predictions.
    """
    device = get_device()
    set_seed(0)
    project_dir = get_project_dir()

    # Download, process, and load graph
    dataset = Placenta(project_dir / "datasets")
    data = dataset[0].get_example(wsi_id - 1)

    # filter the graph using the patch coordinates
    if x_min != 0 or y_min != 0 or width != -1 or height != -1:
        mask = get_nodes_within_tiles(
            (x_min, y_min), width, height, data.pos[:, 0], data.pos[:, 1]
        )
        data = data.subgraph(mask)

    pretrained_path = (
        project_dir
        / "results"
        / model_type.value
        / exp_name
        / run_time_stamp
        / model_name
    )
    eval_params = EvalParams(data, device, pretrained_path, model_type, 512, organ)
    eval_runner = EvalRunner.new(eval_params)

    # Run inference and get predicted labels for nodes
    out, embeddings, predicted_labels = eval_runner.inference()

    pos = data.pos
    # Remove unlabelled (class 0) ground truth points
    if remove_unlabelled:
        predicted_labels, pos, out = _remove_unlabelled(
            data.y, predicted_labels, pos, out
        )

    # Setup paths
    model_epochs = (
        "model_final"
        if model_name == "graph_model.pt"
        else f"model_{model_name.split('_')[0]}"
    )
    save_path = Path(*pretrained_path.parts[:-1]) / "eval" / model_epochs
    save_path.mkdir(parents=True, exist_ok=True)

    # Visualise predictions on graph patch
    print("Generating image")
    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in predicted_labels]
    visualize_points(
        organ,
        save_path / f"x{x_min}_y{y_min}_w{width}_h{height}.png",
        pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
    )

    # make tsv of predictions and coordinates
    label_dict = {tissue.id: tissue.label for tissue in organ.tissues}
    predicted_labels = [label_dict[label] for label in predicted_labels]
    _save_tissue_preds_as_tsv(
        predicted_labels, pos, save_path / f"x{x_min}_y{y_min}_w{width}_h{height}.csv"
    )


def _remove_unlabelled(groundtruth, predicted_labels, pos, out):
    labelled_inds = groundtruth.nonzero()[:, 0]
    pos = pos[labelled_inds]
    out = out[labelled_inds]
    out = np.delete(out, 0, axis=1)
    predicted_labels = predicted_labels[labelled_inds]
    return predicted_labels, pos, out


def _save_tissue_preds_as_tsv(predicted_labels, coords, save_path):
    tissue_preds_df = pd.DataFrame(
        {
            "x": coords[:, 0].numpy().astype(int),
            "y": coords[:, 1].numpy().astype(int),
            "class": predicted_labels,
        }
    )
    tissue_preds_df.to_csv(save_path, sep="\t", index=False)


if __name__ == "__main__":
    typer.run(main)
