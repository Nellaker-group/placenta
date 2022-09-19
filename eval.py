from pathlib import Path

import typer
import numpy as np

from placenta.utils.utils import get_device
from placenta.organs.organs import Placenta as organ
from placenta.utils.utils import set_seed, get_project_dir
from placenta.analysis.vis_graph_patch import visualize_points
from placenta.data.dataset import Placenta
from placenta.eval.eval import evaluate
from placenta.graphs.enums import ModelsArg
from placenta.runners.eval_runner import EvalParams, EvalRunner


def main(
    exp_name: str = typer.Option(...),
    run_time_stamp: str = typer.Option(...),
    model_name: str = typer.Option(...),
    model_type: ModelsArg = typer.Option(...),
    use_test_set: bool = False,
):
    device = get_device()
    set_seed(0)
    project_dir = get_project_dir()

    # Load graph of WSI 1
    dataset = Placenta(project_dir / "datasets")
    data = dataset[0].get_example(0)

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

    # restrict to only data in patch_files using val_mask
    mask = data.val_mask if not use_test_set else data.test_mask
    predicted_labels = predicted_labels[mask]
    out = out[mask]
    pos = data.pos[mask]
    groundtruth = data.y[mask]

    # Remove unlabelled (class 0) ground truth points
    groundtruth, predicted_labels, pos, out = _remove_unlabelled(
        groundtruth, predicted_labels, pos, out
    )

    # Setup paths
    model_epochs = (
        "model_final"
        if model_name == "graph_model.pt"
        else f"model_{model_name.split('_')[0]}"
    )
    save_path = Path(*pretrained_path.parts[:-1]) / "eval" / model_epochs
    save_path.mkdir(parents=True, exist_ok=True)
    plot_name = "test_patch.png" if use_test_set else "val_patch.png"

    # Evaluate against ground truth tissue annotations
    evaluate(groundtruth, predicted_labels, out, organ, save_path)

    # Visualise predictions on graph patch
    print("Generating image")
    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in predicted_labels]
    visualize_points(
        organ,
        save_path / plot_name,
        pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
    )


def _remove_unlabelled(groundtruth, predicted_labels, pos, out):
    labelled_inds = groundtruth.nonzero()[:, 0]
    groundtruth = groundtruth[labelled_inds]
    pos = pos[labelled_inds]
    out = out[labelled_inds]
    out = np.delete(out, 0, axis=1)
    predicted_labels = predicted_labels[labelled_inds]
    return groundtruth, predicted_labels, pos, out


if __name__ == "__main__":
    typer.run(main)
