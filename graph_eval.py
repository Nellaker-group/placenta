from pathlib import Path
from typing import Optional, List

import typer
import torch
from torch_geometric.transforms import ToUndirected, SIGN
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import (
    NeighborSampler,
    NeighborLoader,
    ShaDowKHopSampler,
    DataLoader,
)
import numpy as np
import pandas as pd

from placenta.utils.utils import get_device
from placenta.organs.organs import Placenta
from placenta.data.get_raw_data import get_groundtruth_patch, get_raw_data
from placenta.graphs.create_graph import setup_graph
from placenta.utils.utils import get_feature, set_seed
from placenta.graphs.enums import FeatureArg, MethodArg
from placenta.analysis.vis_graph_patch import visualize_points
from placenta.graphs.graph_supervised import (
    setup_node_splits,
    inference,
    inference_mlp,
    evaluate,
)

def main(
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    wsi_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    val_patch_files: Optional[List[str]] = None,
    k: int = 5,
    feature: FeatureArg = FeatureArg.embeddings,
    top_conf: bool = False,
    model_type: str = typer.Option(...),
    graph_method: MethodArg = MethodArg.k,
    remove_unlabelled: bool = True,
    groundtruth_tsv: Optional[str] = None,
    verbose: bool = True,
):
    organ = Placenta
    device = get_device()
    set_seed(0)
    project_dir = Path.cwd()
    patch_files = [project_dir / "config" / file for file in val_patch_files]

    print("Begin graph construction...")
    predictions, embeddings, coords, confidence = get_raw_data(
        wsi_id, x_min, y_min, width, height, verbose=verbose
    )
    # Get ground truth manually annotated data
    _, _, groundtruth = get_groundtruth_patch(
        organ,
        project_dir,
        x_min,
        y_min,
        width,
        height,
        groundtruth_tsv,
    )
    # Covert input cell data into a graph
    feature_data = get_feature(feature, predictions, embeddings, organ)
    data = setup_graph(
        coords, k, feature_data, graph_method, loop=False, verbose=verbose
    )
    data = ToUndirected()(data)
    data.edge_index, data.edge_attr = add_self_loops(
        data["edge_index"], data["edge_attr"], fill_value="mean"
    )
    pos = data.pos
    x = data.x.to(device)

    data = setup_node_splits(
        data, groundtruth, remove_unlabelled, True, patch_files, verbose=verbose
    )
    print("Graph construction complete")

    # Setup trained model
    pretrained_path = (
        project_dir
        / "results"
        / "graph"
        / model_type
        / exp_name
        / model_weights_dir
        / model_name
    )
    model = torch.load(pretrained_path, map_location=device)
    model_epochs = (
        "model_final"
        if model_name == "graph_model.pt"
        else f"model_{model_name.split('_')[0]}"
    )

    if model_type == "sup_sign":
        data = SIGN(model.num_layers)(data)  # precompute SIGN fixed embeddings

    # Setup paths
    save_path = (
        Path(*pretrained_path.parts[:-1]) / "eval" / model_epochs / f"run_{wsi_id}"
    )
    save_path.mkdir(parents=True, exist_ok=True)
    conf_str = "_top_conf" if top_conf else ""
    plot_name = f"{val_patch_files[0].split('.csv')[0]}_{conf_str}"

    # Dataloader for eval, feeds in whole graph
    if model_type == "sup_graphsage":
        eval_loader = NeighborLoader(
            data,
            num_neighbors=[-1],
            batch_size=512,
            shuffle=False,
        )
        eval_loader.data.num_nodes = data.num_nodes
        eval_loader.data.n_id = torch.arange(data.num_nodes)
    elif model_type == "sup_shadow":
        eval_loader = ShaDowKHopSampler(
            data,
            depth=6,
            num_neighbors=5,
            node_idx=None,
            batch_size=4000,
            shuffle=False,
        )
    elif model_type == "sup_sign" or model_type == "sup_mlp":
        eval_loader = DataLoader(range(data.num_nodes), batch_size=512, shuffle=False)
    else:
        eval_loader = NeighborSampler(
            data.edge_index,
            node_idx=None,
            sizes=[-1],
            batch_size=512,
            shuffle=False,
        )

    # Run inference and get predicted labels for nodes
    if model_type == "sup_mlp" or model_type == "sup_sign":
        out, graph_embeddings, predicted_labels = inference_mlp(
            model, data, eval_loader, device
        )
    else:
        out, graph_embeddings, predicted_labels = inference(
            model, x, eval_loader, device
        )

    # restrict to only data in patch_files using val_mask
    val_nodes = data.val_mask
    predicted_labels = predicted_labels[val_nodes]
    out = out[val_nodes]
    pos = pos[val_nodes]
    groundtruth = (
        groundtruth[val_nodes] if groundtruth_tsv is not None else groundtruth
    )

    # Remove unlabelled (class 0) ground truth points
    if remove_unlabelled and groundtruth_tsv is not None:
        unlabelled_inds, groundtruth, predicted_labels, pos, out = _remove_unlabelled(
            groundtruth, predicted_labels, pos, out
        )

    # Print some prediction count info
    tissue_label_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    _print_prediction_stats(predicted_labels, tissue_label_mapping)

    # Evaluate against ground truth tissue annotations
    if groundtruth_tsv is not None:
        evaluate(
            groundtruth,
            predicted_labels,
            out,
            organ,
            save_path,
            remove_unlabelled,
        )

    # Visualise cluster labels on graph patch
    print("Generating image")
    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in predicted_labels]
    visualize_points(
        organ,
        save_path / f"{plot_name.split('.png')[0]}.png",
        pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
    )

    # make tsv if the whole graph was used
    if len(data.pos) == len(data.pos[data.val_mask]):
        label_dict = {tissue.id: tissue.label for tissue in organ.tissues}
        predicted_labels = [label_dict[label] for label in predicted_labels]
        _save_tissue_preds_as_tsv(predicted_labels, pos, save_path)


def _remove_unlabelled(groundtruth, predicted_labels, pos, out):
    labelled_inds = groundtruth.nonzero()[0]
    groundtruth = groundtruth[labelled_inds]
    pos = pos[labelled_inds]
    out = out[labelled_inds]
    out = np.delete(out, 0, axis=1)
    predicted_labels = predicted_labels[labelled_inds]
    return labelled_inds, groundtruth, predicted_labels, pos, out


def _print_prediction_stats(predicted_labels, tissue_label_mapping):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    unique_labels = []
    for label in unique:
        unique_labels.append(tissue_label_mapping[label])
    unique_counts = dict(zip(unique_labels, counts))
    print(f"Predictions per label: {unique_counts}")


def _save_tissue_preds_as_tsv(predicted_labels, coords, save_path):
    tissue_preds_df = pd.DataFrame(
        {
            "x": coords[:, 0].numpy().astype(int),
            "y": coords[:, 1].numpy().astype(int),
            "class": predicted_labels,
        }
    )
    tissue_preds_df.to_csv(save_path / "tissue_preds.tsv", sep="\t", index=False)


if __name__ == "__main__":
    typer.run(main)
