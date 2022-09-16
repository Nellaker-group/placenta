import torch
from torch_geometric.transforms import RandomNodeSplit
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    matthews_corrcoef,
)

from placenta.eval.eval import (
    plot_confusion_matrix,
    plot_tissue_pr_curves,
    get_tissue_confusion_matrix,
)
from placenta.models.graphsaint import GraphSAINT
from placenta.models.shadow import ShaDowGCN
from placenta.models.sign import SIGN as SIGN_MLP
from placenta.data.get_raw_data import get_nodes_within_tiles


def setup_node_splits(
    data,
    groundtruth,
    mask_unlabelled,
    include_validation=True,
    val_patch_files=[],
    test_patch_files=[],
    verbose=True,
):
    all_xs = data["pos"][:, 0]
    all_ys = data["pos"][:, 1]

    # Mark everything as training data first
    train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    data.train_mask = train_mask

    # Mask unlabelled data to ignore during training
    if mask_unlabelled and groundtruth is not None:
        unlabelled_inds = (groundtruth == 0).nonzero()[0]
        unlabelled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        unlabelled_mask[unlabelled_inds] = True
        data.unlabelled_mask = unlabelled_mask
        train_mask[unlabelled_inds] = False
        data.train_mask = train_mask
        if verbose:
            print(f"{len(unlabelled_inds)} nodes marked as unlabelled")

    # Split the graph by masks into training, validation and test nodes
    if include_validation:
        if len(val_patch_files) == 0:
            if len(test_patch_files) == 0:
                if verbose:
                    print("No validation patch provided, splitting nodes randomly")
                data = RandomNodeSplit(num_val=0.15, num_test=0.15)(data)
            else:
                if verbose:
                    print(
                        "No validation patch provided, splitting nodes randomly into "
                        "train and val and using test patch"
                    )
                data = RandomNodeSplit(num_val=0.15, num_test=0)(data)
                test_node_inds = []
                for file in test_patch_files:
                    patches_df = pd.read_csv(file)
                    for row in patches_df.itertuples(index=False):
                        test_node_inds.extend(
                            get_nodes_within_tiles(
                                (row.x, row.y), row.width, row.height, all_xs, all_ys
                            )
                        )
                test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                test_mask[test_node_inds] = True
                data.val_mask[test_node_inds] = False
                data.train_mask[test_node_inds] = False
                data.test_mask = test_mask
            if mask_unlabelled and groundtruth is not None:
                data.val_mask[unlabelled_inds] = False
                data.train_mask[unlabelled_inds] = False
                data.test_mask[unlabelled_inds] = False
        else:
            if verbose:
                print("Splitting graph by validation patch")
            val_node_inds = []
            for file in val_patch_files:
                patches_df = pd.read_csv(file)
                for row in patches_df.itertuples(index=False):
                    if (
                        row.x == 0
                        and row.y == 0
                        and row.width == -1
                        and row.height == -1
                    ):
                        data.val_mask = torch.ones(data.num_nodes, dtype=torch.bool)
                        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                        if mask_unlabelled and groundtruth is not None:
                            data.val_mask[unlabelled_inds] = False
                            data.train_mask[unlabelled_inds] = False
                            data.test_mask[unlabelled_inds] = False
                        if verbose:
                            print(
                                f"All nodes marked as validation: "
                                f"{data.val_mask.sum().item()}"
                            )
                        return data
                    val_node_inds.extend(
                        get_nodes_within_tiles(
                            (row.x, row.y), row.width, row.height, all_xs, all_ys
                        )
                    )
            val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            val_mask[val_node_inds] = True
            train_mask[val_node_inds] = False
            if len(test_patch_files) > 0:
                test_node_inds = []
                for file in test_patch_files:
                    patches_df = pd.read_csv(file)
                    for row in patches_df.itertuples(index=False):
                        test_node_inds.extend(
                            get_nodes_within_tiles(
                                (row.x, row.y), row.width, row.height, all_xs, all_ys
                            )
                        )
                test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                test_mask[test_node_inds] = True
                train_mask[test_node_inds] = False
                data.test_mask = test_mask
            else:
                data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = val_mask
            data.train_mask = train_mask
        if verbose:
            print(
                f"Graph split into {data.train_mask.sum().item()} train nodes "
                f"and {data.val_mask.sum().item()} validation nodes "
                f"and {data.test_mask.sum().item()} test nodes"
            )
    else:
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    return data


@torch.no_grad()
def inference(model, x, eval_loader, device):
    print("Running inference")
    model.eval()
    if not isinstance(model, ShaDowGCN):
        if isinstance(model, GraphSAINT):
            model.set_aggr("mean")
        out, graph_embeddings = model.inference(x, eval_loader, device)
    else:
        out = []
        graph_embeddings = []
        for batch in eval_loader:
            batch = batch.to(device)
            batch_out, batch_embed = model.inference(
                batch.x, batch.edge_index, batch.batch, batch.root_n_id
            )
            out.append(batch_out)
            graph_embeddings.append(batch_embed)
        out = torch.cat(out, dim=0)
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
    predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()
    predicted_labels = predicted_labels.cpu().numpy()
    out = out.cpu().detach().numpy()
    return out, graph_embeddings, predicted_labels


@torch.no_grad()
def inference_mlp(model, data, eval_loader, device):
    print("Running inference")
    model.eval()
    out = []
    graph_embeddings = []
    for idx in eval_loader:
        eval_x = data.x[idx].to(device)
        if isinstance(model, SIGN_MLP):
            eval_x = [eval_x]
            eval_x += [
                data[f"x{i}"][idx].to(device) for i in range(1, model.num_layers + 1)
            ]
        out_i, emb_i = model.inference(eval_x)
        out.append(out_i)
        graph_embeddings.append(emb_i)
    out = torch.cat(out, dim=0)
    graph_embeddings = torch.cat(graph_embeddings, dim=0)
    predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()
    predicted_labels = predicted_labels.cpu().numpy()
    out = out.cpu().detach().numpy()
    return out, graph_embeddings, predicted_labels


def evaluate(groundtruth, predicted_labels, out, organ, run_path, remove_unlabelled):
    tissue_ids = [tissue.id for tissue in organ.tissues]
    if remove_unlabelled:
        tissue_ids = tissue_ids[1:]

    accuracy = accuracy_score(groundtruth, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(groundtruth, predicted_labels)
    top_2_accuracy = top_k_accuracy_score(groundtruth, out, k=2, labels=tissue_ids)
    f1_macro = f1_score(groundtruth, predicted_labels, average="macro")
    cohen_kappa = cohen_kappa_score(groundtruth, predicted_labels)
    mcc = matthews_corrcoef(groundtruth, predicted_labels)
    roc_auc = roc_auc_score(
        groundtruth,
        softmax(out, axis=-1),
        average="macro",
        multi_class="ovo",
        labels=tissue_ids,
    )
    weighted_roc_auc = roc_auc_score(
        groundtruth,
        softmax(out, axis=-1),
        average="weighted",
        multi_class="ovo",
        labels=tissue_ids,
    )
    print("-----------------------")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Top 2 accuracy: {top_2_accuracy:.6f}")
    print(f"Balanced accuracy: {balanced_accuracy:.6f}")
    print(f"F1 macro score: {f1_macro:.6f}")
    print(f"Cohen's Kappa score: {cohen_kappa:.6f}")
    print(f"MCC score: {mcc:.6f}")
    print(f"ROC AUC macro: {roc_auc:.6f}")
    print(f"Weighted ROC AUC macro: {weighted_roc_auc:.6f}")
    print("-----------------------")

    print("Plotting confusion matrices")
    cm_df, cm_df_props = get_tissue_confusion_matrix(
        organ, predicted_labels, groundtruth, proportion_label=True
    )
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df, "All Tissues", run_path, "d")
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df_props, "All Tissues Proportion", run_path, ".2f")

    print("Plotting pr curves")
    tissue_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    tissue_colours = {tissue.id: tissue.colour for tissue in organ.tissues}
    plot_tissue_pr_curves(
        tissue_mapping,
        tissue_colours,
        groundtruth,
        predicted_labels,
        out,
        run_path / "pr_curves.png",
    )


def collect_params(
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
):
    return pd.DataFrame(
        {
            "seed": seed,
            "exp_name": exp_name,
            "wsi_ids": [np.array(wsi_ids)],
            "x_min": x_min,
            "y_min": y_min,
            "width": width,
            "height": height,
            "k": k,
            "feature": feature,
            "graph_method": graph_method,
            "batch_size": run_params.batch_size,
            "num_neighbours": run_params.num_neighbours,
            "learning_rate": run_params.learning_rate,
            "weighted_loss": run_params.weighted_loss,
            "custom_weights": run_params.custom_weights,
            "epochs": run_params.epochs,
            "layers": run_params.layers,
            "hidden_units": run_params.hidden_units,
            "dropout": run_params.dropout,
        },
        index=[0],
    )


def save_state(run_path, logger, model, epoch):
    torch.save(model, run_path / f"{epoch}_graph_model.pt")
    logger.to_csv(run_path / "graph_train_stats.csv")
