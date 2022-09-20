import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    f1_score,
    roc_auc_score,
)


def plot_confusion_matrix(cm, dataset_name, run_path, fmt="d"):
    save_path = run_path / f"{dataset_name}_confusion_matrix.png"

    plt.rcParams["figure.dpi"] = 300
    sns.heatmap(cm, annot=True, cmap="Blues", square=True, cbar=False, fmt=fmt)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    plt.clf()

def plot_tissue_pr_curves(id_to_label, colours, ground_truth, preds, scores, save_path):
    unique_values_in_pred = set(preds)
    unique_values_in_truth = set(np.array(ground_truth))
    unique_values_in_both = list(unique_values_in_pred.union(unique_values_in_truth))

    ground_truth = label_binarize(ground_truth, classes=unique_values_in_both)
    scores = np.array(scores)
    scores = softmax(scores, axis=-1)

    ground_truth_label_map = {
        unique_values_in_both[i]: i for i in list(range(len(unique_values_in_both)))
    }

    # Compute Precision-Recall and plot curve
    precision, recall, average_precision = {}, {}, {}
    for i in list(unique_values_in_truth):
        precision[i], recall[i], _ = precision_recall_curve(
            ground_truth[:, ground_truth_label_map[i]], scores[:, i - 1]
        )
        average_precision[i] = average_precision_score(
            ground_truth[:, ground_truth_label_map[i]], scores[:, i - 1]
        )
    plt.clf()
    sns.set(style="white")
    plt.figure(figsize=(9, 6), dpi=300)
    ax = plt.subplot(111)
    for i in unique_values_in_truth:
        plt.plot(
            recall[i],
            precision[i],
            label=f"{id_to_label[i]} ({average_precision[i]:0.2f})",
            color=colours[i],
        )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.savefig(save_path)
    plt.clf()


def get_tissue_confusion_matrix(organ, pred, truth, proportion_label=False):
    tissue_ids = {tissue.id for tissue in organ.tissues}
    tissue_labels = [tissue.label for tissue in organ.tissues]

    unique_values_in_pred = set(pred)
    unique_values_in_truth = set(np.array(truth))
    unique_values_in_matrix = unique_values_in_pred.union(unique_values_in_truth)
    missing_tissue_ids = list(tissue_ids - unique_values_in_matrix)
    missing_tissue_ids.sort()

    cm = confusion_matrix(truth, pred)

    if len(missing_tissue_ids) > 0:
        for missing_id in missing_tissue_ids:
            column_insert = np.zeros((cm.shape[0], 1))
            cm = np.hstack((cm[:, :missing_id], column_insert, cm[:, missing_id:]))
            row_insert = np.zeros((1, cm.shape[1]))
            cm = np.insert(cm, missing_id, row_insert, 0)

    row_labels = []
    if proportion_label:
        unique_counts = cm.sum(axis=1)
        total_counts = cm.sum()
        label_proportions = ((unique_counts / total_counts) * 100).round(2)
        for i, label in enumerate(tissue_labels):
            row_labels.append(f"{label} ({label_proportions[i]}%)")

    cm_df = pd.DataFrame(cm, columns=tissue_labels, index=tissue_labels).astype(int)
    unique_counts = cm.sum(axis=1)

    cm_df_props = (
        pd.DataFrame(
            cm / unique_counts[:, None], columns=tissue_labels, index=tissue_labels
        )
        .fillna(0)
        .astype(float)
    )

    non_empty_rows = (cm_df.T != 0).any()
    cm_df = cm_df[non_empty_rows]
    cm_df_props = cm_df_props[non_empty_rows]
    empty_row_names = non_empty_rows[non_empty_rows == False].index.tolist()
    cm_df = cm_df.drop(columns=empty_row_names)
    cm_df_props = cm_df_props.drop(columns=empty_row_names)

    if proportion_label:
        row_labels = np.array(row_labels)
        row_labels = row_labels[non_empty_rows]
        cm_df_props.set_index(row_labels, drop=True, inplace=True)

    return cm_df, cm_df_props


def evaluate(groundtruth, predicted_labels, out, organ, run_path):
    tissue_ids = [tissue.id for tissue in organ.tissues]
    tissue_ids = tissue_ids[1:]

    accuracy = accuracy_score(groundtruth, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(groundtruth, predicted_labels)
    top_2_accuracy = top_k_accuracy_score(groundtruth, out, k=2, labels=tissue_ids)
    f1_macro = f1_score(groundtruth, predicted_labels, average="macro")
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
