import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from placenta.hdf5.utils import get_data_in_patch


def get_groundtruth_patch(
    organ, project_dir, x_min, y_min, width, height, groundtruth_tsv
):
    if not groundtruth_tsv:
        print("No tissue label tsv supplied")
        return None, None, None
    groundtruth_path = project_dir / "datasets" / "raw" / groundtruth_tsv
    if not os.path.exists(str(groundtruth_path)):
        print("No tissue label tsv found")
        return None, None, None

    groundtruth_df = pd.read_csv(groundtruth_path, sep="\t")
    xs = groundtruth_df["px"].to_numpy()
    ys = groundtruth_df["py"].to_numpy()
    groundtruth = groundtruth_df["class"].to_numpy()

    if x_min == 0 and y_min == 0 and width == -1 and height == -1:
        sort_args = np.lexsort((ys, xs))
        tissue_ids = np.array(
            [organ.tissue_by_label(tissue_name).id for tissue_name in groundtruth]
        )
        return xs[sort_args], ys[sort_args], tissue_ids[sort_args]

    mask = np.logical_and(
        (np.logical_and(xs > x_min, (ys > y_min))),
        (np.logical_and(xs < (x_min + width), (ys < (y_min + height)))),
    )
    patch_xs = xs[mask]
    patch_ys = ys[mask]
    patch_groundtruth = groundtruth[mask]

    patch_tissue_ids = np.array(
        [organ.tissue_by_label(tissue_name).id for tissue_name in patch_groundtruth]
    )
    sort_args = np.lexsort((patch_ys, patch_xs))

    return patch_xs[sort_args], patch_ys[sort_args], patch_tissue_ids[sort_args]


def get_raw_data(wsi_id, x_min, y_min, width, height, verbose=True):
    raw_data_path = (
        Path(__file__).absolute().parent.parent.parent
        / "datasets"
        / "raw"
        / f"wsi_{wsi_id}.hdf5"
    )
    if verbose:
        print(f"Getting data from: {raw_data_path}")
        print(f"Using patch of size: x{x_min}, y{y_min}, w{width}, h{height}")
    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_data_in_patch(
        raw_data_path, x_min, y_min, width, height, verbose=verbose
    )
    if verbose:
        print(f"Data loaded with {len(predictions)} nodes")
    sort_args = np.lexsort((coords[:, 1], coords[:, 0]))
    coords = coords[sort_args]
    predictions = predictions[sort_args]
    embeddings = embeddings[sort_args]
    confidence = confidence[sort_args]
    if verbose:
        print("Data sorted by x coordinates")

    return predictions, embeddings, coords, confidence


def get_nodes_within_tiles(tile_coords, tile_width, tile_height, all_xs, all_ys):
    tile_min_x, tile_min_y = tile_coords[0], tile_coords[1]
    tile_max_x, tile_max_y = tile_min_x + tile_width, tile_min_y + tile_height
    if isinstance(all_xs, torch.Tensor) and isinstance(all_ys, torch.Tensor):
        mask = torch.logical_and(
            (torch.logical_and(all_xs > tile_min_x, (all_ys > tile_min_y))),
            (torch.logical_and(all_xs < tile_max_x, (all_ys < tile_max_y))),
        )
        return mask.nonzero()[:, 0].tolist()
    else:
        mask = np.logical_and(
            (np.logical_and(all_xs > tile_min_x, (all_ys > tile_min_y))),
            (np.logical_and(all_xs < tile_max_x, (all_ys < tile_max_y))),
        )
        return mask.nonzero()[0].tolist()