import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_self_loops, degree

from placenta.data.get_raw_data import get_groundtruth_patch, get_raw_data
from placenta.graphs.create_graph import setup_graph
from placenta.utils.utils import get_feature


def get_data(
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
):
    (
        predictions,
        embeddings,
        coords,
        confidence,
        groundtruth,
    ) = get_data_and_groundtruth(
        project_dir,
        organ,
        wsi_id,
        x_min,
        y_min,
        width,
        height,
        groundtruth_tsvs,
        verbose,
    )
    # Covert input cell data into a graph
    data = build_graph(
        feature,
        organ,
        predictions,
        embeddings,
        coords,
        groundtruth,
        graph_method,
        k,
    )
    return data, groundtruth


def get_data_and_groundtruth(
    project_dir, organ, wsi_id, x_min, y_min, width, height, groundtruth_tsvs, verbose
):
    # Get training data from hdf5 files
    predictions, embeddings, coords, confidence = get_raw_data(
        wsi_id, x_min, y_min, width, height, verbose
    )
    # Get ground truth manually annotated data
    _, _, groundtruth = get_groundtruth_patch(
        organ,
        project_dir,
        x_min,
        y_min,
        width,
        height,
        groundtruth_tsvs[wsi_id - 1],
    )
    return predictions, embeddings, coords, confidence, groundtruth


def build_graph(
    feature, organ, predictions, embeddings, coords, groundtruth, graph_method, k
):
    # Covert input cell data into a graph
    feature_data = get_feature(feature, predictions, embeddings, organ)
    data = setup_graph(coords, k, feature_data, graph_method, loop=False)
    data.y = torch.Tensor(groundtruth).type(torch.LongTensor)
    data = ToUndirected()(data)
    data.edge_index, data.edge_attr = add_self_loops(
        data["edge_index"], data["edge_attr"], fill_value="mean"
    )
    row, col = data.edge_index
    data.edge_weight = 1.0 / degree(col, data.num_nodes)[col]
    return data
