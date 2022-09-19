from torch_geometric.data import Data
from torch_geometric.transforms import Distance, KNNGraph
import matplotlib.tri as tri
import numpy as np
import torch


def setup_graph(
    coords, k, feature, graph_method, norm_edges=True, loop=True, verbose=True
):
    data = Data(x=torch.Tensor(feature), pos=torch.Tensor(coords.astype("int32")))
    if graph_method == "k":
        graph = make_k_graph(data, k, norm_edges, loop, verbose=verbose)
    elif graph_method == "delaunay":
        graph = make_delaunay_graph(data, norm_edges, verbose=verbose)
    elif graph_method == "intersection":
        graph = make_intersection_graph(data, k, norm_edges, verbose=verbose)
    else:
        raise ValueError(f"No such graph method: {graph_method}")
    if graph.x.ndim == 1:
        graph.x = graph.x.view(-1, 1)
    return graph


def make_k_graph(data, k, norm_edges=True, loop=True, verbose=True):
    if verbose:
        print(f"Generating graph for k={k}")
    data = KNNGraph(k=k + 1, loop=loop, force_undirected=True)(data)
    get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
    data = get_edge_distance_weights(data)
    if verbose:
        print(f"Graph made with {len(data.edge_index[0])} edges!")
    return data


def make_delaunay_triangulation(data, verbose=True):
    if verbose:
        print(f"Generating delaunay triangulation")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    if verbose:
        print("Triangulation made!")
    return triang


def make_delaunay_graph(data, norm_edges=True, verbose=True):
    if verbose:
        print(f"Generating delaunay graph")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    data.edge_index = torch.tensor(triang.edges.astype("int64"), dtype=torch.long).T
    get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
    data = get_edge_distance_weights(data)
    if verbose:
        print(f"Graph made with {len(data.edge_index[0])} edges!")
    return data


def make_intersection_graph(data, k, norm_edges=True, verbose=True):
    if verbose:
        print(f"Generating graph for k={k}")
    knn_graph = KNNGraph(k=k + 1, loop=False, force_undirected=True)(data)
    knn_edge_index = knn_graph.edge_index.T
    knn_edge_index = np.array(knn_edge_index.tolist())
    if verbose:
        print(f"Generating delaunay graph")
    try:
        triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
        delaunay_edge_index = triang.edges.astype("int64")
    except ValueError:
        print("Too few points to make a triangulation, returning knn graph")
        get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
        knn_graph = get_edge_distance_weights(knn_graph)
        if verbose:
            print(f"Graph made with {len(knn_graph.edge_index[0])} edges!")
        return knn_graph

    if verbose:
        print(f"Generating intersection of both graphs")
    _, ncols = knn_edge_index.shape
    dtype = ", ".join([str(knn_edge_index.dtype)] * ncols)
    intersection = np.intersect1d(
        knn_edge_index.view(dtype), delaunay_edge_index.view(dtype)
    )
    intersection = intersection.view(knn_edge_index.dtype).reshape(-1, ncols)
    intersection = torch.tensor(intersection, dtype=torch.long).T
    data.edge_index = intersection

    get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
    data = get_edge_distance_weights(data)
    if verbose:
        print(f"Graph made with {len(data.edge_index[0])} edges!")
    return data
