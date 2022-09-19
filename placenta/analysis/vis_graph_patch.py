from enum import Enum

import matplotlib.pyplot as plt
import torch
import typer
from matplotlib.collections import LineCollection
from torch_geometric.data import Data

from placenta.organs.organs import Placenta
from placenta.utils.utils import get_project_dir
from placenta.hdf5.utils import get_data_in_patch
from placenta.graphs.create_graph import (
    make_k_graph,
    make_delaunay_triangulation,
    make_intersection_graph,
)


class MethodArg(str, Enum):
    k = "k"
    delaunay = "delaunay"
    intersection = "intersection"


def main(
    wsi_id: int = typer.Option(...),
    method: MethodArg = MethodArg.intersection,
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    plot_edges: bool = False,
):
    """Generates a graph and saves its visualisation. Node are coloured by cell type

    Args:
        wsi_id: id of the wsi
        method: graph creation method to use.
        x_min: min x coordinate for defining a subsection/patch of the WSI
        y_min: min y coordinate for defining a subsection/patch of the WSI
        width: width for defining a subsection/patch of the WSI. -1 for all
        height: height for defining a subsection/patch of the WSI. -1 for all
        plot_edges: whether to plot edges or just points
    """
    organ = Placenta
    project_dir = get_project_dir()
    raw_data_path = project_dir / "datasets" / "raw_data" / f"{wsi_id}.hdf5"
    print(f"Getting data from: {raw_data_path}")

    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_data_in_patch(
        raw_data_path, x_min, y_min, width, height
    )
    print(f"Data loaded with {len(predictions)} nodes")

    # Make graph data object
    data = Data(x=predictions, pos=torch.Tensor(coords.astype("int32")))

    save_dir = (
        project_dir
        / "visualisations"
        / "cell_graphs"
        / f"wsi_{wsi_id}"
    )
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}"

    method = method.value
    if method == "k":
        vis_for_k(6, data, plot_name, save_dir, organ, width, height, plot_edges)
    elif method == "delaunay":
        vis_delaunay(data, plot_name, save_dir, organ, width, height)
    elif method == "intersection":
        vis_intersection(data, 6, plot_name, save_dir, organ, width, height)
    else:
        raise ValueError(f"no such method: {method}")


def vis_for_k(k, data, plot_name, save_dir, organ, width, height, plot_edges=True):
    # Specify save graph vis location
    save_path = save_dir / "k"
    save_path.mkdir(parents=True, exist_ok=True)

    data = make_k_graph(data, k)
    if not plot_edges:
        edge_index = None
        edge_weight = None
    else:
        edge_index = data.edge_index
        edge_weight = data.edge_attr

    plot_name = f"k{k}_{plot_name}.png"
    print(f"Plotting...")
    visualize_points(
        organ,
        save_path / plot_name,
        data.pos,
        labels=data.x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        width=width,
        height=height,
    )
    print(f"Plot saved to {save_path / plot_name}")


def vis_delaunay(data, plot_name, save_dir, organ, width, height):
    colours_dict = {cell.id: cell.colour for cell in organ.cells}
    colours = [colours_dict[label] for label in data.x]

    # Specify save graph vis location
    save_path = save_dir / "delaunay"
    save_path.mkdir(parents=True, exist_ok=True)

    delaunay = make_delaunay_triangulation(data)
    print(f"Plotting...")

    point_size = 1 if len(delaunay.edges) >= 10000 else 2

    figsize = _calc_figsize(data.pos, width, height)
    fig = plt.figure(figsize=figsize, dpi=150)
    plt.triplot(delaunay, linewidth=0.5, color="black")
    plt.scatter(
        data.pos[:, 0], data.pos[:, 1], marker=".", s=point_size, zorder=1000, c=colours
    )
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()

    plot_name = f"{plot_name}.png"
    plt.savefig(save_path / plot_name)
    print(f"Plot saved to {save_path / plot_name}")


def vis_intersection(data, k, plot_name, save_dir, organ, width, height):
    # Specify save graph vis location
    save_path = save_dir / "intersection"
    save_path.mkdir(parents=True, exist_ok=True)

    intersection_graph = make_intersection_graph(data, k)
    edge_index = intersection_graph.edge_index
    edge_weight = intersection_graph.edge_attr

    plot_name = f"k{k}_{plot_name}.png"
    print(f"Plotting...")
    visualize_points(
        organ,
        save_path / plot_name,
        data.pos,
        labels=data.x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        width=width,
        height=height,
    )
    print(f"Plot saved to {save_path / plot_name}")


def visualize_points(
    organ,
    save_path,
    pos,
    width=None,
    height=None,
    labels=None,
    edge_index=None,
    edge_weight=None,
    colours=None,
    point_size=None,
):
    if colours is None:
        colours_dict = {cell.id: cell.colour for cell in organ.cells}
        colours = [colours_dict[label] for label in labels]

    if point_size is None:
        point_size = 1 if len(pos) >= 10000 else 2

    figsize = _calc_figsize(pos, width, height)
    fig = plt.figure(figsize=figsize, dpi=150)

    if edge_index is not None:
        line_collection = []
        for i, (src, dst) in enumerate(edge_index.t().tolist()):
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            line_collection.append((src, dst))
        line_colour = (
            [str(weight) for weight in edge_weight.t()[0].tolist()]
            if edge_weight is not None
            else "grey"
        )
        lc = LineCollection(line_collection, linewidths=0.5, colors=line_colour)
        ax = plt.gca()
        ax.add_collection(lc)
        ax.autoscale()
    plt.scatter(
        pos[:, 0],
        pos[:, 1],
        marker=".",
        s=point_size,
        zorder=1000,
        c=colours,
        cmap="Spectral",
    )
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)


def _calc_figsize(pos, width, height):
    if width is None and height is None:
        return 8, 8
    if width == -1 and height == -1:
        pos_width = max(pos[:, 0]) - min(pos[:, 0])
        pos_height = max(pos[:, 1]) - min(pos[:, 1])
        ratio = pos_width / pos_height
        length = ratio * 8
        return length, 8
    else:
        ratio = width / height
        length = ratio * 8
        return length, 8


if __name__ == "__main__":
    typer.run(main)
