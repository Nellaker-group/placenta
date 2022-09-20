import typer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dataset import Placenta, get_nodes_within_tiles
from utils import get_project_dir
from organs import Placenta as organ


def main(
    wsi_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    remove_unlabelled: bool = False,
):
    """
    Plots the ground truth points for a given WSI and patch coordinates.

    Args:
        wsi_id: The WSI ID to plot the ground truth points for.
        x_min: The minimum x coordinate of the patch.
        y_min: The minimum y coordinate of the patch.
        width: The width of the patch. If -1, use max width.
        height: The height of the patch. If -1, use max height.
        remove_unlabelled: Whether to remove unlabelled points from the plot.
    """
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
    xs = np.array(data.pos[:, 0])
    ys = np.array(data.pos[:, 1])
    groundtruth = np.array(data.y)

    if remove_unlabelled:
        labelled_inds = groundtruth.nonzero()
        groundtruth = groundtruth[labelled_inds]
        xs = xs[labelled_inds]
        ys = ys[labelled_inds]

    save_dir = project_dir / "visualisations" / "groundtruth" / f"wsi_{wsi_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}"
    save_path = save_dir / plot_name

    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in groundtruth]
    visualize_points(
        save_path,
        np.stack((xs, ys), axis=1),
        colours=colours,
        width=width,
        height=height,
    )


def visualize_points(
    save_path,
    pos,
    width=None,
    height=None,
    edge_index=None,
    edge_weight=None,
    colours=None,
    point_size=None,
):
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
