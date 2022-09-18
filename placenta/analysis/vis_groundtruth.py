import typer
import numpy as np

from placenta.data.get_raw_data import get_groundtruth_patch
from placenta.analysis.vis_graph_patch import visualize_points
from placenta.utils.utils import get_project_dir
from placenta.organs.organs import Placenta


def main(
    wsi_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    groundtruth_tsv: str = typer.Option(...),
    remove_unlabelled: bool = False,
):
    project_dir = get_project_dir()
    organ = Placenta

    xs, ys, groundtruth = get_groundtruth_patch(
        project_dir, x_min, y_min, width, height, groundtruth_tsv
    )

    if remove_unlabelled:
        labelled_inds = groundtruth.nonzero()
        groundtruth = groundtruth[labelled_inds]
        xs = xs[labelled_inds]
        ys = ys[labelled_inds]

    unique, counts = np.unique(groundtruth, return_counts=True)
    print(dict(zip(unique, counts)))

    save_dir = project_dir / "visualisations" / "groundtruth" / f"wsi_{wsi_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}"
    save_path = save_dir / plot_name

    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in groundtruth]
    visualize_points(
        organ,
        save_path,
        np.stack((xs, ys), axis=1),
        colours=colours,
        width=width,
        height=height,
    )


if __name__ == "__main__":
    typer.run(main)
