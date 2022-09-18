from abc import ABC, abstractmethod
import os.path as osp
from pathlib import Path
import h5py

import torch
from torch_geometric.data import InMemoryDataset, download_url, Data, Batch
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.transforms import ToUndirected, KNNGraph
import matplotlib.tri as tri
import numpy as np
import pandas as pd

from organs import organs


class GraphConstructor(ABC):
    @abstractmethod
    def construct(self, data_file: str, gt_file: str) -> Data:
        pass

    def read_hdf5(self, path: Path):
        with h5py.File(path, "r") as f:
            # The [()] loads the data set into memory so that we can close the file
            # and still access it.
            cells = f["predictions"][()]
            embeddings = f["embeddings"][()]
            coords = f["coords"][()]
            confidence = f["confidence"][()]
        return cells, embeddings, coords, confidence


class DefaultGraphConstructor(GraphConstructor):
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir

    def construct(self, data_file, gt_file) -> Data:

        # Get cell data from hdf5 datasets
        cells, embeddings, coords, confidence = self.read_hdf5(self.raw_dir / data_file)
        cells, embeddings, coords, confidence = self._sort_cell_data(
            cells, embeddings, coords, confidence
        )
        # Get groundtruth data from tsv file
        groundtruth_df = pd.read_csv(self.raw_dir / gt_file, sep="\t")
        xs, ys, groundtruth = self._sort_groundtruth(groundtruth_df)
        # Create graph
        data = Data(
            x=torch.Tensor(embeddings), pos=torch.Tensor(coords.astype("int32"))
        )
        data = self._build_edges(data)
        data.y = torch.Tensor(groundtruth).type(torch.LongTensor)
        data = self._finalise_graph_properties(data)
        return data

    def node_splits(self, data, val_file, test_file):
        data = self._create_data_splits(data, val_file, test_file)
        return data

    def _sort_cell_data(self, cells, embeddings, coords, confidence):
        sort_args = np.lexsort((coords[:, 1], coords[:, 0]))
        coords = coords[sort_args]
        cells = cells[sort_args]
        embeddings = embeddings[sort_args]
        confidence = confidence[sort_args]
        return cells, embeddings, coords, confidence

    def _sort_groundtruth(self, groundtruth_df):
        xs = groundtruth_df["px"].to_numpy()
        ys = groundtruth_df["py"].to_numpy()
        groundtruth = groundtruth_df["class"].to_numpy()
        sort_args = np.lexsort((ys, xs))
        return xs[sort_args], ys[sort_args], groundtruth[sort_args]

    def _get_tissue_label_mapping(self):
        return {tissue.label: tissue.id for tissue in organs.Placenta.tissues}

    def _build_edges(self, data):
        # knn graph
        knn_graph = KNNGraph(k=5, loop=False, force_undirected=True)(data)
        knn_edge_index = knn_graph.edge_index.T
        knn_edge_index = np.array(knn_edge_index.tolist())
        # delaunay graph
        triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
        delaunay_edge_index = triang.edges.astype("int64")
        # intersection graph
        _, ncols = knn_edge_index.shape
        dtype = ", ".join([str(knn_edge_index.dtype)] * ncols)
        intersection = np.intersect1d(
            knn_edge_index.view(dtype), delaunay_edge_index.view(dtype)
        )
        intersection = intersection.view(knn_edge_index.dtype).reshape(-1, ncols)
        intersection = torch.tensor(intersection, dtype=torch.long).T
        data.edge_index = intersection
        return data

    def _finalise_graph_properties(self, data):
        # TODO: check if this step is needed for intersection graph
        if data.x.ndim == 1:
            data.x = data.x.view(-1, 1)
        data = ToUndirected()(data)
        data.edge_index, data.edge_attr = add_self_loops(data["edge_index"])
        row, col = data.edge_index
        data.edge_weight = 1.0 / degree(col, data.num_nodes)[col]
        return data

    def _create_data_splits(self, data, val_file, test_file):
        all_xs = data["pos"][:, 0]
        all_ys = data["pos"][:, 1]

        # Mark everything as training data first
        train_mask = torch.ones(data.num_nodes, dtype=torch.bool)

        # Mask unlabelled data to ignore during training
        unlabelled_inds = (data.y == 0).nonzero()[0]
        unlabelled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        unlabelled_mask[unlabelled_inds] = True
        data.unlabelled_mask = unlabelled_mask
        train_mask[unlabelled_inds] = False

        # Mask validation nodes
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        if val_file is not None:
            val_node_inds = []
            patches_df = pd.read_csv(val_file)
            for row in patches_df.itertuples(index=False):
                val_node_inds.extend(
                    get_nodes_within_tiles(
                        (row.x, row.y), row.width, row.height, all_xs, all_ys
                    )
                )
            val_mask[val_node_inds] = True
            train_mask[val_node_inds] = False

        # Mask test nodes
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        if test_file is not None:
            test_node_inds = []
            patches_df = pd.read_csv(test_file)
            for row in patches_df.itertuples(index=False):
                test_node_inds.extend(
                    get_nodes_within_tiles(
                        (row.x, row.y), row.width, row.height, all_xs, all_ys
                    )
                )
            test_mask[test_node_inds] = True
            train_mask[test_node_inds] = False

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data


# class DelaunyGraphConstructor(GraphConstructor):
#     def construct(self, foo: int, bar: str) -> Data:
#         self.read_hdf5()
#
#
# class KNNGraphConstructor(GraphConstructor):
#     def construct(self, foo: int, bar: str) -> Data:
#         self.read_hdf5()


def get_nodes_within_tiles(tile_coords, tile_width, tile_height, all_xs, all_ys):
    tile_min_x, tile_min_y = tile_coords[0], tile_coords[1]
    tile_max_x, tile_max_y = tile_min_x + tile_width, tile_min_y + tile_height
    mask = np.logical_and(
        (np.logical_and(all_xs > tile_min_x, (all_ys > tile_min_y))),
        (np.logical_and(all_xs < tile_max_x, (all_ys < tile_max_y))),
    )
    return mask.nonzero()[0].tolist()


class Placenta(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        graph_constructor: GraphConstructor = None,
    ):
        self._graph_constructor = graph_constructor
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def graph_constructor(self):
        if self._graph_constructor is None:
            self._graph_constructor = DefaultGraphConstructor(Path(self.raw_dir))
        return self._graph_constructor

    @property
    def raw_file_names(self):
        return [
            "wsi_1.hdf5",
            "wsi_1.tsv",
            "wsi_2.hdf5",
            "wsi_2.tsv",
            "val_patches.csv",
            "test_patches.csv",
        ]

    @property
    def processed_file_names(self):
        return ["wsi_1.pt", "wsi_2.pt"]

    def download(self):
        pass  # Anon until camera-ready version
        # Download to `self.raw_dir`
        # path = download_url(url, self.raw_dir)

    def process(self):
        val_path = Path(self.raw_dir) / self.raw_file_names[-2]
        test_path = Path(self.raw_dir) / self.raw_file_names[-1]

        wsi_1_data = self.raw_file_names[0]
        wsi_1_gt = self.raw_file_names[1]
        wsi_2_data = self.raw_file_names[2]
        wsi_2_gt = self.raw_file_names[3]

        data = self.graph_constructor.construct(wsi_1_data, wsi_1_gt)
        data = self.graph_constructor.node_splits(data, val_path, test_path)
        torch.save(data, osp.join(self.processed_dir, "wsi_1.pt"))

        data = self.graph_constructor.construct(wsi_2_data, wsi_2_gt)
        data = self.graph_constructor.node_splits(data, None, None)
        torch.save(data, osp.join(self.processed_dir, "wsi_2.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        wsi_1_data = torch.load(osp.join(self.processed_dir, "wsi_1.pt"))
        wsi_2_data = torch.load(osp.join(self.processed_dir, "wsi_2.pt"))
        data = Batch.from_data_list([wsi_1_data, wsi_2_data])
        return data
