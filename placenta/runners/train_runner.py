import copy
from dataclasses import dataclass, asdict
from typing import Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import (
    DataLoader,
    ClusterData,
    ClusterLoader,
    NeighborSampler,
    NeighborLoader,
    GraphSAINTRandomWalkSampler,
    ShaDowKHopSampler,
)

from placenta.models.graphsage import SupervisedSAGE
from placenta.models.clustergcn import ClusterGCN
from placenta.models.gat import GAT, GATv2
from placenta.models.graphsaint import GraphSAINT
from placenta.models.shadow import ShaDowGCN
from placenta.models.sign import SIGN as SIGN_MLP
from placenta.models.mlp import MLP
from placenta.organs.organs import Organ
from placenta.graphs.enums import ModelsArg


@dataclass
class TrainParams:
    data: Data
    device: str
    pretrained: Optional[str]
    model_type: ModelsArg
    batch_size: int
    num_neighbours: int
    epochs: int
    layers: int
    hidden_units: int
    dropout: float
    learning_rate: float
    weighted_loss: bool
    custom_weights: bool
    organ: Organ

    def save(self, run_path):
        with open(run_path / "train_params.csv", 'w') as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class BatchResult:
    loss: torch.Tensor
    correct_predictions: int
    nodes: int


class TrainRunner:
    def __init__(self, params: TrainParams):
        self.params: TrainParams = params
        self._model: Optional[nn.Module] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._optimiser: Optional[torch.optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None
        self.num_classes = len(self.params.organ.tissues)

    @staticmethod
    def new(params: TrainParams) -> "TrainRunner":
        cls = {
            ModelsArg.graphsage: GraphSAGERunner,
            ModelsArg.clustergcn: ClusterGCNRunner,
            ModelsArg.gat: GATRunner,
            ModelsArg.gatv2: GATV2Runner,
            ModelsArg.graphsaint: GraphSAINTRunner,
            ModelsArg.shadow: ShaDowRunner,
            ModelsArg.sign: SIGNRunner,
            ModelsArg.mlp: MLPRunner,
        }
        ModelClass = cls[params.model_type]
        return ModelClass(params)

    @property
    def model(self):
        if self._model is None:
            # If we are using a pretrained model, load it from disk instead of
            # creating a new one.
            if self.params.pretrained is not None:
                self._model = torch.load(self.params.pretrained)
            else:
                self._model = self.setup_model()

        return self._model

    @property
    def train_loader(self):
        if self._train_loader is None:
            self._setup_loaders()
        return self._train_loader

    @property
    def val_loader(self):
        if self._val_loader is None:
            self._setup_loaders()
        return self._val_loader

    @property
    def optimiser(self):
        if self._optimiser is None:
            self._setup_optimiser()
        return self._optimiser

    @property
    def criterion(self):
        if self._criterion is None:
            self._criterion = self.setup_criterion()
        return self._criterion

    def _setup_loaders(self):
        self._train_loader, self._val_loader = self.setup_dataloader()

    def _setup_optimiser(self):
        self._optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.params.learning_rate
        )

    @classmethod
    def setup_dataloader(cls):
        raise NotImplementedError(
            f"setup_dataloader not implemented for {cls.__name__}"
        )

    @classmethod
    def setup_model(cls):
        raise NotImplementedError(f"setup_model not implemented for {cls.__name__}")

    @classmethod
    def setup_criterion(cls):
        raise NotImplementedError(f"setup_criterion not implemented for {cls.__name__}")

    def train(self):
        self.model.train()
        total_loss = 0
        total_examples = 0
        total_correct = 0
        for batch in self.train_loader:
            batch = batch.to(self.params.device)
            self.optimiser.zero_grad()

            result = self.process_batch(batch)

            total_loss += result.loss.item() * result.nodes
            total_correct += result.correct_predictions
            total_examples += result.nodes
        return total_loss / total_examples, total_correct / total_examples

    @classmethod
    def process_batch(cls, batch) -> BatchResult:
        """Processes a batch and returns the number of correct predictions."""
        raise NotImplementedError(f"setup_criterion not implemented for {cls.__name__}")

    def validate(self):
        data = self.params.data
        self.model.eval()
        out, _ = self.model.inference(data.x, self.val_loader, self.params.device)
        out = out.argmax(dim=-1)
        y = data.y.to(out.device)
        train_accuracy = int((out[data.train_mask].eq(y[data.train_mask])).sum()) / int(
            data.train_mask.sum()
        )
        val_accuracy = int((out[data.val_mask].eq(y[data.val_mask])).sum()) / int(
            data.val_mask.sum()
        )
        return train_accuracy, val_accuracy

    def save_state(self, run_path, logger, epoch):
        torch.save(self.model, run_path / f"{epoch}_graph_model.pt")
        logger.to_csv(run_path / "graph_train_stats.csv")


class GraphSAGERunner(TrainRunner):
    def setup_dataloader(self):
        train_loader = NeighborLoader(
            self.params.data,
            input_nodes=self.params.data.train_mask,
            num_neighbors=[
                self.params.num_neighbours for _ in range(self.params.layers)
            ],
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=12,
        )
        val_loader = NeighborLoader(
            copy.copy(self.params.data),
            num_neighbors=[-1],
            shuffle=False,
            batch_size=512,
        )
        val_loader.data.num_nodes = self.params.data.num_nodes
        val_loader.data.n_id = torch.arange(self.params.data.num_nodes)

        return train_loader, val_loader

    def setup_model(self):
        return SupervisedSAGE(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )

    def setup_criterion(self):
        if self.params.weighted_loss:
            data_classes = self.params.data.y[self.params.data.train_mask].numpy()
            class_weights = _compute_tissue_weights(
                data_classes, self.params.organ, self.params.custom_weights
            )
            class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(self.params.device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def process_batch(self, batch) -> BatchResult:
        out = self.model(batch.x, batch.edge_index)
        train_out = out[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        loss = self.criterion(train_out, train_y)
        loss.backward()
        self.optimiser.step()

        nodes = batch.train_mask.sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int(train_out.argmax(dim=-1).eq(train_y).sum().item()),
            nodes=nodes,
        )


class ClusterGCNRunner(TrainRunner):
    def setup_dataloader(self):
        cluster_data = ClusterData(
            self.params.data,
            num_parts=int(self.params.data.x.size()[0] / self.params.num_neighbours),
            recursive=False,
        )
        train_loader = ClusterLoader(
            cluster_data,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=12,
        )
        val_loader = NeighborSampler(
            self.params.data.edge_index,
            sizes=[-1],
            batch_size=1024,
            shuffle=False,
            num_workers=12,
        )
        return train_loader, val_loader

    def setup_model(self):
        return ClusterGCN(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )

    def setup_criterion(self):
        return _default_criterion(
            self.params.weighted_loss,
            self.params.data,
            self.params.organ,
            self.params.custom_weights,
            self.params.device,
        )

    def process_batch(self, batch) -> BatchResult:
        out = self.model(batch.x, batch.edge_index)
        train_out = out[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        loss = self.criterion(train_out, train_y)
        loss.backward()
        self.optimiser.step()
        nodes = batch.train_mask.sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int(train_out.argmax(dim=-1).eq(train_y).sum().item()),
            nodes=nodes,
        )


class GATRunner(ClusterGCNRunner):
    def setup_model(self):
        return GAT(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            heads=4,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )


class GATV2Runner(ClusterGCNRunner):
    def setup_model(self):
        return GATv2(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            heads=4,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )


class GraphSAINTRunner(TrainRunner):
    def setup_dataloader(self):
        train_loader = GraphSAINTRandomWalkSampler(
            self.params.data,
            batch_size=self.params.batch_size,
            walk_length=self.params.layers,
            num_steps=30,
            sample_coverage=self.params.num_neighbours,
            shuffle=True,
            num_workers=12,
        )
        val_loader = NeighborSampler(
            self.params.data.edge_index,
            sizes=[-1],
            batch_size=1024,
            shuffle=False,
            num_workers=12,
        )
        return train_loader, val_loader

    def setup_model(self):
        return GraphSAINT(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )

    def setup_criterion(self):
        return _default_criterion(
            self.params.weighted_loss,
            self.params.data,
            self.params.organ,
            self.params.custom_weights,
            self.params.device,
            reduction="none",
        )

    def process_batch(self, batch) -> BatchResult:
        self.model.set_aggr("add")
        edge_weight = batch.edge_norm * batch.edge_weight
        out = self.model(batch.x, batch.edge_index, edge_weight)
        loss = self.criterion(out, batch.y)
        loss = (loss * batch.node_norm)[batch.train_mask].sum()
        loss.backward()
        self.optimiser.step()
        nodes = batch.train_mask.sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int(
                out[batch.train_mask]
                .argmax(dim=-1)
                .eq(batch.y[batch.train_mask])
                .sum()
                .item()
            ),
            nodes=nodes,
        )

    def validate(self):
        data = self.params.data
        self.model.eval()
        self.model.set_aggr("mean")
        out, _ = self.model.inference(data.x, self.val_loader, self.params.device)
        out = out.argmax(dim=-1)
        y = data.y.to(out.device)
        train_accuracy = int((out[data.train_mask].eq(y[data.train_mask])).sum()) / int(
            data.train_mask.sum()
        )
        val_accuracy = int((out[data.val_mask].eq(y[data.val_mask])).sum()) / int(
            data.val_mask.sum()
        )
        return train_accuracy, val_accuracy


class ShaDowRunner(TrainRunner):
    def setup_dataloader(self):
        train_loader = ShaDowKHopSampler(
            self.params.data,
            depth=6,
            num_neighbors=self.params.num_neighbours,
            node_idx=self.params.data.train_mask,
            batch_size=self.params.batch_size,
            num_workers=12,
        )
        val_loader = ShaDowKHopSampler(
            self.params.data,
            depth=6,
            num_neighbors=self.params.num_neighbours,
            node_idx=None,
            batch_size=self.params.batch_size,
            num_workers=12,
            shuffle=False,
        )
        return train_loader, val_loader

    def setup_model(self):
        return ShaDowGCN(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )

    def setup_criterion(self):
        return _default_criterion(
            self.params.weighted_loss,
            self.params.data,
            self.params.organ,
            self.params.custom_weights,
            self.params.device,
        )

    def process_batch(self, batch) -> BatchResult:
        out = self.model(batch.x, batch.edge_index, batch.batch, batch.root_n_id)
        loss = self.criterion(out, batch.y)
        loss.backward()
        self.optimiser.step()
        nodes = out.size()[0]
        return BatchResult(
            loss=loss,
            correct_predictions=int(out.argmax(dim=-1).eq(batch.y).sum().item()),
            nodes=nodes,
        )

    def validate(self):
        data = self.params.data
        self.model.eval()
        out = []
        for batch in self.val_loader:
            batch = batch.to(self.params.device)
            batch_out = self.model(
                batch.x, batch.edge_index, batch.batch, batch.root_n_id
            )
            out.append(batch_out)
        out = torch.cat(out, dim=0)
        out = out.argmax(dim=-1)
        y = data.y.to(out.device)
        train_accuracy = int((out[data.train_mask].eq(y[data.train_mask])).sum()) / int(
            data.train_mask.sum()
        )
        val_accuracy = int((out[data.val_mask].eq(y[data.val_mask])).sum()) / int(
            data.val_mask.sum()
        )
        return train_accuracy, val_accuracy


class SIGNRunner(TrainRunner):
    def setup_dataloader(self):
        train_idx = self.params.data.train_mask.nonzero(as_tuple=False).view(-1)
        val_idx = self.params.data.val_mask.nonzero(as_tuple=False).view(-1)
        train_loader = DataLoader(
            train_idx, batch_size=self.params.batch_size, shuffle=True, num_workers=12
        )
        val_loader = DataLoader(
            val_idx, batch_size=self.params.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def setup_model(self):
        return SIGN_MLP(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )

    def setup_criterion(self):
        return _default_criterion(
            self.params.weighted_loss,
            self.params.data,
            self.params.organ,
            self.params.custom_weights,
            self.params.device,
        )

    def process_batch(self, batch) -> BatchResult:
        sign_k = self.model.num_layers
        data = self.params.data
        train_x = [data.x[batch].to(self.params.device)]
        train_y = data.y[batch].to(self.params.device)
        train_x += [
            data[f"x{i}"][batch].to(self.params.device) for i in range(1, sign_k + 1)
        ]
        out = self.model(train_x)
        loss = self.criterion(out, train_y)
        loss.backward()
        self.optimiser.step()
        nodes = data.train_mask[batch].sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int((out.argmax(dim=-1).eq(train_y)).sum()),
            nodes=nodes,
        )

    def validate(self):
        data = self.params.data
        self.model.eval()
        train_out = []
        for idx in self.train_loader:
            eval_x = [data.x[idx].to(self.params.device)]
            eval_x += [
                data[f"x{i}"][idx].to(self.params.device)
                for i in range(1, self.model.num_layers + 1)
            ]
            out_i, _ = self.model.inference(eval_x)
            train_out.append(out_i)
        train_out = torch.cat(train_out, dim=0)
        train_out = train_out.argmax(dim=-1)
        y = data.y.to(train_out.device)
        train_accuracy = int((train_out.eq(y[data.train_mask])).sum()) / int(
            data.train_mask.sum()
        )
        val_out = []
        for idx in self.val_loader:
            eval_x = [data.x[idx].to(self.params.device)]
            eval_x += [
                data[f"x{i}"][idx].to(self.params.device)
                for i in range(1, self.model.num_layers + 1)
            ]
            out_i, _ = self.model.inference(eval_x)
            val_out.append(out_i)
        val_out = torch.cat(val_out, dim=0)
        val_out = val_out.argmax(dim=-1)
        val_accuracy = int((val_out.eq(y[data.val_mask])).sum()) / int(
            data.val_mask.sum()
        )
        return train_accuracy, val_accuracy


class MLPRunner(TrainRunner):
    def setup_dataloader(self):
        train_idx = self.params.data.train_mask.nonzero(as_tuple=False).view(-1)
        val_idx = self.params.data.val_mask.nonzero(as_tuple=False).view(-1)
        train_loader = DataLoader(
            train_idx, batch_size=self.params.batch_size, shuffle=True, num_workers=12
        )
        val_loader = DataLoader(
            val_idx, batch_size=self.params.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def setup_model(self):
        return MLP(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )

    def setup_criterion(self):
        return _default_criterion(
            self.params.weighted_loss,
            self.params.data,
            self.params.organ,
            self.params.custom_weights,
            self.params.device,
        )

    def process_batch(self, batch) -> BatchResult:
        data = self.params.data
        train_x = data.x[batch].to(self.params.device)
        train_y = data.y[batch].to(self.params.device)
        out = self.model(train_x)
        loss = self.criterion(out, train_y)
        loss.backward()
        self.optimiser.step()
        nodes = data.train_mask[batch].sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int((out.argmax(dim=-1).eq(train_y)).sum()),
            nodes=nodes,
        )

    def validate(self):
        data = self.params.data
        self.model.eval()
        train_out = []
        for idx in self.train_loader:
            eval_x = data.x[idx].to(self.params.device)
            out_i, _ = self.model.inference(eval_x)
            train_out.append(out_i)
        train_out = torch.cat(train_out, dim=0)
        train_out = train_out.argmax(dim=-1)
        y = data.y.to(train_out.device)
        train_accuracy = int((train_out.eq(y[data.train_mask])).sum()) / int(
            data.train_mask.sum()
        )
        val_out = []
        for idx in self.val_loader:
            eval_x = data.x[idx].to(self.params.device)
            out_i, _ = self.model.inference(eval_x)
            val_out.append(out_i)
        val_out = torch.cat(val_out, dim=0)
        val_out = val_out.argmax(dim=-1)
        val_accuracy = int((val_out.eq(y[data.val_mask])).sum()) / int(
            data.val_mask.sum()
        )
        return train_accuracy, val_accuracy


def _default_criterion(
    weighted_loss, data, organ, custom_weights, device, reduction="mean"
):
    if weighted_loss:
        data_classes = data.y[data.train_mask].numpy()
        class_weights = _compute_tissue_weights(data_classes, organ, custom_weights)
        class_weights = torch.FloatTensor(class_weights)
        class_weights = class_weights.to(device)
        criterion = torch.nn.NLLLoss(weight=class_weights, reduction=reduction)
    else:
        criterion = torch.nn.NLLLoss(reduction=reduction)
    return criterion


def _compute_tissue_weights(data_classes, organ, custom_weights):
    unique_classes = np.unique(data_classes)
    if not custom_weights:
        weighting = "balanced"
    else:
        custom_weights = [1, 0.85, 0.9, 10.5, 0.8, 1.3, 5.6, 3, 77]
        weighting = dict(zip(list(unique_classes), custom_weights))
    class_weights = compute_class_weight(
        weighting, classes=unique_classes, y=data_classes
    )
    # Account for missing tissues in training data
    classes_in_training = set(unique_classes)
    all_classes = {tissue.id for tissue in organ.tissues}
    missing_classes = list(all_classes - classes_in_training)
    missing_classes.sort()
    for i in missing_classes:
        class_weights = np.insert(class_weights, i, 0.0)
    return class_weights
