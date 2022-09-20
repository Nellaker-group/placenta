from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.transforms import SIGN
from torch_geometric.loader import (
    DataLoader,
    NeighborSampler,
    NeighborLoader,
    ShaDowKHopSampler,
)

from placenta.models.graphsaint import GraphSAINT
from placenta.models.sign import SIGN as SIGN_MLP
from placenta.organs import Organ
from placenta.enums import ModelsArg


@dataclass
class EvalParams:
    data: Data
    device: str
    pretrained: str
    model_type: ModelsArg
    batch_size: int
    organ: Organ


@dataclass
class EvalResult:
    out: torch.Tensor
    embeddings: torch.Tensor


class EvalRunner:
    def __init__(self, params: EvalParams):
        self.params: EvalParams = params
        self._model: Optional[nn.Module] = None
        self._loader: Optional[DataLoader] = None

        if isinstance(self.model, SIGN_MLP):
            self.params.data = SIGN(self.model.num_layers)(self.params.data)

    @staticmethod
    def new(params: EvalParams) -> "EvalRunner":
        cls = {
            ModelsArg.graphsage: LoaderRunner,
            ModelsArg.clustergcn: SamplerRunner,
            ModelsArg.gat: SamplerRunner,
            ModelsArg.gatv2: SamplerRunner,
            ModelsArg.graphsaint: SamplerRunner,
            ModelsArg.shadow: ShaDowRunner,
            ModelsArg.sign: SIGNRunner,
            ModelsArg.mlp: MLPRunner,
        }
        ModelClass = cls[params.model_type]
        return ModelClass(params)

    @property
    def model(self):
        if self._model is None:
            self._model = torch.load(
                self.params.pretrained, map_location=self.params.device
            )
        return self._model

    @property
    def loader(self):
        if self._loader is None:
            self._setup_loader()
        return self._loader

    def _setup_loader(self):
        self._loader = self.setup_dataloader()

    @classmethod
    def setup_dataloader(cls):
        raise NotImplementedError(
            f"setup_dataloader not implemented for {cls.__name__}"
        )

    @torch.no_grad()
    def inference(self):
        self.model.eval()
        result = self.model_inference()
        out = result.out
        predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()
        predicted_labels = predicted_labels.cpu().numpy()
        out = out.cpu().detach().numpy()
        return out, result.embeddings, predicted_labels

    @classmethod
    def model_inference(cls) -> EvalResult:
        """Runs inference across all data in loader."""
        raise NotImplementedError(f"model_inference not implemented for {cls.__name__}")


class LoaderRunner(EvalRunner):
    def setup_dataloader(self):
        eval_loader = NeighborLoader(
            self.params.data,
            num_neighbors=[-1],
            batch_size=self.params.batch_size,
            shuffle=False,
        )
        eval_loader.data.num_nodes = self.params.data.num_nodes
        eval_loader.data.n_id = torch.arange(self.params.data.num_nodes)
        return eval_loader

    def model_inference(self) -> EvalResult:
        out, embeddings = self.model.inference(
            self.params.data.x, self.loader, self.params.device
        )
        return EvalResult(out=out, embeddings=embeddings)


class SamplerRunner(EvalRunner):
    def setup_dataloader(self):
        return NeighborSampler(
            self.params.data.edge_index,
            node_idx=None,
            sizes=[-1],
            batch_size=self.params.batch_size,
            shuffle=False,
        )

    def model_inference(self) -> EvalResult:
        if isinstance(self.model, GraphSAINT):
            self.model.set_aggr("mean")
        out, embeddings = self.model.inference(
            self.params.data.x, self.loader, self.params.device
        )
        return EvalResult(out=out, embeddings=embeddings)


class ShaDowRunner(EvalRunner):
    def setup_dataloader(self):
        return ShaDowKHopSampler(
            self.params.data,
            depth=6,
            num_neighbors=12,
            node_idx=None,
            batch_size=self.params.batch_size,
            shuffle=False,
        )

    def model_inference(self) -> EvalResult:
        out = []
        embeddings = []
        for batch in self.loader:
            batch = batch.to(self.params.device)
            batch_out, batch_embed = self.model.inference(
                batch.x, batch.edge_index, batch.batch, batch.root_n_id
            )
            out.append(batch_out)
            embeddings.append(batch_embed)
        out = torch.cat(out, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        return EvalResult(out=out, embeddings=embeddings)


class SIGNRunner(EvalRunner):
    def setup_dataloader(self):
        return DataLoader(
            range(self.params.data.num_nodes),
            batch_size=self.params.batch_size,
            shuffle=False,
        )

    def model_inference(self):
        out = []
        embeddings = []
        for idx in self.loader:
            eval_x = [self.params.data.x[idx].to(self.params.device)]
            eval_x += [
                self.params.data[f"x{i}"][idx].to(self.params.device)
                for i in range(1, self.model.num_layers + 1)
            ]
            out_i, emb_i = self.model.inference(eval_x)
            out.append(out_i)
            embeddings.append(emb_i)
        out = torch.cat(out, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        return EvalResult(out=out, embeddings=embeddings)


class MLPRunner(EvalRunner):
    def setup_dataloader(self):
        return DataLoader(
            range(self.params.data.num_nodes),
            batch_size=self.params.batch_size,
            shuffle=False,
        )

    def model_inference(self):
        out = []
        embeddings = []
        for idx in self.loader:
            eval_x = self.params.data.x[idx].to(self.params.device)
            out_i, emb_i = self.model.inference(eval_x)
            out.append(out_i)
            embeddings.append(emb_i)
        out = torch.cat(out, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        return EvalResult(out=out, embeddings=embeddings)
