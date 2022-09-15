import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv, norm


class SupervisedSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super(SupervisedSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.bns.append(norm.BatchNorm(hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = x.relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = self.bns[i](x)
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)

            if i == self.num_layers - 2:
                embeddings = x_all.detach().clone()

        return x_all, embeddings
