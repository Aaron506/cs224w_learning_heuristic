import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool

class TargetGraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, layer_type, **layer_args):
        """Given the input graph of targets, find a graph encoding."""
        super(TargetGraphEncoder, self).__init__()

        # A list of message-passing layers
        # input_dim -> hidden_dim, intermediate hidden_dim -> hidden_dim
        # last time hidden_dim -> output_dim
        # Make sure to set concat = False if number of heads > 1 if using GATConv
        if layer_type == 'gcn':
          layer = GCNConv
        elif layer_type == 'gat':
          layer = GATConv
        else:
          raise ValueError('layer_type either gcn or gat')
        conv_layers = [layer(input_dim, hidden_dim, **layer_args)] + \
                      [layer(hidden_dim, hidden_dim, **layer_args) for _ in range(num_layers-2)] + \
                      [layer(hidden_dim, output_dim, **layer_args)]
        self.convs = torch.nn.ModuleList(conv_layers)

        # A list of 1D batch normalization layers
        bn_layers = [torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)]
        self.bns = torch.nn.ModuleList(bn_layers)

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Pooling across nodes into one embedding for graph
        self.pool = global_add_pool

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, batched_data, pool=True):
      x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

      out = x
      num_layers = len(self.convs)

      for i, conv in enumerate(self.convs):
        # a. Apply the GCN
        out = conv(out, edge_index)
        if i < num_layers - 1:
          # b. Apply BN
          out = self.bns[i](out)
          # c. Apply ReLU
          out = F.relu(out)
          # d. Apply dropout
          out = F.dropout(out, p=self.dropout, training=self.training)

      # Pool into one global graph embedding
      if pool:
        out = self.pool(out, batch)

      return out