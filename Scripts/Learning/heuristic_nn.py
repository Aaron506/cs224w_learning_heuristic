from Learning.target_graph_encoder import TargetGraphEncoder
import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
    """Implement a simple feedforward MLP."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
      super(MLP, self).__init__()

      # A list of linear layers
      # input_dim -> hidden_dim, intermediate hidden_dim -> hidden_dim
      # last time hidden_dim -> output_dim
      layers = [Linear(input_dim, hidden_dim)] + \
                [Linear(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + \
                [Linear(hidden_dim, output_dim)]
      self.layers = torch.nn.ModuleList(layers)

      # A list of 1D batch normalization layers
      bn_layers = [torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)]
      self.bns = torch.nn.ModuleList(bn_layers)

      # Probability of an element getting zeroed
      self.dropout = dropout

    def reset_parameters(self):
      for layer in self.layers:
        layer.reset_parameters()
      for bn in self.bns:
        bn.reset_parameters()

    def forward(self, x):
      out = x
      num_layers = len(self.layers)

      for i, layer in enumerate(self.layers):
        # a. Apply the GCN
        out = layer(out)
        if i < num_layers - 1:
          # b. Apply BN
          out = self.bns[i](out)
          # c. Apply ReLU
          out = F.relu(out)
          # d. Apply dropout
          out = F.dropout(out, p=self.dropout, training=self.training)

      return out

class HeuristicNN(torch.nn.Module):
    def __init__(self, args, layer_type='gcn', device=None, **layer_args):
        super(HeuristicNN, self).__init__()

        # 1. g0(curr_pose, wind_yaw, wind_speed) MLP
        self.no_target_mlp = MLP(5, args['no_target_hidden'], 1, args['no_target_num_layers'], args['dropout'])
        # 2. TargetGraphEncoder
        # input targets graph B_k -> phi(B_k) global graph embedding
        self.graph_encoder = TargetGraphEncoder(6, args['graph_enc_hidden'], args['graph_enc_out'],
                                                args['graph_enc_num_layers'], args['dropout'], layer_type, **layer_args)
        # 3. g1(curr_pose, wind_yaw, wind_speed, phi(B_k)) MLP
        self.target_mlp = MLP(5 + args['graph_enc_out'], args['target_hidden'], 1, 
                              args['target_num_layers'], args['dropout'])

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.to(self.device)

    def get_base_heuristic(self, base_feats):
      """Compute the base heuristic g0 with no targets."""
      # base_feats is (B, 5) where concatenate pose: (B, 3), wind_speed: (B, 1), wind_yaw: (B, 1)
      return self.no_target_mlp.forward(base_feats) # (B,)
    
    # Just compute the target graph encoding
    def get_target_encoding(self, batched_data):
      """Compute the graph-level encoding of the targets."""
      return self.graph_encoder.forward(batched_data, pool=True)

    def get_target_heuristic(self, base_feats, phi):
      """Compute the target-adjustment heuristic g1 using target encoding phi."""
      # base_feats: (B,5), phi: (B,phi_dim)
      target_feats = torch.cat([base_feats, phi], dim=-1) # (B, 5 + phi_dim)
      return self.target_mlp.forward(target_feats) # (B, )

    def forward(self, batched_data):
      """Compute full heuristic g0 + g1 from scratch."""
      # Extract per-graph scalar features
      pose = batched_data.pose # (B, 3)
      wind_speed = batched_data.wind_speed # (B, 1)
      wind_yaw = batched_data.wind_yaw # (B, 1)

      # Base input: pose + wind features
      base_feats = torch.cat([pose, wind_yaw, wind_speed], dim=-1) # (B, 5)

      # 1. Call no_target_mlp only using base_feats
      g0 = self.get_base_heuristic(base_feats)

      # 2. Call target_encoder, uses only the per-node graph data
      phi = self.get_target_encoding(batched_data)

      # 3. Call target MLP using base_feats and phi
      g1 = self.get_target_heuristic(base_feats, phi)

      total = g0 + g1
      return g0, g1, total