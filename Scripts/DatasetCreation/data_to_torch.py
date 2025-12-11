import pickle
import os
from DatasetCreation.data_generator import GraphDatum
import torch
from torch_geometric.data import Data, Dataset

def clique_edge_index(num_nodes):
    # Form a fully connected, undirected graph
    idx = torch.arange(num_nodes)
    pairs = torch.combinations(idx, r=2)
    src = pairs[:,0]
    dst = pairs[:,1]
    # Undirected so both ways
    edge_index = torch.cat([
        torch.stack([src, dst], dim=0),
        torch.stack([dst, src], dim=0)], dim=1
    )
    return edge_index

def form_target_graph(targets: torch.tensor, wind_yaw: float, wind_speed: float) -> Data:
    num_nodes = len(targets)
    if num_nodes == 0:
        raise ValueError("No targets, so no nodes")
    
    # Each node will have initial features (px,py,vx,vy,wind_yaw,wind_speed)
    x_list = []
    for target in targets:
        px, py, vx, vy, h, w = target
        x_list.append([px, py, vx, vy, wind_yaw, wind_speed])
    x = torch.tensor(x_list, dtype=torch.float) # (N, 6)

    # Fully connected graph
    edge_index = clique_edge_index(num_nodes)

    data = Data(
        x=x,
        edge_index=edge_index
    )

    return data

def graphdatum_to_pyg(datum):
    """Convert a GraphDatum object into torch_geometric.data.Data object."""
    data = form_target_graph(datum.targets, datum.wind_yaw, datum.wind_speed)    
    
    # The graph label is the scalar cost-to-go
    data.y = torch.tensor([[datum.cost_to_go]], dtype=torch.float)

    # Also, store additional attributes
    data.traj_id = datum.traj_id
    data.path_index = datum.path_index

    data.bboxs = torch.from_numpy(datum.targets).float()
    data.pose = torch.tensor([[datum.pose.x, datum.pose.y, datum.pose.psi]], dtype=torch.float)
    data.wind_yaw = torch.tensor([[datum.wind_yaw]], dtype=torch.float)
    data.wind_speed = torch.tensor([[datum.wind_speed]], dtype=torch.float)
    data.goal_wrt_costmap = torch.tensor(
        [[datum.goal_wrt_costmap.x, datum.goal_wrt_costmap.y, datum.goal_wrt_costmap.psi]],
        dtype=torch.float)

    return data

class GraphDatumDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        # dataset is list of GraphDatum objects, so convert each to PyG Data
        self.graphs = [graphdatum_to_pyg(d) for d in dataset]

        # Dict to look up index by (traj_id, path_index)
        self.idx_by_key = {(d.traj_id, d.path_index): i for i, d in enumerate(dataset)}

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    # Can also index by traj_id and path_index
    def get_by_traj_and_path(self, traj_id: str, path_index: int) -> Data:
        idx = self.idx_by_key[(traj_id, path_index)]
        return self[idx]