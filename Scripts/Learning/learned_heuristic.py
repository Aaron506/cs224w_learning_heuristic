import numpy as np
import torch

from GraphPlanner.state_node import Pose2dState
from GraphPlanner.graph_planner import PlanningHeuristic
from Learning.heuristic_nn import HeuristicNN
from Math.geometry import form_2d_passive_rot
from DatasetCreation.data_to_torch import clique_edge_index
from torch_geometric.data import Data

class LearnedHeuristic(PlanningHeuristic):
    """Use a learned heuristic the graph-based motion planning."""
    def __init__(self, model: HeuristicNN):
        self.model = model

        # Store for repeated use the target graph encodings indexed by time
        self.phis = dict()

    def set_goal(self, goal: Pose2dState, success_Q: np.ndarray = None):
        # Note: set_goal should be called after set_wind, set_map
        self.goal = goal
        # Store the passive rotation for repeated use
        self.R = form_2d_passive_rot(goal.psi)
        self.success_Q = success_Q

        # 0. Reset the graph encodings
        self.phis.clear()

        # 1. Convert wind_yaw to be wrt. goal
        self.wind_yaw_wrt_goal = self.wind_yaw - self.goal.psi

    def set_wind(self, wind_yaw: float, wind_speed: float):
        self.wind_yaw = wind_yaw
        self.wind_speed = wind_speed

    def set_map(self, costmap=None, target_list=None):
        # Note: costmap not currently supported
        self.target_list = target_list

    @torch.no_grad
    def find_cost_to_go(self, poses: np.ndarray, time: float, merge_type='total') -> np.ndarray:
        B = poses.shape[0]
        # If merge_type is 'g0', then out stores g0, g1 if 'g1', and g0 + g1 if 'total'
        out = torch.zeros((B,1)).to(self.model.device).float()

        # 1. Convert poses and wind to be wrt. goal
        poses_wrt_goal = poses - self.goal.as_array()
        poses_wrt_goal[:,:2] = (self.R @ poses_wrt_goal[:,:2].T).T
        poses_wrt_goal = torch.from_numpy(poses_wrt_goal).float()
        wind_yaw_wrt_goal = torch.ones((B,1)) * self.wind_yaw_wrt_goal
        wind_speed = torch.ones((B,1)) * self.wind_speed
        base_feats = torch.cat([poses_wrt_goal, wind_yaw_wrt_goal, wind_speed], dim=-1).float() # (B, 5)
        base_feats = base_feats.to(self.model.device)

        # 2. Call base model without targets
        if merge_type in ['g0','total']:
            g0 = self.model.get_base_heuristic(base_feats)
            out += g0

        # 3. Encode the target graph if needed
        if self.target_list is not None and merge_type in ['g1','total']:
            # Check if already have current time cached
            try:
                phi = self.phis[time]
            # Otherwise, make a new encoding
            except KeyError:
                # a. Extract the target info at current time
                target_positions = self.target_list.get_coords_over_time(np.array([time])).squeeze()
                target_velocities = self.target_list.vels.copy()

                # b. Convert to be wrt. goal
                target_positions_wrt_goal = target_positions - self.goal.as_array()[:2]
                target_positions_wrt_goal = (self.R @ target_positions_wrt_goal.T).T
                target_positions_wrt_goal = torch.from_numpy(target_positions_wrt_goal).float()
                target_velocities_wrt_goal = (self.R @ target_velocities.T).T
                target_velocities_wrt_goal = torch.from_numpy(target_velocities_wrt_goal).float()

                # c. Format so that can pass to PyG
                num_nodes = target_positions.shape[0]
                # Node feature matrix x: (px, py, vx, vy, wind_yaw, wind_speed)
                x = torch.cat(
                    [target_positions_wrt_goal, target_velocities_wrt_goal, 
                    torch.ones((num_nodes,1)) * self.wind_yaw_wrt_goal,
                    torch.ones((num_nodes,1)) * self.wind_speed],
                    dim=-1
                )
                x = x.to(self.model.device)
                # Fully connected graph
                edge_index = clique_edge_index(num_nodes).to(self.model.device)
                # Single-graph batch vector
                batch = torch.zeros(num_nodes, dtype=torch.long, device=self.model.device)
                batched_data = Data(x=x, edge_index=edge_index, batch=batch)

                # d. Encode the target graph
                phi = self.model.get_target_encoding(batched_data) # (1,phi_dim)

                # e. Store for future use
                self.phis[time] = phi

                # Can use to verify that are reusing phis, because length should remain small
                # print(f'Added phi, total # = {len(self.phis.keys())}')

            rep_phis = torch.tile(phi, (B,1))
            g1 = self.model.get_target_heuristic(base_feats, rep_phis)
            out += g1

        out = out.cpu().numpy()
        return out
