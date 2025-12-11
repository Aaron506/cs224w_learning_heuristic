import numpy as np
from dataclasses import dataclass 

@dataclass
class Pose2dState():
    x: float
    y: float
    psi: float
    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi])

class StateNode(Pose2dState):
    """Class for representing one node in the graph search planner/
    Inputs:
    x: ENU x-coordinate (m)
    y: ENU y-coordinate (m)
    psi: Ego-vessel heading (rad)
    time: Time (sec) at which node reached
    parent: Points to parent node
    cost: Cost accumulated to reach current node from start
    cost_to_go: Heuristic cost-to-go to reach goal state
    status: False if popped, True if in queue
    """
    def __init__(self, pose: Pose2dState, time: float, parent: 'StateNode', cost: float, cost_to_go: float, status: bool) -> None:
        super().__init__(pose.x, pose.y, pose.psi)
        self.time = time
        self.parent = parent
        self.cost = cost
        self.cost_to_go = cost_to_go
        self.status = status

    def get_priority(self, priority_type='tot_cost'):
        """Get the node priority, where lower indicates higher priority."""
        # Prioritize expanding the longest in time paths first
        if priority_type == 'dfs':
            return -1 * self.time
        # Prioritize expanding the shortest in time paths first
        elif priority_type == 'bfs':
            return self.time
        # Prioritize the lowest paths by estimated total cost first
        elif priority_type == 'tot_cost':
            return self.cost + self.cost_to_go
        else:
            raise ValueError("priority_type either dfs, bfs, or tot_cost")

def form_success_Q(sigmas: np.ndarray, rot_angle: float = 0):
    """
    Forms an ellipsoid defining success region for motion planner
    Inputs:
    sigmas: (ontrack_error, offtrack_error, yaw_error) limits defining each ellipsoid axis, 
    infinity to ignore a dimension
    rot_angle: ellipsoid rotation e.g., the yaw of the reference line/track
    """
    # sqrt(offset.T @ success_Q @ offset) <= 1
    # First, define in track frame
    success_Q = np.diag(np.divide(1, sigmas**2))
    # Passive rotation from EN to track frame
    R = np.array([
        [np.cos(rot_angle), np.sin(rot_angle)],
        [-np.sin(rot_angle), np.cos(rot_angle)]
    ])
    success_Q[:2,:2] = R.T @ success_Q[:2,:2] @ R
    return success_Q