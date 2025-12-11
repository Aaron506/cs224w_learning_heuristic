import numpy as np
from GraphPlanner.state_node import Pose2dState
from GraphPlanner.sailboat_polar import SailboatPolar
from scipy.linalg import solve_triangular
from typing import Tuple

class PlanningHeuristic():
    """Base class representing a cost-to-go heuristic for use in graph-based motion planning."""
    def __init__(self):
        pass

    def set_goal(self, goal: Pose2dState, success_Q: np.ndarray = None):
        """Set the goal and possibly success_Q goal region ellipse."""
        pass

    def set_wind(self, wind_yaw: float, wind_speed: float):
        """Set the wind information."""
        pass

    def set_map(self, costmap=None, target_list=None):
        """Set the target graph information."""
        pass

    def find_cost_to_go(self, poses: np.ndarray, time: float) -> np.ndarray:
        """Compute the cost to go for each pose at given time using the heuristic."""
        pass