import numpy as np
from Math.math_utils import wrap_pi
from GraphPlanner.state_node import StateNode

class ObstacleInterface():
    """Base class representing an obstacle for use in graph-based motion planning."""
    def __init__(self):
        pass

    def find_distances(self, poses: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Given poses and associated times when reached, find distance to obstacle."""
        pass