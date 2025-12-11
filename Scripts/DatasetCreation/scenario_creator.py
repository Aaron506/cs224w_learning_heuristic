import numpy as np
import pymap3d as pm
from typing import Tuple

from GraphPlanner.state_node import Pose2dState
from Obstacles.costmap import Costmap
from Obstacles.target import Target, TargetList

class EndpointSampler():
    """Creates random start and goal conditions to test motion planner."""
    def __init__(self, start_bounds: np.ndarray, goal_dist_range, costmap: Costmap = None, min_allowed: float = 0):
        # (x_center, y_center, bbox_height, bbox_width)
        self.start_bounds = start_bounds
        # (min dist, max dist)
        self.goal_dist_range = goal_dist_range
        self.costmap = costmap
        self.min_allowed = min_allowed

    def sample(self) -> Tuple[Pose2dState, Pose2dState]:
        while True:
            # 1. sample a start within bounding box
            x_center, y_center, bbox_h, bbox_w = self.start_bounds
            start_x = np.random.uniform(x_center - bbox_w / 2, x_center + bbox_w / 2)
            start_y = np.random.uniform(y_center - bbox_h / 2, y_center + bbox_h / 2)
            start = np.array([start_x, start_y])

            # 2. sample goal within goal_dist_range away
            dist = np.random.uniform(*self.goal_dist_range)
            vel = np.random.normal(size=2)
            vel /= np.linalg.norm(vel)
            goal = start + dist * vel

            # Make sure goal is within costmap bounds
            if self.costmap is not None:
                valid_x = -self.costmap.width_m/2 <= goal[0] and goal[0] <= self.costmap.width_m/2 
                valid_y = -self.costmap.height_m/2 <= goal[1] and goal[1] <= self.costmap.height_m/2 
                if not valid_x or not valid_y:
                    continue

            # 3. Add a random start and goal yaw within 0, 2pi
            yaws = np.random.uniform(0, 2 * np.pi, size=2)
            start = np.append(start, yaws[0])
            goal = np.append(goal, yaws[1])

            # should have poses[0,:] = start, poses[1,:] = goal
            poses = np.stack([start, goal], axis=0)

            # Make sure that both endpoints >= self.min_allowed
            # from any occupied region
            if self.costmap is not None:
                # Times is unused because static
                dists = self.costmap.find_distances(poses, np.array([0,0]))
                if np.all(dists >= self.min_allowed):
                    break

        start = Pose2dState(*start)
        goal = Pose2dState(*goal)

        return start, goal

class WindSampler():
    """Creates random wind conditions to test motion planner."""
    def __init__(self, wind_speed_range, wind_yaw_range):
        self.wind_speed_range = wind_speed_range
        self.wind_yaw_range = wind_yaw_range

    def sample(self):
        wind_speed = np.random.uniform(self.wind_speed_range[0], self.wind_speed_range[1])
        wind_yaw = np.mod(np.random.uniform(self.wind_yaw_range[0], self.wind_yaw_range[1]), 2 * np.pi)
        return wind_yaw, wind_speed

class TargetListSampler():
    """Create random target setups to test motion planner."""
    def __init__(self, n_targets_range: np.ndarray, speed_range: np.ndarray, 
                 goal_range: np.ndarray, offset_range: np.ndarray, size_range: np.ndarray,
                 n_disc: float = 10, min_allowed: float = 0) -> None:
        """Inputs:
        n_targets_range: [lower, upper] bounds for number of targets to generate
        speed_range: [lower, upper] bounds for speed of targets
        goal_range: [lower, upper] bounds for how far away to place target vessel's desired goal
        offset_range: [lower, upper] bounds for how far to start targets away from center
        size_range: [lower, upper] bounds for the bbox height/width
        n_disc: how many discretizations in edge between target start and goal to check if movement valid
        min_allowed: how close to static obstacle can generate
        """
        self.n_targets_range = n_targets_range
        self.speed_range = speed_range
        self.goal_range = goal_range
        self.offset_range = offset_range
        self.size_range = size_range
        self.n_disc = n_disc
        self.min_allowed = min_allowed

    def create_target_list(self, costmap: Costmap, center: np.ndarray, id_cmap_name=None) -> TargetList:
        """Makes one TargetList object """
        n_targets = np.random.randint(self.n_targets_range[0], self.n_targets_range[1]+1)
        target_list = []

        n_succ = 0
        while n_succ < n_targets:
            # 1. Generate a location and nearby goal for this target vessel
            
            # Generate offsets for location and target's goal uniformly on sphere
            offsets = np.random.normal(size=(2,2))

            # Generate a distance from center
            dist = np.random.uniform(self.offset_range[0], self.offset_range[1], size=1)
            # Generate a distance for target's goal relative to its start
            goal_dist = np.random.uniform(self.goal_range[0], self.goal_range[1], size=1)

            offsets /= np.linalg.norm(offsets, axis=1)[:,None]
            offsets[0,:] *= dist
            offsets[1,:] *= goal_dist
            # Sample target position by perturbing from center position
            target_pos = center[:2] + offsets[0,:]
            # Sample target's goal by perturbing from target's starting position
            target_goal_pos = target_pos + offsets[1,:]
            positions = np.stack([target_pos, target_goal_pos], axis=0) # (2,2)

            # Reject if either outside costmap bounds
            inside = (positions[:,0] >= -costmap.width_m/2) * (positions[:,0] <= costmap.width_m/2) * \
                (positions[:,1] >= -costmap.height_m/2) * (positions[:,1] <= costmap.height_m/2)
            if not np.all(inside):
                continue

            # Reject if can't connect with straight line without hitting static obstacle
            alphas = np.linspace(0, 1, self.n_disc)[:,None] # (n_disc,1)
            disc_positions = (1 - alphas) * positions[0,:] + alphas * positions[1,:] # (n_disc,2)
            disc_distances = costmap.find_distances(disc_positions, np.array([costmap.time])).squeeze()
            if np.any(disc_distances <= self.min_allowed):
                continue

            # 2. Generate a speed for this target vessel
            speed = np.random.uniform(self.speed_range[0], self.speed_range[1], size=1)
            # Align direction with vector from start to target's goal
            direction = positions[1,:] - positions[0,:]
            direction /= np.linalg.norm(direction)
            vel = speed * direction

            # 3. Generate a size for this target vessel 
            bbox = np.random.uniform(self.size_range[0], self.size_range[1], size=2)

            # 4. Convert position to latlon
            # Both center and positions are wrt. costmap center
            # Convert from en to LLA
            lat0, lon0 = costmap.latlon_img_center
            lla = pm.enu2geodetic(target_pos[0], target_pos[1],
                                0, lat0, lon0, 0)
            latlon = np.array([lla[0], lla[1]])

            target = Target(costmap.time, n_succ, True, latlon, vel, bbox)
            target.update_ref_latlon(costmap.ref_latlon)

            target_list.append(target)
            n_succ += 1

        target_list = TargetList(target_list, id_cmap_name)

        return target_list

class TrialSampler():
    """Samples a full trial conditions (start, goal, costmap, target_list, wind_yaw, wind_speed)."""
    def __init__(self, endpoint_sampler: EndpointSampler, costmap: Costmap, 
                 target_sampler: TargetListSampler, wind_sampler: WindSampler):
        self.endpoint_sampler = endpoint_sampler
        self.costmap = costmap
        self.target_sampler = target_sampler
        self.wind_sampler = wind_sampler

    def sample(self):
        start, goal = self.endpoint_sampler.sample()
        center = start.as_array()
        target_list = self.target_sampler.create_target_list(self.costmap, center)
        wind_yaw, wind_speed = self.wind_sampler.sample()
        return start, goal, self.costmap, target_list, wind_yaw, wind_speed
