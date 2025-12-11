import numpy as np
import pymap3d as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import Math.geometry as geom
from Obstacles.obstacle_helpers import ObstacleInterface
from Obstacles.costmap import Costmap, map_to_pixel_units
from Plotting.plotting_helpers import time_to_color, draw_rectangle

class Target(ObstacleInterface):
    """Class to store target information for avoidance."""

    def __init__(self, time, track_id, is_tracked, latlon: np.ndarray, vel: np.ndarray, bbox: np.ndarray, id_color=None) -> None:
        """Inputs:
        time: time in seconds for this object track received
        track_id: id for this tracked dynamic object
        is_tracked: boolean which is True if this object is tracked
        latlon: (latitude, longitude) pair 
        vel: (vel_e, vel_n) velocity in (m/s)
        bbox: (bbox_height, bbox_width) in (m)
        id_color: RGB color to assign as unique visual label for this target
        """
        self.time = time
        self.track_id = track_id
        self.is_tracked = is_tracked
        self.latlon = latlon
        self.vel = vel
        self.bbox = bbox
        self.id_color = id_color

    def update_ref_latlon(self, ref_latlon: np.ndarray):
        """Use given latlon to define this track's coordinates."""
        # 1. Find ENU wrt. costmap latlon reference
        self.ref_latlon = ref_latlon
        lat0, lon0 = self.ref_latlon

        # Assume both altitudes are 0 since not currently provided
        # Returns a height for ENU, but ignore
        lat, lon = self.latlon
        pos_en = pm.geodetic2enu(lat, lon, 0.0, lat0, lon0, 0.0)[:2]
        self.pos_en = np.array(pos_en)

    def get_coords_over_time(self, times: np.ndarray) -> np.ndarray:
        """Find EN coordinates at specified times using straight line propagation."""
         # Predict position at time t assuming constant velocity
        dt = times - self.time # (n,)
        offsets = np.tile(self.vel, (dt.shape[0],1)) # (n,2)
        positions = self.pos_en + offsets * dt[:,None] # (n,2)
        return positions

    def find_distances(self, poses: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Find distance between this object and given poses achieved at specified times.
        Inputs:
        poses: (n,3) array of poses to compare to this object,
        times: (n,) associated times when these poses are reached
        """
        # 1. Find the location of this target at those times
        target_positions = self.get_coords_over_time(times) # (n,2)

        # 2. Form associated bounding box info (x,y,height,width) -> (n,4)
        bboxs_tiled = np.broadcast_to(self.bbox[None, :], (target_positions.shape[0],2))
        rectangles = np.concatenate([target_positions, bboxs_tiled], axis=-1) # (n, 4)

        # 3. Using the bbox centered about target_positions[i], find the distance to poses[i]
        distances = geom.points_to_paired_rectangles_distance(poses[:,:2], rectangles[:,None,:]).squeeze()

        return distances

    def visualize(self, costmap, times, ax=None, start_time=-1, abs_max_time=-1, cmap_name='plasma', 
                  use_id_color=False, alpha=1, show_id=True, yaw=0):
        """Visualize the location of the object at designated times."""
        if ax is None:
            _, ax = plt.subplots()
        
        # Color time using heatmap
        min_time = start_time if start_time >= 0 else times[0]
        max_time = abs_max_time if abs_max_time > 0 else times[-1]
        colors = time_to_color(times, min_time, max_time, cmap_name=cmap_name)

        # Get positions and map to pixel resolution
        positions = self.get_coords_over_time(times)
        if costmap is not None:
            positions = map_to_pixel_units(positions, costmap)
            height = self.bbox[0] / costmap.m_per_pix
            width = self.bbox[1] / costmap.m_per_pix
        else:
            height, width = self.bbox[0], self.bbox[1]

        # Plot colored bbox at each time
        for i, color in enumerate(colors):

            draw_rectangle(ax, positions[i,0], positions[i,1], height, width,
                       yaw=yaw, color=color, zorder=5, alpha=alpha)

        # Plot a text box next to initial time with the id
        if show_id:
            ax.text(positions[0,0] + width / 2 + 5,  # x offset slightly to the right
                positions[0,1],                 # y stays the same
                str(self.track_id),            # convert ID to string
                fontsize=8,
                color='white',
                zorder=10,
                clip_on=True)

        # Plot the initial time with the id color
        if use_id_color:
            if self.id_color is None:
                raise ValueError("Please set id_color when use_id_color is True")
            draw_rectangle(ax, positions[0,0], positions[0,1], height, width,
                       yaw=yaw, color=self.id_color, zorder=5, alpha=alpha)

        return ax

    def visualize_distance(self, costmap, time, ax=None, resolution=100, cmap=None, num_ticks=6):
        """Visualize the distance to given object at specified times as an image."""
        # 1. Get discretization points using costmap bounds
        x_vals = np.linspace(-costmap.width_m / 2, costmap.width_m / 2, resolution)
        y_vals = np.linspace(-costmap.height_m / 2, costmap.height_m / 2, resolution)

        # 2. Create dense grid of poses
        X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
        poses = np.stack([X_mesh, Y_mesh], axis=-1).reshape(-1, 2)  # (resolution^2, 2)

        # 3. Call find_distances for each pose, all at same time
        times = time * np.ones(poses.shape[0])
        distances = self.find_distances(poses, times)

        # 4. Reshape to match X_mesh shape
        # dist_mesh[i,j] stores distance associated with y_vals[i], x_vals[j]
        # So, when visualizing with imshow do not need to transpose
        dist_mesh = distances.reshape(X_mesh.shape)  # (resolution, resolution)

        # 5. Plot as heatmap
        if cmap is None:
            cmap = plt.cm.viridis.copy()
            cmap.set_bad('white')

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Pretend like it is costmap shaped
        h = ax.imshow(dist_mesh, cmap=cmap, origin='lower',
              extent=[0, costmap.image.shape[1], 0, costmap.image.shape[0]])
        ax.set_title(f'Distance to Targets at t={time:.2f} [s]')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(0, costmap.image.shape[1])
        ax.set_ylim(0, costmap.image.shape[0])

        # Pixel axis limits
        xlim = (0, costmap.image.shape[1])
        ylim = (0, costmap.image.shape[0])
        # Convert pixel range to meter range
        xtick_pix = np.linspace(*xlim, num=num_ticks)
        ytick_pix = np.linspace(*ylim, num=num_ticks)
        
        # Display with real-world coordinates relative to image center
        center_x_pix = self.image.shape[1] / 2
        center_y_pix = self.image.shape[0] / 2
        xtick_labels = [f"{(x - center_x_pix) * costmap.m_per_pix:.0f}" for x in xtick_pix]
        ytick_labels = [f"{(y - center_y_pix) * costmap.m_per_pix:.0f}" for y in ytick_pix]

        ax.set_xticks(xtick_pix)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(ytick_pix)
        ax.set_yticklabels(ytick_labels)

        fig.colorbar(h, ax=ax, label='Distance')

        # 5. Overlay the bbox
        self.visualize(costmap, np.array([time]), ax, show_id=True)

        return ax

class TargetList(Target):
    """A wrapper class for efficient operation with a list of multiple Target objects."""
    def __init__(self, target_list, id_cmap_name=None):
        self.target_list = target_list
        self.num_targets = len(self.target_list)

        # Package various quantities as arrays for efficient operation
        self.times = np.array([target.time for target in self.target_list])
        self.ids = [target.track_id for target in self.target_list]
        self.is_tracked = np.array([target.is_tracked for target in self.target_list])
        self.latlons = np.array([target.latlon for target in self.target_list])
        self.vels = np.array([target.vel for target in self.target_list])
        self.bboxs = np.array([target.bbox for target in self.target_list])

        # Initialize id_color for each target if id_cmap_name provided
        if id_cmap_name is not None:
            cmap = cm.get_cmap(id_cmap_name)
            norm = mcolors.Normalize(vmin=0, vmax=self.num_targets - 1)
            colors = [cmap(norm(i)) for i in range(self.num_targets)]
            # Choose colors uniformly over the cmap
            for i, target in enumerate(self.target_list):
                target.id_color = colors[i]        

    def update_ref_latlon(self, ref_latlon):
        self.ref_latlon = ref_latlon
        for target in self.target_list:
            target.update_ref_latlon(ref_latlon)

        # (m,2) where m = number of targets
        self.positions = np.array([target.pos_en for target in self.target_list])

    def get_coords_over_time(self, times: np.ndarray) -> np.ndarray:
        """Find EN coordinates for all targets at specified times using straight line propagation."""
        # Predict position at time t assuming constant velocity
        
        # times is shape (n,)
        # self.times is shape (m,)
        # dt should store time offset for i'th object when queried at j'th time
        # i.e., dt[i,j] = times[j] - self.times[i], dt is (m,n)
        dt = times[None, :] - self.times[:, None]
        m,n = dt.shape

        # self.vels is (m,2) where m = number of targets
        # offsets stores the predicted position offsets for each target and each time
        # offsets[i,j] = 2D offset for i'th target at j'th query time
        # i.e., offsets[i,j,:] = self.vels[i,:] * dt[i,j]
        offsets = self.vels[:, None, :] * dt[:, :, None]

        # Now, add to the original position
        # (m,n,2) where pred_positions[i,j,:] = 2D predicted position for i'th object at j'th time
        pred_positions = self.positions[:, None, :] + offsets

        # Actually for consistency with rest of code, make (n,m,2)
        pred_positions = pred_positions.transpose((1,0,2))

        return pred_positions    

    def find_distances(self, poses: np.ndarray, times: np.ndarray, extra_info=False) -> np.ndarray:
        """Find distance between each object and given poses achieved at specified times.
        Inputs:
        poses: (n,3) array of poses to compare to each of m objects,
        times: (n,) associated times when these poses are reached
        extra_info: bool, True if also want to return distances matrix and nearest_ind
        Outputs:
        nearest_distances: (n,) array of distance to nearest object at given times
        nearest_ind: (n,) index for associated neartest object
        distances: (n,m) array of distance of each pose to each object at given times
        i.e., distances[i,j] = pose[i] to object[j] at times[i]
        """
        # 1. Find the location of each target at those times
        target_positions = self.get_coords_over_time(times) # (n,m,2)
        n,m,_ = target_positions.shape

        # 2. Form associated bounding box info (x,y,height,width) -> (n,m,4)
        # self.bboxs shape (m,2), want to tile same bbox over time dimension n 
        bboxs_tiled = np.broadcast_to(self.bboxs[None, :, :], (n, m, 2))
        rectangles = np.concatenate([target_positions, bboxs_tiled], axis=-1) # (n, m, 4)

        # 3. Find the distances (n,m)
        distances = geom.points_to_paired_rectangles_distance(poses[:,:2], rectangles)

        # 4. Find nearest object information (n,)
        nearest_ind = np.argmin(distances, axis=1)
        # Extract distances[i,nearest_ind[i]]
        nearest_distances = distances[np.arange(len(nearest_ind)), nearest_ind]

        if extra_info:
            return nearest_distances, nearest_ind, distances
        else:
            return nearest_distances

    def visualize(self, costmap, times, ax=None,  start_time=-1, abs_max_time=-1, cmap_name='plasma', 
                  use_id_color=False, alpha=1, show_id=True, yaw=0):
        """Visualize the location of each object at designated times."""
        if ax is None:
            if costmap is not None:
                ax = costmap.visualize()
            else:
                _, ax = plt.subplots()

        for target in self.target_list:
            target.visualize(costmap, times, ax=ax, start_time=start_time, 
                             abs_max_time=abs_max_time, cmap_name=cmap_name, 
                             use_id_color=use_id_color, alpha=alpha, show_id=show_id, yaw=yaw)

        return ax

    def visualize_distance(self, costmap, time, ax=None, resolution=100, cmap=None, num_ticks=6):
        """Visualize the distance to nearest object at specified times as an image."""
        # 1. Get discretization points using costmap bounds
        x_vals = np.linspace(-costmap.width_m/2, costmap.width_m/2, resolution)
        y_vals = np.linspace(-costmap.height_m/2, costmap.height_m/2, resolution)

        # 2. Create dense grid of poses
        X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
        poses = np.stack([X_mesh, Y_mesh], axis=-1).reshape(-1, 2)  # (resolution^2, 2)

        # 3. Call find_distances for each pose, all at same time
        times = time * np.ones(poses.shape[0])
        distances = self.find_distances(poses, times)

        # 4. Reshape to match X_mesh shape
        # dist_mesh[i,j] stores distance associated with y_vals[i], x_vals[j]
        # So, when visualizing with imshow do not need to transpose
        dist_mesh = distances.reshape(X_mesh.shape)  # (resolution, resolution)

        # 5. Plot as heatmap
        if cmap is None:
            cmap = plt.cm.viridis.copy()
            cmap.set_bad('white')

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Pretend like it is costmap shaped
        h = ax.imshow(dist_mesh, cmap=cmap, origin='lower',
                      extent=[0, costmap.image.shape[1], 0, costmap.image.shape[0]])
        ax.set_title(f'Distance to Object at t={time:.2f} [s]')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(0, costmap.image.shape[1])
        ax.set_ylim(0, costmap.image.shape[0])

        # Pixel axis limits
        xlim = (0, costmap.image.shape[1])
        ylim = (0, costmap.image.shape[0])
        # Convert pixel range to meter range
        xtick_pix = np.linspace(*xlim, num=num_ticks)
        ytick_pix = np.linspace(*ylim, num=num_ticks)

        # Display with real-world coordinates relative to image center
        center_x_pix = costmap.image.shape[1] / 2
        center_y_pix = costmap.image.shape[0] / 2
        xtick_labels = [f"{(x - center_x_pix) * costmap.m_per_pix:.0f}" for x in xtick_pix]
        ytick_labels = [f"{(y - center_y_pix) * costmap.m_per_pix:.0f}" for y in ytick_pix]

        ax.set_xticks(xtick_pix)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(ytick_pix)
        ax.set_yticklabels(ytick_labels)

        fig.colorbar(h, ax=ax, label='Distance')

        # 5. Overlay the bbox
        self.visualize(costmap, np.array([time]), ax, show_id=True)

        return ax