import numpy as np
import matplotlib.pyplot as plt
import pymap3d as pm
from scipy.ndimage import distance_transform_edt
from Obstacles.obstacle_helpers import ObstacleInterface

def map_to_pixel_units(positions: np.ndarray, costmap: 'Costmap'):
    pixel_positions = positions.copy().astype('float')
    pixel_positions[:,:2] /= costmap.m_per_pix
    pixel_positions[:,0] += costmap.image.shape[1]/2
    pixel_positions[:,1] += costmap.image.shape[0]/2
    return pixel_positions

def map_from_pixel_units(pixel_positions: np.ndarray, costmap: 'Costmap'):
    positions = pixel_positions.copy().astype('float')
    positions[:,0] -= costmap.image.shape[1]/2
    positions[:,1] -= costmap.image.shape[0]/2
    positions[:,:2] *= costmap.m_per_pix
    return positions

class Costmap(ObstacleInterface):
    """A wrapper class for an obstacle costmap."""
    def __init__(self, image: np.ndarray, time: float, latlon_img_center: np.ndarray, m_per_pix: float, threshold: float=-1) -> None:
        """Inputs:
        image: (H,W) np.uint8 monochrome image/heatmap of obstacles
        time: associated timestamp in seconds,
        latlon_img_center: (lat, lon) pair of latitude, longitude coordinates (deg) of image center,
        m_per_pix: the resolution meters/pixel"""
        self.image = image
        self.time = time
        self.latlon_img_center = latlon_img_center
        self.m_per_pix = m_per_pix

        # Define an ENU frame positioned at bottom left of image
        # Compute the associated ENU bounding box info associated with image
        # lower left corner = (x=0,y=0), upper right corner = (x=width, y=height)
        self.height_m = self.m_per_pix * self.image.shape[0] # m
        self.width_m = self.m_per_pix * self.image.shape[1] # m

        self.ref_latlon = self.latlon_img_center
        
        if threshold >= 0:
            self.binarize(threshold)

    def binarize(self, threshold: float) -> None:
        """Form a binary occupancy map by thresholding the costmap image.
        Inputs:
        threshold: pixel value >= threshold -> 1, pixel value < threshold -> 0
        so like 1 = occupied. Since image is uint8, threshold should be within 0 to 255."""
        self.threshold = threshold
        self.occ_map = (self.image > self.threshold)

        # Also, store associated distance map
        # Compute distance transform (in pixels), then scale to meters
        # Note: distance will be zero inside occupied cells

        if not np.any(self.occ_map):
            # No occupied cells at all so assign infinite distances
            self.dist_map = np.full_like(self.image, np.inf, dtype=float)
        else:
            # Compute normal distance transform
            self.dist_map = distance_transform_edt(~self.occ_map) * self.m_per_pix

    def find_distances(self, poses: np.ndarray, times: np.ndarray) -> np.ndarray:
        """For each pose, find the Euclidean distance to the nearest occupied cell.
        Inputs:
        poses: (n,3) where pose = (x,y,psi)
        times: (n,) but unused because costmap static
        Outputs:
        distances: (n,) > 0 if pose outside nearest occupied cell, 0 if inside
        """
        if not hasattr(self, 'dist_map'):
            raise ValueError("Call binarize() first to generate dist_map.")
        # Each pixel corresponds to m_per_pix x m_per_pix region
        # Convert poses from (x, y, psi) in meters to pixel indices
        # Note: (0,0) is at center image in ENU
        # Numpy image has (0,0) at bottom-left
        # So, convert to pixel scale and then shift by half image dimensions
        pixel_y = poses[:, 1] / self.m_per_pix + self.image.shape[0]/2
        pixel_x = poses[:, 0] / self.m_per_pix + self.image.shape[1]/2

        # Round to nearest integer pixel and clip to image bounds
        pixel_y = np.clip(np.round(pixel_y).astype(int), 0, self.image.shape[0] - 1)
        pixel_x = np.clip(np.round(pixel_x).astype(int), 0, self.image.shape[1] - 1)

        # Look up distance values
        distances = self.dist_map[pixel_y, pixel_x]

        return distances

    def visualize(self, ax=None, cmap=None, title='Radar Image', num_ticks=6, flag='costmap', add_colorbar=True):
        if flag == 'occupancy':
            disp_image = self.occ_map
            label = 'Occupancy'
        elif flag == 'distance':
            disp_image = self.dist_map
            label = 'Distance (m)'
        elif flag == 'costmap':
            disp_image = self.image
            label = 'Intensity'
        else:
            raise ValueError('flag should be either occupancy, distance, or costmap')

        if cmap is None:
            cmap = plt.cm.viridis.copy()
            cmap.set_bad('white')

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        h = ax.imshow(disp_image, cmap=cmap, origin='lower')
        ax.set_title(title)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(0, self.image.shape[1])
        ax.set_ylim(0, self.image.shape[0])

        # Pixel axis limits
        xlim = (0, self.image.shape[1])
        ylim = (0, self.image.shape[0])
        # Convert pixel range to meter range
        xtick_pix = np.linspace(*xlim, num=num_ticks)
        ytick_pix = np.linspace(*ylim, num=num_ticks)

        # Display with real-world coordinates relative to image center
        center_x_pix = self.image.shape[1] / 2
        center_y_pix = self.image.shape[0] / 2
        xtick_labels = [f"{(x - center_x_pix) * self.m_per_pix:.0f}" for x in xtick_pix]
        ytick_labels = [f"{(y - center_y_pix) * self.m_per_pix:.0f}" for y in ytick_pix]

        ax.set_xticks(xtick_pix)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(ytick_pix)
        ax.set_yticklabels(ytick_labels)

        if add_colorbar:
            fig.colorbar(h, ax=ax, label=label)

        return ax
    
def empty_costmap(image_shape = (1000, 1000), time = 0.0, 
                  latlon_image_center = np.array([37.52752887990091, -122.19717998304971]),
                  m_per_pix = 2.8) -> Costmap:
    """Initialize an empty costmap, which is useful for defining
    a global time, (lat, lon) reference, and m_per_pix resolution.
    Current default values are taken from a previously collected rosbag.
    """
    # image_shape is (H, W)
    image = np.zeros(image_shape, dtype=np.uint8)
    costmap = Costmap(image, time, latlon_image_center, m_per_pix)
    costmap.binarize(threshold=255.0) # All unoccupied
    return costmap