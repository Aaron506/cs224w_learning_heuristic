import numpy as np
import matplotlib.pyplot as plt

import Plotting.plotting_helpers as ph
from Obstacles.costmap import Costmap, map_to_pixel_units

def start_animation(costmap, target_list=None, times=None, goal=None, success_Q=None, pause=0.01):
    """Setup initial animation of graph solve."""
    ax = costmap.visualize()

    if target_list is not None:
        if times is None:
            raise ValueError('Provide times to visualize target_list')
        target_list.visualize(costmap, times, ax)

    if goal is not None:
        pix_goal = np.array([goal.x, goal.y, goal.psi])
        pix_goal = map_to_pixel_units(pix_goal[None,:], costmap).squeeze()
        ph.draw_triangle(ax, pix_goal[0], pix_goal[1], pix_goal[2], 
            length=20, width=10, color='blue')
        if success_Q is not None:
            M = success_Q[:2,:2].copy()
            M[:2,:2] *= costmap.m_per_pix**2
            ph.draw_ellipsoid(ax, pix_goal[0], pix_goal[1], M, color='blue', alpha=0.5)

    plt.ion()
    plt.draw()
    plt.pause(pause)

    return ax

def expansion_animation(curr_node, new_poses, costmap, 
                        start_time=-1, abs_max_time=-1, goal=None, pause=0.01, ax=None):
    """Animate the process of expanding a node into neighbors."""
    if ax is None:
        start_animation(costmap, goal, pause)
    
    # Change the color of current node once processed
    if start_time >= 0 and abs_max_time > start_time:
        heat_color = ph.time_to_color(curr_node.time, start_time, abs_max_time)
    else:
        heat_color = 'black'

    # Show start node in green
    if curr_node.parent is None:
        heat_color = 'green'

    curr_pose = np.array([curr_node.x, curr_node.y, curr_node.psi])
    pix_curr_pose = map_to_pixel_units(curr_pose[None,:], costmap).squeeze()
    ph.draw_triangle(ax, pix_curr_pose[0], pix_curr_pose[1], pix_curr_pose[2],
                     length=20, width=10, color=heat_color, alpha=1)
    
    # Add children
    for i, pose in enumerate(new_poses):
        pix_pose = pose.copy()
        pix_pose = map_to_pixel_units(pix_pose[None,:], costmap).squeeze()
        ph.draw_triangle(ax, pix_pose[0], pix_pose[1], pix_pose[2],
                         length=20, width=10, color='white', alpha=0.1)
    
        # Show dashed line from parent to each child
        ax.plot([pix_curr_pose[0], pix_pose[0]], [pix_curr_pose[1], pix_pose[1]],
                linestyle='--', color='gray', linewidth=1, alpha=0.3)

    plt.draw()
    plt.pause(pause)

    return ax
