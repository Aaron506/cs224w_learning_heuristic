import numpy as np
import matplotlib.pyplot as plt
from Obstacles.costmap import empty_costmap, map_to_pixel_units
from Plotting.plotting_helpers import draw_triangle
import Configs.load_from_config as lc

if __name__ == '__main__':
    #### Set costmap ####
    costmap = empty_costmap()

    #### Set trial sampler ####
    scenario_config_path = 'Scripts/Configs/scenario_config.yaml'
    trial_sampler = lc.load_trial_sampler(scenario_config_path, costmap)

    #### Sample a trial ####
    start, goal, _, target_list, wind_yaw, wind_speed = trial_sampler.sample()

    #### Plotting params ####
    max_rel_time = 150 # s
    num_times = 10
    vis_times = np.linspace(costmap.time, costmap.time + max_rel_time, num_times)

    #### Visualize scenario ####
    # Visualize costmap
    ax = costmap.visualize(flag='occupancy')
    # Visualize target
    target_list.visualize(costmap, vis_times, ax)
    # Visualize start
    start_pix = map_to_pixel_units(start.as_array()[None,:], costmap).squeeze()
    draw_triangle(ax, start_pix[0], start_pix[1], start_pix[2], length=30, width=20, color='green')
    # Visualize goal
    goal_pix = map_to_pixel_units(goal.as_array()[None,:], costmap).squeeze()
    draw_triangle(ax, goal_pix[0], goal_pix[1], goal_pix[2], length=30, width=20, color='blue')

    # Overlay wind
    height, width = costmap.height_m / costmap.m_per_pix,  costmap.width_m / costmap.m_per_pix
    arrow_length = 0.1 * width
    # Wind blows from wind_yaw, so the arrow points *against* wind_yaw
    dx = -arrow_length * np.cos(wind_yaw)
    dy = -arrow_length * np.sin(wind_yaw)
    x0 = int(0.85 * width)
    y0 = int(0.85 * height)
    ax.arrow(x0, y0, dx, dy, width=2, head_width=20, head_length=15, 
             length_includes_head=True, color='cyan', zorder=15, label='Wind')
    ax.set_aspect('equal')

    plt.show()