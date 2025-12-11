import numpy as np
import os
import matplotlib.pyplot as plt

from Plotting.heuristic_plotting import heuristic_heatmap, find_cost_to_go_arr
from Obstacles.costmap import empty_costmap
import Configs.load_from_config as lc
from GraphPlanner.state_node import Pose2dState

if __name__ == '__main__':
    # seed = 0
    # np.random.seed(seed)

    #### Load learned heuristic ####
    timestamp = '20251208_225420'
    foldername = f'Data/{timestamp}'
    layer_type = 'gcn'
    heuristic_model_path = os.path.join(foldername, f'model_{layer_type}')
    heuristic = lc.load_learned_heuristic(heuristic_model_path)

    #### Set costmap ####
    # Empty for now, but can later use real-world radar images
    costmap = empty_costmap()
    time = costmap.time

    #### Set targets ####
    scenario_config_path = 'Scripts/Configs/scenario_config.yaml'
    target_sampler = lc.load_target_sampler(scenario_config_path)
    # Generate random target_list
    # Can also set target_list = None for no dynamic obstacles
    # The starting state for ego-vessel
    goal = Pose2dState(0, 0, 0)
    pose_psi = goal.psi
    target_list = target_sampler.create_target_list(costmap, goal.as_array())
    target_list.update_ref_latlon(costmap.ref_latlon)

    #### Plot parameters ####
    planner_config_path = 'Scripts/Configs/planner_config.yaml'
    planner_cfg = lc.load_config(planner_config_path)
    # h, w = np.array(planner_cfg['planning_bounds'])/2
    h, w = 1000, 1000
    bounds = np.array([[-w,w],[-h,h]])
    num_disc = 100
    # Use to ensure that all plots have same colorbar range
    vmin = 0
    wind_yaw = np.pi/4
    wind_speed = 18
    _, _, cost_to_go_arr, _ = find_cost_to_go_arr(time, pose_psi, wind_yaw, wind_speed, target_list, 
                                                    bounds, num_disc, heuristic, goal, None)
    vmax = 1.05 * np.max(cost_to_go_arr) # Add a bit

    vis_times = np.array([time + planner_cfg['step_time'] * i for i in range(3)])

    #### Heatmap as change wind speed ####
    wind_yaw = np.pi/4
    wind_speeds = [10,25]
    for wind_speed in wind_speeds:
        ax = heuristic_heatmap(time, pose_psi, wind_yaw, wind_speed, target_list, bounds, num_disc, heuristic, goal, None,
                               vmin, vmax, vis_times)
        ax.set_title(r'$v_w$ = ' + f'{wind_speed}')

    #### Heatmap as change wind angle ####
    wind_yaws = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
    wind_speed = 10
    for wind_yaw in wind_yaws:
        ax = heuristic_heatmap(time, pose_psi, wind_yaw, wind_speed, target_list, bounds, num_disc, heuristic, goal, None,
                               vmin, vmax, vis_times)
        ax.set_title(r'$\psi_w$ (deg) = ' + f'{np.rad2deg(wind_yaw)}')

    #### Decompose several instances into g0, g1 as progressively add targets ####
    wind_yaw = np.pi/4
    wind_speed = 18
    num_reps = 3
    for i in range(num_reps):
        fig, (ax_g0, ax_g1, ax_tot) = plt.subplots(1,3,figsize=(15,5))
        target_list = target_sampler.create_target_list(costmap, goal.as_array())
        target_list.update_ref_latlon(costmap.ref_latlon)

        ax_g0 = heuristic_heatmap(time, pose_psi, wind_yaw, wind_speed, None, bounds, num_disc, heuristic, goal, None,
                                None, None, vis_times, ax=ax_g0, merge_type='g0')
        ax_g0.set_title(r'$g_0$')
        ax_g1 = heuristic_heatmap(time, pose_psi, wind_yaw, wind_speed, target_list, bounds, num_disc, heuristic, goal, None,
                                None, None, vis_times, ax=ax_g1, merge_type='g1')
        ax_g1.set_title(r'$g_1$')
        ax_tot = heuristic_heatmap(time, pose_psi, wind_yaw, wind_speed, target_list, bounds, num_disc, heuristic, goal, None,
                                None, None, vis_times, ax=ax_tot, merge_type='total')
        ax_tot.set_title(r'$g_0 + g_1$')
        fig.suptitle(f'Repetition {i}')

    plt.show()
