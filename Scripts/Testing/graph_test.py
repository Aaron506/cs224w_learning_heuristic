import numpy as np
import os
import matplotlib.pyplot as plt
import time

from GraphPlanner.state_node import Pose2dState, form_success_Q
from GraphPlanner.sailboat_polar import SailboatPolar
from GraphPlanner.graph_planner import GraphPlanner, count_tacks
from GraphPlanner.pruner import Pruner
from GraphPlanner.heuristic import MaxSpeedHeuristic
from Obstacles.obstacle_helpers import ObstacleCostCalculator
from Obstacles.costmap import empty_costmap
from DatasetCreation.scenario_creator import TargetListSampler
import Plotting.waypoint_plotting as wp

import Configs.load_from_config as lc

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    #### Extract polar ####
    polar_file = 'Data/polar.csv'
    polar = SailboatPolar(polar_file)

    #### Set costmap ####
    # Empty for now, but can later use real-world radar images
    costmap = empty_costmap()

    #### Set targets ####
    scenario_config_path = 'Scripts/Configs/scenario_config.yaml'
    target_sampler = lc.load_target_sampler(scenario_config_path)
    # Generate random target_list
    # Can also set target_list = None for no dynamic obstacles
    # The starting state for ego-vessel
    ego_pose = np.array([0, 0, 0])
    target_list = target_sampler.create_target_list(costmap, ego_pose)

    #### Build graph planner ####
    planner_config_path = 'Scripts/Configs/planner_config.yaml'

    timestamp = '20251208_225420'
    foldername = f'Data/{timestamp}'
    heuristic_model_path = os.path.join(foldername, f'model_gcn')
    # Can set heuristic_model_path to None to use original heuristic
    heuristic_model_path = None

    planner = lc.load_graph_planner(planner_config_path, polar_file, heuristic_model_path)

    #### Provide an example problem ####
    wind_yaw = 5 * np.pi/4 # rad
    wind_speed = 20 # m/s
    planner.set_env_conditions(wind_yaw, wind_speed, costmap, target_list)

    start_time = costmap.time # s
    start = Pose2dState(ego_pose[0], ego_pose[1], ego_pose[2])
    goal = Pose2dState(500, 1000, 0)
    sigmas = lc.load_sigmas(planner_config_path)
    success_Q = form_success_Q(sigmas, goal.psi)

    #### Solve ####
    animate = False # True to visualize expansion
    verbose = False
    t0 = time.time()
    waypoints, times, costs, num_nodes, flag = planner.solve(start_time, start, goal, success_Q,
                                                  add_goal_last=True, animate=animate, pause=0.01, verbose=verbose)
    tf = time.time()
    print(f'Time to solve {tf - t0}')
    print(f'Number of tacks {count_tacks(waypoints, wind_yaw)[0]}')
    print(f'Final cost {costs[-1]}')
    print(f'# Nodes {num_nodes}')
    if not flag:
        print('Solver failed!')

    #### Visualize ####
    ax = wp.plot_waypoints(waypoints, times, costmap, start_time=times[0],
                           max_rel_time=planner.max_rel_time, goal=goal, success_Q=success_Q, wind_yaw=wind_yaw)
    if target_list is not None:
        target_list.visualize(costmap, times, ax, start_time=times[0], abs_max_time=times[0]+planner.max_rel_time)

    wp.plot_waypoint_state_info(waypoints, times, wind_yaw, debug=False)

    if target_list is not None:
        wp.plot_waypoint_dist_to_targets(waypoints, times, target_list, n_disc=10, cmap_name="tab10")

    wp.animate_waypoints(waypoints, times, costmap, target_list, goal, success_Q, wind_yaw)

    plt.show()