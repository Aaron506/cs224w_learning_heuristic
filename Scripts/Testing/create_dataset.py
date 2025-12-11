import numpy as np
import datetime
import pickle
import os
import matplotlib.pyplot as plt

from Obstacles.costmap import empty_costmap
from Obstacles.target import Target, TargetList
from GraphPlanner.sailboat_polar import SailboatPolar
import Plotting.waypoint_plotting as wp
import DatasetCreation.data_generator as dg
import Configs.load_from_config as lc
from GraphPlanner.state_node import Pose2dState, form_success_Q

def vis_datum(datum: dg.GraphDatum, ax=None, costmap = empty_costmap(), 
              time=0.0, start_time=-1, max_rel_time=-1, sigmas=None):
    times = np.array([time])

    if sigmas is not None:
        success_Q = form_success_Q(sigmas, rot_angle=0)
    else:
        success_Q = None

    # Plot the ego information
    ax = wp.plot_waypoints(np.array([datum.pose.as_array()]), times, costmap,
                           start_time=start_time, max_rel_time=max_rel_time, 
                           wind_yaw=datum.wind_yaw, ax=ax, goal=Pose2dState(0,0,0),
                           success_Q=success_Q)

    target_list = []
    for i, target_arr in enumerate(datum.targets):
        px, py, vx, vy, h, w = target_arr
        dummy_latlon = np.zeros(2) # Unused
        vel = np.array([vx,vy])
        bbox = np.array([h,w])
        target = Target(times[0], i, True, dummy_latlon, vel, bbox)
        # Directly set the pos_en
        target.pos_en = np.array([px,py])
        target_list.append(target)
    target_list = TargetList(target_list)

    if target_list is not None:
        first_time = start_time if start_time >= 0 else times[0]
        abs_max_time = first_time + max_rel_time if max_rel_time > 0 else times[-1]
        # psi became psi - goal_wrt_costmap.psi, so corresponds to an active rotation by -goal_wrt_costmap.psi
        target_list.visualize(costmap, times, ax=ax, start_time=first_time, abs_max_time=abs_max_time, 
                              yaw=-datum.goal_wrt_costmap.psi)

    ax.set_title(f'h(x) = {np.round(datum.cost_to_go,2)}')

    return ax

def vis_one_traj(dataset, traj_id, costmap = empty_costmap(), title='', max_cols=3, sigmas=None):
    """Visualize the datapoints associated with one traj_id."""
    traj_data = dg.get_one_traj_data(dataset, traj_id)

    # Compute layout
    n = len(traj_data)
    ncols = min(n, max_cols)
    nrows = int(np.ceil(n / max_cols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    fig.suptitle(title)

    # Flatten axes array for consistent indexing
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, datum in enumerate(traj_data):
        vis_datum(datum, axes[i], costmap, time=i, start_time=0, max_rel_time=len(traj_data), sigmas=sigmas)
        # Turn off axis ticks/labels
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Hide unused subplots (if total < nrows*ncols)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    return fig, axes

if __name__ == '__main__':
    #### Load trial sampler ####
    # Polar
    polar_file = 'Data/polar.csv'
    polar = SailboatPolar(polar_file)
    # Empty costmap
    costmap = empty_costmap()
    scenario_config_path = 'Scripts/Configs/scenario_config.yaml'
    trial_sampler = lc.load_trial_sampler(scenario_config_path, costmap)

    #### Load planner ####
    planner_config_path = 'Scripts/Configs/planner_config.yaml'
    planner = lc.load_graph_planner(planner_config_path, polar_file, None)

    #### Generate dataset ####
    ml_config_path = 'Scripts/Configs/ml_config.yaml'
    ml_cfg = lc.load_config(ml_config_path)
    n_trials = ml_cfg['n_trials']
    sigmas = lc.load_sigmas(planner_config_path)
    dataset, success_rate = dg.generate_dataset(n_trials, planner, sigmas, trial_sampler, 
                                             verbose=True, num_workers=-1)
    print(f'Generated dataset, success_rate = {success_rate}')

    #### Save dataset ####
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = f'Data/{timestamp}'
    os.makedirs(foldername, exist_ok=True)
    pickle.dump(dataset, open(os.path.join(foldername, 'dataset'), 'wb'))

    #### Visualize examples ####
    NUM_VIS = 3
    traj_ids = dg.get_dataset_traj_ids(dataset)[:NUM_VIS]

    for i, traj_id in enumerate(traj_ids):
        vis_one_traj(dataset, traj_id, costmap = empty_costmap(), title=f'Trajectory {i}', sigmas=sigmas)

    plt.show()