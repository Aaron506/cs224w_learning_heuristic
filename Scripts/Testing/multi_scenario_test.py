import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import List
import copy
import matplotlib
# Non-GUI so can parallelize video saving
matplotlib.use("Agg")

from GraphPlanner.sailboat_polar import SailboatPolar
from GraphPlanner.state_node import form_success_Q
from GraphPlanner.graph_planner import GraphPlanner
from Obstacles.costmap import empty_costmap
import Plotting.waypoint_plotting as wp
import DatasetCreation.scenario_creator as sc
from Obstacles.obstacle_helpers import get_dist_to_obs_over_time
import Configs.load_from_config as lc

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
_SPAWN_CTX = mp.get_context("spawn")

def _executor_factory(use_processes=True):
    """Return an Executor factory with correct multiprocessing context."""
    if use_processes:
        return (lambda **kw: ProcessPoolExecutor(mp_context=_SPAWN_CTX, **kw))
    else:
        return ThreadPoolExecutor

def run_trial(planners: List[GraphPlanner], start, goal, success_Q, costmap, 
              target_list, wind_yaw, wind_speed, n_disc: float = 10):
    """Runs common trial for multiple planners"""
    waypoints_list = []
    times_list = []
    costs_list = []
    num_nodes_list = []
    flags = []
    solve_times = []
    nearest_static = []
    nearest_dyn = []

    for i, planner in enumerate(planners):
        # 1. Set the planner environment conditions
        planner.set_env_conditions(wind_yaw, wind_speed, costmap, target_list)

        # 2. Solve with the planner
        t0 = time.time()

        waypoints, times, costs, num_nodes, flag = planner.solve(costmap.time, start, goal, success_Q,
                            add_goal_last=False, animate=False, pause=0.01, verbose=False)
        tf = time.time()

        # 3. Add the relevant info
        waypoints_list.append(waypoints)
        times_list.append(times)
        costs_list.append(costs)
        num_nodes_list.append(num_nodes)
        flags.append(flag)
        solve_times.append(tf - t0)

        # 4. Get nearest static and dynamic obstacle distances
        _, static_disc_dists = get_dist_to_obs_over_time(waypoints, times, costmap, 
                                                           n_disc=n_disc, extra_info=False)
        nearest_static.append(np.min(static_disc_dists))
        if target_list is not None:
            _, dyn_disc_dists = get_dist_to_obs_over_time(waypoints, times, target_list, 
                                                           n_disc=n_disc, extra_info=False)
            nearest_dyn.append(np.min(dyn_disc_dists))
        else:
            nearest_dyn.append(np.inf)

    return waypoints_list, times_list, costs_list, num_nodes_list, flags, solve_times, nearest_static, nearest_dyn

def _run_one_trial_worker(args):
    """Worker that runs a single trial for all planners and returns summary metrics."""
    (trial_idx, planners, sigmas, trial_sampler, output_folder,
     names, vid_time, speed_up) = args

    # Make thread-local copies of the planners to avoid shared mutable state
    local_planners = [copy.deepcopy(p) for p in planners]

    # 0. Sample the trial
    start, goal, costmap, target_list, wind_yaw, wind_speed = trial_sampler.sample()
    success_Q = form_success_Q(sigmas, rot_angle=goal.psi)

    # 1. Run the trial for all planners
    (waypoints_list, times_list, trial_costs_list, trial_num_nodes,
     trial_flags, trial_solve_times, trial_nearest_static,
     trial_nearest_dyn) = run_trial(local_planners, start, goal, success_Q,
                                    costmap, target_list, wind_yaw, wind_speed)

    # 2. Build per-trial summary arrays
    runtime_row = np.array(trial_solve_times)
    nearest_static_row = np.array(trial_nearest_static)
    nearest_dyn_row = np.array(trial_nearest_dyn)
    num_nodes_row = np.array(trial_num_nodes)
    final_costs_row = np.array([costs[-1] for costs in trial_costs_list])
    flags_row = np.array(trial_flags, dtype=bool)

    # 3. Optionally save video
    if output_folder is not None:
        folder = os.path.join(output_folder, f'trial_{trial_idx}')
        os.makedirs(folder, exist_ok=True)

        for j, waypoints in enumerate(waypoints_list):
            if names is not None:
                name = names[j]
            else:
                name = f'planner_{j}'
            video_path = os.path.join(folder, name + '.mp4')
            wp.save_waypoints_video(waypoints, times_list[j], costmap,
                                    video_path, vid_time, speed_up,
                                    target_list, goal, success_Q, wind_yaw)

    return (runtime_row, nearest_static_row, nearest_dyn_row,
            final_costs_row, num_nodes_row, flags_row)

def run_trials(n_trials: float, planners: List, sigmas: np.ndarray, 
               trial_sampler: sc.TrialSampler, output_folder: str = None, 
               names: List = None, vid_time: float = 5, speed_up: float = 100, 
               num_workers: int = 1):

    n_planners = len(planners)

    # Record runtime among success
    runtimes = np.zeros((n_trials, n_planners))
    # Record nearest got to static and dynamic obstacles
    nearest_static = np.zeros((n_trials, n_planners))
    nearest_dyn = np.zeros((n_trials, n_planners))
    # Record final trajectory cost
    final_costs = np.zeros((n_trials, n_planners))
    # Record number of nodes at termination
    num_nodes = np.zeros((n_trials, n_planners))
    # Record success/failure
    flags = np.zeros((n_trials, n_planners), dtype=bool)

    # Single-process version
    if num_workers == 1:
        for i in range(n_trials):       
            print(f"Starting trial {i}")
            arg = (i, planners, sigmas, trial_sampler,
                    output_folder, names, vid_time, speed_up)
            (runtime_row, static_row, dyn_row,
                final_cost_row, num_nodes_row, flags_row) = \
                _run_one_trial_worker(arg)
            runtimes[i, :] = runtime_row
            nearest_static[i, :] = static_row
            nearest_dyn[i, :] = dyn_row
            final_costs[i, :] = final_cost_row
            num_nodes[i, :] = num_nodes_row
            flags[i, :] = flags_row
        return runtimes, nearest_static, nearest_dyn, final_costs, num_nodes, flags
    else:            
        if num_workers == -1:
            num_workers = max(1, os.cpu_count() - 1)

        def _run_pool(ExecutorFactory):
            nonlocal runtimes, nearest_static, nearest_dyn, final_costs, num_nodes, flags

            with ExecutorFactory(max_workers=num_workers) as ex:
                futures = {
                    ex.submit(
                        _run_one_trial_worker,
                        (i, planners, sigmas, trial_sampler,
                        output_folder, names, vid_time, speed_up)
                    ): i
                    for i in range(n_trials)
                }

                iterator = tqdm(as_completed(futures), total=n_trials, desc="Running trials")

                for fut in iterator:
                    i = futures[fut]
                    try:
                        (runtime_row, static_row, dyn_row,
                        final_cost_row, num_nodes_row, flags_row) = fut.result()

                        runtimes[i, :] = runtime_row
                        nearest_static[i, :] = static_row
                        nearest_dyn[i, :] = dyn_row
                        final_costs[i, :] = final_cost_row
                        num_nodes[i, :] = num_nodes_row
                        flags[i, :] = flags_row
                    except Exception as e:
                        print(f"Trial {i} failed with {type(e).__name__}: {e}")

        try:
            exec = _executor_factory(use_processes=False)
            return _run_pool(exec)
        except Exception as e:
            print(f"Process parallelism failed ({type(e).__name__}: {e}), "
                    f"falling back to single-thread.")
            return run_trials(n_trials, planners, sigmas, trial_sampler,
                            output_folder, names, vid_time, speed_up,
                            num_workers=1)

def plot_trial_metrics(names, runtimes, nearest_static, nearest_dyn, final_costs, num_nodes, flags, only_success=False):
    n_methods = len(names)
    n_trials = flags.shape[0]
    colors = plt.cm.tab10.colors  # Use tab10 for consistent color palette

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 0. Fraction of success
    frac_success = np.sum(flags, axis=0) / n_trials
    axes[0].bar(names, frac_success, color=colors[:n_methods])
    axes[0].set_title("Fraction of Success")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True)
    
    # Prepare mask for successful trials
    success_mask = flags

    # 1. Runtime (median with quantiles)
    medians = []
    lowers = []
    uppers = []
    for i in range(n_methods):
        valid = runtimes[:, i]
        if only_success:
            valid = valid[success_mask[:, i]]
        medians.append(np.median(valid))
        lowers.append(np.percentile(valid, 25))
        uppers.append(np.percentile(valid, 75))
    axes[1].bar(names, medians, color=colors[:n_methods], 
                yerr=[np.array(medians) - np.array(lowers), np.array(uppers) - np.array(medians)], capsize=5)
    axes[1].set_title("Runtime" + only_success * " (Successful Trials)")
    axes[1].grid(True)

    # 2. Number of Nodes
    medians = []
    lowers = []
    uppers = []
    for i in range(n_methods):
        valid = num_nodes[:, i]
        if only_success:
            valid = valid[success_mask[:, i]]
        medians.append(np.median(valid))
        lowers.append(np.percentile(valid, 25))
        uppers.append(np.percentile(valid, 75))
    axes[2].bar(names, medians, color=colors[:n_methods], 
                yerr=[np.array(medians) - np.array(lowers), np.array(uppers) - np.array(medians)], capsize=5)
    axes[2].set_title("Number Nodes" + only_success * " (Successful Trials)")
    axes[2].grid(True)

    # 3. Final Cost
    medians = []
    lowers = []
    uppers = []
    for i in range(n_methods):
        valid = final_costs[:, i]
        if only_success:
            valid = valid[success_mask[:, i]]
        medians.append(np.median(valid))
        lowers.append(np.percentile(valid, 25))
        uppers.append(np.percentile(valid, 75))
    axes[3].bar(names, medians, color=colors[:n_methods], yerr=[np.array(medians) - np.array(lowers), np.array(uppers) - np.array(medians)], capsize=5)
    axes[3].set_title("Final Costs" + only_success * " (Successful Trials)")
    axes[3].grid(True)

    # # 4. Nearest Static and Dynamic (side by side)
    # width = 0.35
    # indices = np.arange(n_methods)
    # med_stat = []
    # med_dyn = []
    # for i in range(n_methods):
    #     valid_stat = nearest_static[:, i][success_mask[:, i]]
    #     valid_dyn = nearest_dyn[:, i][success_mask[:, i]]
    #     med_stat.append(np.median(valid_stat))
    #     med_dyn.append(np.median(valid_dyn))
    # axes[4].bar(indices - width / 2, med_stat, width, label='Static', color=[tuple(c*0.6 for c in colors[i]) for i in range(n_methods)])
    # axes[4].bar(indices + width / 2, med_dyn, width, label='Dynamic', color=colors[:n_methods])
    # axes[4].set_xticks(indices)
    # axes[4].set_xticklabels(names)
    # axes[4].set_title("Nearest Distance (Successful Trials)")
    # axes[4].legend()
    # axes[4].grid(True)

    fig.tight_layout()
    return axes

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    #### Extract polar ####
    polar_file = 'Data/polar.csv'
    polar = SailboatPolar(polar_file)

    #### Set costmap ####
    # Empty for now, but can later use real-world radar images
    costmap = empty_costmap()

    #### Set trial sampler ####
    scenario_config_path = 'Scripts/Configs/scenario_config.yaml'
    trial_sampler = lc.load_trial_sampler(scenario_config_path, costmap)

    #### Build planners ####
    planner_config_path = 'Scripts/Configs/planner_config.yaml'
    orig_planner = lc.load_graph_planner(planner_config_path, polar_file, None)

    timestamp = '20251208_225420'
    foldername = f'Data/{timestamp}'
    gcn_model_path = os.path.join(foldername, f'model_gcn')
    gcn_planner = lc.load_graph_planner(planner_config_path, polar_file, gcn_model_path)
    gat_model_path = os.path.join(foldername, f'model_gat')
    gat_planner = lc.load_graph_planner(planner_config_path, polar_file, gat_model_path)

    planners = [orig_planner, gcn_planner, gat_planner]
    names = ['Original', 'GCN', 'GAT']

    #### Run trials #####
    n_trials = 20
    output_folder = f'Results/12-10/'
    sigmas = lc.load_sigmas(planner_config_path)
    runtimes, nearest_static, nearest_dyn, final_costs, num_nodes, flags = \
        run_trials(n_trials, planners, sigmas, trial_sampler, output_folder, names,
                   vid_time=5, speed_up=100, num_workers=1)
    
    # So can plot
    plt.switch_backend("TkAgg")
    plot_trial_metrics(names, runtimes, nearest_static, nearest_dyn, final_costs, num_nodes, flags, only_success=False)
    plt.show()