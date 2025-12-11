import numpy as np
import os
from dataclasses import dataclass
import uuid
from typing import List

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import numpy as np
import multiprocessing as mp
_SPAWN_CTX = mp.get_context("spawn")

from GraphPlanner.state_node import Pose2dState, form_success_Q
from GraphPlanner.graph_planner import GraphPlanner
from Math.geometry import form_2d_passive_rot

@dataclass
class GraphDatum():
    pose: Pose2dState # (x,y,psi)
    wind_yaw: float
    wind_speed: float
    targets: List[np.ndarray] # list of (px, py, vx, vy, h, w)
    cost_to_go: float
    traj_id: str
    path_index: int
    goal_wrt_costmap: Pose2dState # (x, y, psi)

def make_relative_goal(pose: Pose2dState, goal: Pose2dState) -> Pose2dState:
    """Make given pose be defined wrt. goal pose."""
    psi_wrt_goal = pose.psi - goal.psi
    # Offset is defined wrt. original x, y
    offset = np.array([pose.x - goal.x, pose.y - goal.y])
    # Passive rotation so defined wrt. goal psi
    R = form_2d_passive_rot(goal.psi)
    x_wrt_goal, y_wrt_goal = R @ offset
    pose_wrt_goal = Pose2dState(x_wrt_goal, y_wrt_goal, psi_wrt_goal)
    return pose_wrt_goal

def form_graph_datum(waypoint: np.ndarray, target_positions: np.ndarray, target_velocities: np.ndarray, 
                    target_bboxs: np.ndarray, wind_yaw: float, wind_speed: float, cost_to_go: float, goal: Pose2dState,
                        traj_id = '', path_index = -1) -> GraphDatum:
    # Convert quantities to be wrt. goal frame
    pose_wrt_goal = make_relative_goal(Pose2dState(waypoint[0], waypoint[1], waypoint[2]), goal)
    wind_yaw_wrt_goal = wind_yaw - goal.psi

    targets_wrt_goal = []
    for i, target_position in enumerate(target_positions):
        # Dummy yaw, target is not oriented
        target_position_wrt_goal = make_relative_goal(Pose2dState(target_position[0], target_position[1], 0.0), goal)
        # Only want to rotate velocities, not translate
        dummy_goal = Pose2dState(0.0, 0.0, goal.psi)
        target_vel_wrt_goal = make_relative_goal(Pose2dState(target_velocities[i,0], target_velocities[i,1], 0.0), dummy_goal)
        target_wrt_goal = np.array([target_position_wrt_goal.x, target_position_wrt_goal.y,
                  target_vel_wrt_goal.x, target_vel_wrt_goal.y, target_bboxs[i,0], target_bboxs[i,1]])
        targets_wrt_goal.append(target_wrt_goal)
    targets_wrt_goal = np.array(targets_wrt_goal)

    datum = GraphDatum(pose_wrt_goal, wind_yaw_wrt_goal, wind_speed, targets_wrt_goal, cost_to_go,
                       traj_id, path_index, goal)
    return datum

def gen_traj_id():
    return str(uuid.uuid4().hex)

def generate_data_one_trial(planner: GraphPlanner, start, goal, success_Q, costmap, 
              target_list, wind_yaw, wind_speed) -> List[GraphDatum]:
    """Extracts the input/output data for one trial."""
    # 1. Set the planner environment conditions
    planner.set_env_conditions(wind_yaw, wind_speed, costmap, target_list)

    # 2. Solve with the planner
    # Do not add goal last, so still reasonable to use data even if planner
    # fails to reach the desired goal
    waypoints, times, costs, num_nodes, flag = planner.solve(costmap.time, start, goal, success_Q,
                        add_goal_last=False)

    # 3. Extract relevant target info
    # (n, m) where n = # times, m = # targets
    target_positions_over_time = target_list.get_coords_over_time(times)
    # (H,W) bbox in each row
    target_bboxs = target_list.bboxs # (m,2)
    graph_data = []

    # 4. Generate a random hash string
    traj_id = gen_traj_id()

    # Exclude the final state which has zero cost-to-go
    for i, cost in enumerate(costs[:-1]):
        # a. Determine the associated state
        waypoint = waypoints[i]

        # b. Determine the current target list states (m, 2)
        target_positions = target_positions_over_time[i,:]
        target_velocities = target_list.vels.copy()

        # c. Determine the remaining cost-to-go
        # costs[-1] is final path cost, 
        # cost is cost up to and including current node
        # Hence, costs[-1] - cost is remaining cost
        cost_to_go = costs[-1] - cost

        datum = form_graph_datum(waypoint, target_positions, target_velocities, target_bboxs,
                         wind_yaw, wind_speed, cost_to_go, goal, traj_id = traj_id, path_index=i)
        graph_data.append(datum)

    return graph_data, flag

def run_one_trial(args):
    """Standalone worker for a single dataset trial."""
    trial_index, planner, sigmas, trial_sampler = args

    try:
        # 0. Sample the trial
        start, goal, costmap, target_list, wind_yaw, wind_speed = trial_sampler.sample()
        success_Q = form_success_Q(sigmas, rot_angle=goal.psi)
        # 1. Generate the data for the trial
        graph_data, flag = generate_data_one_trial(planner, start, goal, success_Q,
                                                   costmap, target_list, wind_yaw, wind_speed)
        return graph_data, flag
    except Exception as e:
        return None, f"Error in trial {trial_index}: {e}", trial_index

def _executor_factory(use_processes=True):
    """Return an Executor factory with correct multiprocessing context."""
    if use_processes:
        return (lambda **kw: ProcessPoolExecutor(mp_context=_SPAWN_CTX, **kw))
    else:
        return ThreadPoolExecutor

def generate_dataset(n_trials: int, planner, sigmas: np.ndarray,
                     trial_sampler, verbose=True, num_workers=1):
    """
    Generate dataset in parallel (spawn-safe).
    """
    # Single-process version
    if num_workers == 1:
        dataset = []
        n_success = 0
        for trial in range(n_trials):
            if verbose:
                print(f"Starting trial {trial}")
            data, flag = run_one_trial((trial, planner, sigmas, trial_sampler))
            if data is not None:
                dataset.extend(data)
                n_success += int(flag)
                if verbose:
                    print(f"Planner reached goal {flag}")
            else:
                print(f"Trial {trial} failed: {flag}")
        success_rate = n_success / n_trials
        return dataset, success_rate

    # Multi-process version
    if num_workers == -1:
        num_workers = max(1, os.cpu_count() - 1)
    
    def _run_pool(ExecutorFactory):
        # Will be list of GraphDatum objects
        dataset = []
        n_success = 0 
        with ExecutorFactory(max_workers=num_workers) as ex:
            futures = {
                ex.submit(run_one_trial, (i, planner, sigmas, trial_sampler)): i
                for i in range(n_trials)
            }
            iterator = tqdm(as_completed(futures), total=n_trials,
                            desc="Generating dataset") if verbose else as_completed(futures)

            for fut in iterator:
                try:
                    data, flag = fut.result()
                    if data is not None:
                        dataset.extend(data)
                        n_success += int(flag)
                    else:
                        print(flag)
                except Exception as e:
                    print(f"Trial failed with {type(e).__name__}: {e}")
        success_rate = n_success / n_trials
        return dataset, success_rate

    try:
        ProcessExec = _executor_factory(use_processes=True)
        dataset, success_rate = _run_pool(ProcessExec)
    except Exception as e:
        if verbose:
            print(f"Process parallelism failed ({type(e).__name__}: {e}), falling back to single-thread.")
        return generate_dataset(n_trials, planner, sigmas, trial_sampler, verbose=verbose, num_workers=1)

    return dataset, success_rate

def get_dataset_traj_ids(dataset):
    return list({datum.traj_id for datum in dataset})

def get_one_traj_data(dataset, traj_id):
    # Filter only ones with given traj_id
    traj_data = list(filter(lambda d: d.traj_id == traj_id, dataset))
    # Order ascending by path_index
    traj_data = sorted(traj_data, key = lambda x : x.path_index)
    return traj_data
