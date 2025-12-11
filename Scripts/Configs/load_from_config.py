import yaml
import pickle
import numpy as np

import DatasetCreation.scenario_creator as sc
from Obstacles.costmap import empty_costmap
from Obstacles.obstacle_helpers import ObstacleCostCalculator
from GraphPlanner.pruner import Pruner
from GraphPlanner.heuristic import MaxSpeedHeuristic
from GraphPlanner.sailboat_polar import SailboatPolar
from GraphPlanner.graph_planner import GraphPlanner
from Learning.learned_heuristic import LearnedHeuristic

def load_config(config_path):
    """Load the configuration YAML into a dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

#### Scenario Creation ####

def load_target_sampler(scenario_config_path):
    cfg = load_config(scenario_config_path)
    num_targets_range = np.array(cfg["num_targets_range"])
    speed_range = np.array(cfg["speed_range"])
    goal_range = np.array(cfg["goal_range"])
    offset_range = np.array(cfg["offset_range"])
    size_range = np.array(cfg["size_range"])
    n_disc = cfg["n_disc"]
    min_allowed = cfg["min_allowed"]
    target_sampler = sc.TargetListSampler(num_targets_range, speed_range, goal_range, offset_range, size_range, 
                                          n_disc, min_allowed)
    return target_sampler

def load_endpoint_sampler(scenario_config_path, costmap=None):
    if costmap is None:
        costmap = empty_costmap()
    cfg = load_config(scenario_config_path)
    start_bounds = np.array(cfg["start_bounds"])
    goal_dist_range = np.array(cfg["goal_dist_range"])
    min_goal_allowed = cfg["min_goal_allowed"]
    endpoint_sampler = sc.EndpointSampler(start_bounds, goal_dist_range, 
                                        costmap, min_allowed=min_goal_allowed)
    return endpoint_sampler

def load_wind_sampler(scenario_config_path):
    cfg = load_config(scenario_config_path)
    wind_speed_range = cfg["wind_speed_range"]
    wind_yaw_range = cfg["wind_yaw_range"]
    wind_sampler = sc.WindSampler(wind_speed_range, wind_yaw_range)
    return wind_sampler

def load_trial_sampler(scenario_config_path, costmap=None):
    endpoint_sampler = load_endpoint_sampler(scenario_config_path, costmap)
    target_sampler = load_target_sampler(scenario_config_path)
    wind_sampler = load_wind_sampler(scenario_config_path)    
    trial_sampler = sc.TrialSampler(endpoint_sampler, costmap, target_sampler, wind_sampler)
    return trial_sampler

#### Graph Planner ####

def load_static_cost_obj(planner_config_path):
    cfg = load_config(planner_config_path)
    static_init_value = cfg['static_init_value']
    static_falloff = cfg['static_falloff']
    static_n_disc = cfg['static_n_disc']
    static_min_allowed = cfg['static_min_allowed']
    static_cost_obj = ObstacleCostCalculator(None, init_value=static_init_value, falloff=static_falloff, 
                                          n_disc=static_n_disc, min_allowed=static_min_allowed)
    return static_cost_obj

def load_dynamic_cost_obj(planner_config_path):
    cfg = load_config(planner_config_path)
    dyn_init_value = cfg['dyn_init_value']
    dyn_falloff = cfg['dyn_falloff']
    dyn_n_disc = cfg['dyn_n_disc']
    dyn_min_allowed = cfg['dyn_min_allowed']
    dyn_cost_obj = ObstacleCostCalculator(None, init_value=dyn_init_value, falloff=dyn_falloff, 
                                      n_disc=dyn_n_disc, min_allowed=dyn_min_allowed)
    return dyn_cost_obj

def load_max_speed_heuristic(planner_config_path, polar_file='Data/polar.csv'):
    cfg = load_config(planner_config_path)
    polar = SailboatPolar(polar_file)
    heuristic = MaxSpeedHeuristic(polar, np.array(cfg['deg_limits']))
    return heuristic

def load_pruner(planner_config_path):
    cfg = load_config(planner_config_path)
    pruner = Pruner(np.array(cfg['prune_limits']), verbose=False)
    return pruner

def load_graph_planner(planner_config_path, polar_file='Data/polar.csv', heuristic_model_path=None, device=None):
    cfg = load_config(planner_config_path)
    max_rel_time = cfg['max_rel_time']
    step_time = cfg['step_time']
    n_branch = cfg['n_branch']
    polar = SailboatPolar(polar_file)
    planning_bounds = np.array(cfg['planning_bounds'])
    deg_limits = np.array(cfg['deg_limits'])
    tack_penalty = cfg['tack_penalty']
    timeout = cfg['timeout']
    max_nodes = cfg['max_nodes']
    angle_rate_penalty = cfg['angle_rate_penalty']

    static_cost_obj = load_static_cost_obj(planner_config_path)
    dyn_cost_obj = load_dynamic_cost_obj(planner_config_path)
    pruner = load_pruner(planner_config_path)

    if heuristic_model_path is not None:
        heuristic = load_learned_heuristic(heuristic_model_path, device=None)
    else:
        heuristic = load_max_speed_heuristic(planner_config_path, polar_file)

    planner = GraphPlanner(max_rel_time, step_time, n_branch, polar, planning_bounds,
                    static_cost_obj, dyn_cost_obj, deg_limits=deg_limits,
                    tack_penalty=tack_penalty, pruner=pruner, heuristic=heuristic,
                    timeout=timeout, max_nodes=max_nodes, angle_rate_penalty=angle_rate_penalty)
    return planner

def load_sigmas(planner_config_path):
    cfg = load_config(planner_config_path)
    sigmas = np.array(cfg['sigmas'])
    return sigmas

#### Machine Learning ####

def load_learned_heuristic(heuristic_model_path, device=None):
    model = pickle.load(open(heuristic_model_path, 'rb'))
    if device is not None:
        model.device = device
        model.to(device)
    heuristic = LearnedHeuristic(model)
    return heuristic
