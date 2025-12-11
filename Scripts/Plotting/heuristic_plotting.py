import matplotlib.pyplot as plt
import numpy as np

from GraphPlanner.graph_planner import PlanningHeuristic
from GraphPlanner.state_node import Pose2dState
from Obstacles.target import TargetList
import Plotting.plotting_helpers as ph

def find_cost_to_go_arr(time: float, pose_psi: float, wind_yaw: float, wind_speed: float, 
                        target_list: TargetList, bounds: np.ndarray, num_disc: float,
                        heuristic: PlanningHeuristic, goal: Pose2dState, success_Q: np.ndarray = None,
                        **cost_to_go_args):
    # 1. Compute x,y points uniformly on grid within bounds, num_disc in each direction
    num_disc = int(num_disc)
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    xs = np.linspace(x_min, x_max, num_disc)
    ys = np.linspace(y_min, y_max, num_disc)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X.ravel(), Y.ravel()], axis=-1) # (N, 2)

    # 2. Concatenate with the angle
    poses = np.hstack([coords, np.ones((coords.shape[0], 1)) * pose_psi])

    # 3. Set up the heuristic
    heuristic.set_wind(wind_yaw, wind_speed)
    heuristic.set_map(target_list=target_list)
    heuristic.set_goal(goal, success_Q)

    # 4. Evaluate the heuristic at each pose at given time
    cost_to_go_arr = heuristic.find_cost_to_go(poses, time, **cost_to_go_args)
    cost_to_go_arr = np.asarray(cost_to_go_arr).reshape(num_disc, num_disc)

    return X, Y, cost_to_go_arr, poses

def heuristic_heatmap(time: float, pose_psi: float, wind_yaw: float, wind_speed: float, 
                      target_list: TargetList, bounds: np.ndarray, num_disc: float,
                      heuristic: PlanningHeuristic, goal: Pose2dState, success_Q: np.ndarray = None,
                      vmin: float = None, vmax: float = None, vis_times: np.ndarray = None, ax=None,
                      **cost_to_go_args):
    # 1. Compute cost_to_go_arr for poses on grid
    _, _, cost_to_go_arr, _ = find_cost_to_go_arr(time, pose_psi, wind_yaw, wind_speed, target_list, 
                                                      bounds, num_disc, heuristic, goal, success_Q,
                                                      **cost_to_go_args)
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # 2. Plot the cost_to_go_arr as a heatmap using coords
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    im = ax.imshow(cost_to_go_arr, extent=[x_min, x_max, y_min, y_max],
                   origin='lower', cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Cost-to-go')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 3. Overlay the goal as a red triangle
    ph.draw_triangle(ax, goal.x, goal.y, goal.psi,
                         length=100, width=50, color='red')

    # 4. Overlay the targets at the current, and possibly future, time
    if target_list is not None:
        if vis_times is None:
            vis_times = np.array([time])
        target_list.visualize(None, vis_times, ax=ax,
                                  start_time=vis_times[0], abs_max_time=vis_times[-1],
                                  cmap_name='plasma', use_id_color=False,
                                  alpha=1, show_id=True)
    
    # 5. Overlay the wind direction
    height, width = (y_max - y_min), (x_max - x_min)
    arrow_length = 0.1 * width
    # Wind blows from wind_yaw, so the arrow points *against* wind_yaw
    dx = -arrow_length * np.cos(wind_yaw)
    dy = -arrow_length * np.sin(wind_yaw)
    x0 = int(x_min + 0.85 * width)
    y0 = int(y_min + 0.85 * height)
    ax.arrow(x0, y0, dx, dy, width=2, head_width=100, head_length=75, 
             length_includes_head=True, color='cyan', zorder=15, label='Wind')
    ax.set_aspect('equal')

    return ax