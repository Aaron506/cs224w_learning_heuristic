import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.widgets import Slider
from matplotlib.cm import get_cmap
import subprocess
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from GraphPlanner.graph_planner import count_tacks
from GraphPlanner.state_node import Pose2dState
from Obstacles.target import TargetList
from Obstacles.costmap import Costmap, map_to_pixel_units
from Obstacles.obstacle_helpers import get_dist_to_obs_over_time
import Plotting.plotting_helpers as ph

def plot_waypoints(waypoints, times, costmap: Costmap,
                   start_time=-1, max_rel_time=-1, goal=None, success_Q=None, wind_yaw=None, ax=None):
    """Plots waypoints overlaid on costmap."""

    ax = costmap.visualize(ax=ax, flag='occupancy', add_colorbar=False)

    # 1. Map to pixel resolution
    pix_waypoints= map_to_pixel_units(waypoints, costmap)

    start_time = start_time if start_time >= 0 else times[0]
    final_time = start_time + max_rel_time if max_rel_time > 0 else times[-1]

    for i, waypoint in enumerate(pix_waypoints):
        time = times[i]

        heat_color = ph.time_to_color(time, start_time, final_time, cmap_name='plasma')
        
        ph.draw_triangle(ax, waypoint[0], waypoint[1], waypoint[2],
                      length=20, width=10, color=heat_color, alpha=1)

    if goal is not None:
        pix_goal = map_to_pixel_units(np.array([goal.x, goal.y, goal.psi])[None,:], costmap).squeeze()
        ph.draw_triangle(ax, pix_goal[0], pix_goal[1], pix_goal[2], 
            length=20, width=10, color='blue')
        if success_Q is not None:
            M = success_Q[:2,:2].copy()
            M[:2,:2] *= costmap.m_per_pix**2
            ph.draw_ellipsoid(ax, pix_goal[0], pix_goal[1], M, color='blue', alpha=0.5)

    # Add dashed line connecting
    ax.plot(pix_waypoints[:,0], pix_waypoints[:,1],
            linestyle='--', color='gray', linewidth=1, alpha=1)
    
    # Show an arrow pointing along wind direction
    if wind_yaw is not None:
        # Wind blows from wind_yaw, so the arrow points *against* wind_yaw
        arrow_length = 0.1 * costmap.image.shape[1]
        dx = -arrow_length * np.cos(wind_yaw)
        dy = -arrow_length * np.sin(wind_yaw)

        # Choose a location for the wind arrow (e.g., top-right corner)
        x0 = int(0.85 * costmap.image.shape[1])
        y0 = int(0.85 * costmap.image.shape[0])

        ax.arrow(x0, y0, dx, dy,
                width=2,
                head_width=20,
                head_length=15,
                length_includes_head=True,
                color='cyan',
                zorder=15,
                label='Wind')

    return ax

def plot_waypoint_state_info(waypoints, times, wind_yaw=None, debug=False):
    """Plot the waypoint state information over time."""

    fig, axes = plt.subplots(1,3)
    fig.suptitle('Waypoint States')
    axes[0].plot(times - times[0], waypoints[:,0], linestyle='--', marker='o')
    axes[0].set_xlabel(r'$\Delta t$ [s]')
    axes[0].set_ylabel('x [m]')
    axes[0].grid(True)
    
    axes[1].plot(times - times[0], waypoints[:,1], linestyle='--', marker='o')
    axes[1].set_xlabel(r'$\Delta t$ [s]')
    axes[1].set_ylabel('y [m]')
    axes[1].grid(True)

    axes[2].plot(times - times[0], waypoints[:,2], linestyle='--', marker='o')
    axes[2].set_xlabel(r'$\Delta t$ [s]')
    axes[2].set_ylabel(r'$\psi$ [rad]')
    if wind_yaw is not None:
        axes[2].plot(times - times[0], wind_yaw * np.ones_like(times), linestyle='--', label='Wind Yaw')
        # Also check for tacks and mark on plot
        _, tack_inds = count_tacks(waypoints, wind_yaw)
        tack_times = np.array([(times[ind] + times[ind+1])/2 for ind in tack_inds])
        axes[2].vlines(tack_times - times[0], ymin=0, ymax=2*np.pi, linestyle='--', color='red', label='Tack Times')
        axes[2].legend()

    if debug:
        diffs = waypoints[1:,:2] - waypoints[:-1,:2]
        diff_angles = np.arctan2(diffs[:,1],diffs[:,0])
        diff_angles = np.mod(np.insert(diff_angles, 0, waypoints[0,2]), 2 * np.pi)
        axes[2].plot(times - times[0], diff_angles, linestyle='--', label=r'$\tan^{-1}(\Delta y, \Delta x)$')
        axes[2].legend()

    axes[2].set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3 * np.pi/2, 7*np.pi/4, 2*np.pi])
    axes[2].set_yticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', 
                             r'$3\pi/2$', r'$7\pi/4$', r'$2\pi$'])
    axes[2].grid(True)

    return axes

def plot_waypoint_dist_to_targets(waypoints: np.ndarray, times: np.ndarray, target_list: TargetList, n_disc: float = 10, 
                                  cmap_name="tab10"):
    """Plots the distance to other targets over time.
    Inputs:
    waypoints: (n,3) points for ego-vessel
    times: (n,) associated times when ego-vessel reaches these waypoints
    target_list: TargetList object for other vessels to measure distance to
    num_disc: how many discretization points to use between waypoints
    """
    # 1. Check the distance to the targets at each interpolated time
    disc_times, dist_info = \
        get_dist_to_obs_over_time(waypoints, times, target_list, n_disc=n_disc, extra_info=True)
    nearest_distances, nearest_ind, _ = dist_info

    # 2. Assign a color to each vessel
    unique_inds = np.unique(nearest_ind)
    cmap = get_cmap(cmap_name)
    color_map = {ind: cmap(i % 10) for i, ind in enumerate(unique_inds)}

    # 4. Plot nearest distance over time, marking the color as the nearest vessel id
    # Dashed line plot in constant black but overlay the markers in the different colors
    # Show a legend with the vessel id, retrieve via target_list[ind].track_id
    fig, ax = plt.subplots()
    ax.plot(disc_times - times[0], nearest_distances, 'k--')  # Black dashed line

    # Overlay markers with colors per vessel
    for ind in unique_inds:
        mask = (nearest_ind == ind)
        target = target_list.target_list[ind]
        ax.scatter(disc_times[mask] - times[0], nearest_distances[mask], label=f"Vessel {target.track_id}", color=color_map[ind])

    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel("Distance to Nearest Target")
    ax.set_title("Distance to Nearest Vessel Over Time")
    ax.legend()
    ax.grid(True)

    return ax

def interpolate_waypoint(waypoints, times, query_time):
    """Interpolate position and orientation from waypoints at query_time."""
    if query_time < times[0]:
        return waypoints[0]
    elif query_time >= times[-1]:
        return waypoints[-1]
    else:
        for i in range(len(times) - 1):
            if times[i] <= query_time <= times[i + 1]:
                ratio = (query_time - times[i]) / (times[i + 1] - times[i])
                interp_pose = (1 - ratio) * waypoints[i] + ratio * waypoints[i + 1]
                # Interpolate between two angles properly
                delta = np.arctan2(np.sin(waypoints[i + 1, 2] - waypoints[i, 2]), \
                                    np.cos(waypoints[i + 1, 2] - waypoints[i, 2]))
                interp_pose[2] = waypoints[i, 2] + ratio * delta
                return interp_pose

def animate_waypoints(waypoints: np.ndarray, times: np.ndarray, costmap: Costmap, target_list: TargetList = None, 
                      goal: Pose2dState = None, success_Q: np.ndarray = None, wind_yaw: float = None):
    """Provide an interactive plot where can visualize the current ego-vessel and target positions over time."""
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)

    # Convert trajectory to pixel space
    pix_waypoints = map_to_pixel_units(waypoints, costmap)

    # Slider setup
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, r'$\Delta t$ [s]', 0, times[-1] - times[0], valinit=0)

    def update():
        query_time = times[0] + slider.val

        # Clear everything
        ax.clear()

        # Show an arrow pointing along wind direction
        if wind_yaw is not None:
            # Wind blows from wind_yaw, so the arrow points *against* wind_yaw
            arrow_length = 0.1 * costmap.image.shape[1]  # Adjust based on your map size
            dx = -arrow_length * np.cos(wind_yaw)
            dy = -arrow_length * np.sin(wind_yaw)

            # Choose a location for the wind arrow (e.g., top-right corner)
            x0 = int(0.85 * costmap.image.shape[1])
            y0 = int(0.85 * costmap.image.shape[0])

            ax.arrow(x0, y0, dx, dy, width=5, head_width=10, head_length=10,
                    length_includes_head=True, color='cyan', zorder=15, label='Wind')

        # Static visualization of costmap and the full ego trajectory
        costmap.visualize(ax=ax, flag='occupancy', add_colorbar=False)
        ax.plot(pix_waypoints[:, 0], pix_waypoints[:, 1], linestyle='--', color='gray', linewidth=1, alpha=1)

        # Draw start and goal
        ph.draw_triangle(ax, pix_waypoints[0,0], pix_waypoints[0,1], pix_waypoints[0,2], 
                        length=20, width=10, color='green')
        if goal is not None:
            pix_goal = map_to_pixel_units(np.array([goal.x, goal.y, goal.psi])[None,:], costmap).squeeze()
            ph.draw_triangle(ax, pix_goal[0], pix_goal[1], pix_goal[2], 
                            length=20, width=10, color='blue')
            if success_Q is not None:
                M = success_Q[:2,:2].copy()
                M[:2,:2] *= costmap.m_per_pix**2
                ph.draw_ellipsoid(ax, pix_goal[0], pix_goal[1], M, color='blue', alpha=0.5)

        # Draw ego at interpolated pose
        interp_pose = interpolate_waypoint(waypoints, times, query_time)
        heat_color = ph.time_to_color(query_time, times[0], times[-1], cmap_name='winter_r')

        pix_interp_pose = map_to_pixel_units(interp_pose[None,:], costmap).squeeze()

        ph.draw_triangle(ax, pix_interp_pose[0], pix_interp_pose[1], pix_interp_pose[2],
                        length=20, width=10, color=heat_color, alpha=1)

        # Plot targets at query_time
        if target_list is not None:
            target_list.visualize(costmap, np.array([query_time]), ax=ax,
                                            start_time=times[0], abs_max_time=times[-1],
                                            cmap_name='plasma', use_id_color=False,
                                            alpha=1, show_id=True)

        fig.canvas.draw_idle()

    # Bind update function to slider
    slider.on_changed(lambda val: update())
    update()  # Initial draw
    plt.show()

def save_waypoints_video(waypoints: np.ndarray, times: np.ndarray, costmap: Costmap,
                                output_path: str, step_time: float, speed_up: float = 1,
                                target_list: TargetList = None, goal: Pose2dState = None, 
                                success_Q: np.ndarray = None, wind_yaw: float = None):
    """Efficient video rendering by manually saving frames and piping to FFmpeg."""
    # Setup
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    canvas = FigureCanvas(fig)
    pix_waypoints = map_to_pixel_units(waypoints, costmap)
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)

    # Frame times
    frame_times = np.arange(times[0], times[-1] + step_time, step_time)

    # Calculate playback FPS
    fps = (1.0 / step_time) * speed_up

    # Prepare FFmpeg process
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        'ffmpeg', '-loglevel', 'quiet', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgba',
        '-r', str(fps), '-i', '-', '-an',
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_path
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for query_time in frame_times:
        ax.clear()

        # Wind arrow
        if wind_yaw is not None:
            arrow_length = 0.1 * costmap.image.shape[1]
            dx = -arrow_length * np.cos(wind_yaw)
            dy = -arrow_length * np.sin(wind_yaw)
            x0 = int(0.85 * costmap.image.shape[1])
            y0 = int(0.85 * costmap.image.shape[0])
            ax.arrow(x0, y0, dx, dy, width=5, head_width=10, head_length=10,
                     length_includes_head=True, color='cyan', zorder=15)

        # Static layers
        costmap.visualize(ax=ax, flag='occupancy', add_colorbar=False)
        ax.plot(pix_waypoints[:, 0], pix_waypoints[:, 1], linestyle='--', color='gray', linewidth=1, alpha=1)

        # Start and goal
        ph.draw_triangle(ax, pix_waypoints[0,0], pix_waypoints[0,1], pix_waypoints[0,2],
                         length=20, width=10, color='green')
        if goal is not None:
            pix_goal = map_to_pixel_units(np.array([goal.x, goal.y, goal.psi])[None,:], costmap).squeeze()
            ph.draw_triangle(ax, pix_goal[0], pix_goal[1], pix_goal[2],
                             length=20, width=10, color='blue')
            if success_Q is not None:
                M = success_Q[:2,:2].copy()
                M[:2,:2] *= costmap.m_per_pix**2
                ph.draw_ellipsoid(ax, pix_goal[0], pix_goal[1], M, color='blue', alpha=0.5)

        # Ego
        interp_pose = interpolate_waypoint(waypoints, times, query_time)
        pix_interp_pose = map_to_pixel_units(interp_pose[None,:], costmap).squeeze()
        heat_color = ph.time_to_color(query_time, times[0], times[-1], cmap_name='winter_r')
        ph.draw_triangle(ax, pix_interp_pose[0], pix_interp_pose[1], pix_interp_pose[2],
                         length=20, width=10, color=heat_color, alpha=1)

        # Targets
        if target_list is not None:
            target_list.visualize(costmap, np.array([query_time]), ax=ax,
                                  start_time=times[0], abs_max_time=times[-1],
                                  cmap_name='plasma', use_id_color=False,
                                  alpha=1, show_id=True)

        ax.set_title(f"Relative Time: {query_time - times[0] :.2f} s")

        # Draw and write raw frame
        canvas.draw()
        buf = canvas.buffer_rgba()
        process.stdin.write(buf)

    process.stdin.close()
    process.wait()
    plt.close(fig)
