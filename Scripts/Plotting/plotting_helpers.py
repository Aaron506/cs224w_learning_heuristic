from matplotlib.patches import Polygon, Ellipse
import matplotlib.cm as cm
import numpy as np

def draw_triangle(ax, x, y, yaw, length=20, width=10, color='red', zorder=10, alpha=1):
    """
    Draw a non-equilateral (arrow-like) triangle at (x, y) with heading `yaw`.
    - yaw = 0 points right (+X)
    - `length` controls tip-to-base distance
    - `width` controls base width
    """
    # Triangle in local coordinates, pointing along +X
    triangle_local = np.array([
        [length, 0],               # Tip of arrow
        [0, -width / 2],           # Left base
        [0, width / 2]             # Right base
    ])

    # Rotation matrix
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    # Rotate and translate
    triangle_world = triangle_local @ rot.T + np.array([x, y])

    # Draw
    triangle_patch = Polygon(triangle_world, closed=True, color=color, zorder=zorder, alpha=alpha)
    ax.add_patch(triangle_patch)
    return triangle_patch

def draw_ellipsoid(ax, x, y, M, color='blue', alpha=1, fill=False):
    """
    Draw a 2D ellipsoid defined by the quadratic form: (v' M v) <= 1
    centered at (x, y) on the given axis `ax`.

    Parameters:
    - ax: matplotlib axis
    - x, y: center of the ellipsoid
    - M: 2x2 positive-definite matrix defining the ellipsoid shape
    - color: edge (or fill) color
    - alpha: transparency
    - fill: whether to fill the ellipse (True) or just draw outline (False)
    """

    # Eigen-decomposition of M
    eigvals, eigvecs = np.linalg.eigh(M)

    # Sort eigenvalues and vectors from largest (major axis) to smallest (minor axis)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Compute widths: full length of axes (factor of 2)
    widths = 2 / np.sqrt(eigvals)

    # Angle in degrees between x-axis and major axis
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    ell = Ellipse(xy=(x, y),
                  width=widths[0],
                  height=widths[1],
                  angle=angle,
                  edgecolor=color,
                  facecolor=color if fill else 'none',
                  alpha=alpha,
                  lw=1)

    ax.add_patch(ell)
    return ell

def draw_rectangle(ax, x, y, height, width, yaw=0, color='blue', zorder=5, alpha=1):
    """
    Draw a rotated rectangle centered at (x, y) with heading `yaw`.
    height: in the vertical direction (default along +Y when yaw=0)
    width: in the horizontal direction (default along +X when yaw=0)
    yaw: in radians, counterclockwise from +X
    """
    half_h = height / 2
    half_w = width / 2

    # Rectangle corners in local frame: origin-centered, upright by default
    rect_local = np.array([
        [-half_w, -half_h],  # Bottom-left
        [ half_w, -half_h],  # Bottom-right
        [ half_w,  half_h],  # Top-right
        [-half_w,  half_h],  # Top-left
    ])

    # Active Rotation matrix (CCW from +X)
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    # Rotate and translate to world coordinates
    rect_world = rect_local @ rot.T + np.array([x, y])

    # Create and add polygon patch
    rect_patch = Polygon(rect_world, closed=True, color=color, fill=False, linewidth=3, zorder=zorder, alpha=alpha)
    ax.add_patch(rect_patch)
    return rect_patch

def time_to_color(times, min_time, max_time, cmap_name='plasma'):
    # Clip time to interval
    times = np.clip(times, min_time, max_time)
    # Normalize to [0,1] for cmap
    norm_times = (times - min_time) / (max_time - min_time + 1e-6)
    return cm.get_cmap(cmap_name)(norm_times)