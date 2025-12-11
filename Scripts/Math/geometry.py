import numpy as np

def form_2d_passive_rot(psi):
    """Computes the 2D passive rotation matrix to go from a point expressed in frame A
    to same point expressed in frame B, where frame B is rotated by psi 
    counterclockwise from frame A."""
    return np.array([[np.cos(psi), np.sin(psi)],
                    [-np.sin(psi), np.cos(psi)]])

def rectangles_to_vertices(rectangles):
    """
    Inputs:
    rectangles: any shape where last dimension is 4 e.g., (n,4) or (n,m,4)
    where last dimension specifies a rectangle via (x_center, y_center, height, width).
    Outputs:
    vertices: shape rectangles.shape[:-1], 4, 2 where last two dimensions specify each
    rectangle's vertices (xmin,ymin), (xmin,ymax), (xmax,ymin), (xmax,ymax).
    """
    flat_rectangles = rectangles.reshape((-1,4)) # (n,4)
    xmin = flat_rectangles[:, 0] - flat_rectangles[:, 3]/2 # (n,)
    xmax = flat_rectangles[:, 0] + flat_rectangles[:, 3]/2
    ymin = flat_rectangles[:, 1] - flat_rectangles[:, 2]/2
    ymax = flat_rectangles[:, 1] + flat_rectangles[:, 2]/2
    vertices = np.stack([np.stack([xmin,ymin], axis=-1), # (n,2)
                np.stack([xmin,ymax], axis=-1), # (n,2)
                np.stack([xmax,ymin], axis=-1), # (n,2)
                np.stack([xmax,ymax], axis=-1)], # (n,2)
                axis=1) # (n,2), (n,2), (n,2), (n,2) -> (n,4,2)
    vertices = vertices.reshape(rectangles.shape[:-1] + (4,2))
    return vertices

def project_points_to_paired_rectangles_hull(points: np.ndarray, rectangles: np.ndarray) -> np.ndarray:
    """
    Finds the closest point on boundary of each rectangle in rectangles[i] to points[i].
    Inputs:
    - points: (n,2) points[i] = (x,y)
    - rectangles: (n,m,4), rectangles[i,j] = (x_center, y_center, height, width)
    Outputs:
    - projections: (n,m,2), projection of i'th point onto j'th rectangle's boundary
    - inside: (n,m) whether i'th point was inside j'th rectangle
    """
    x, y = points[:, None, 0], points[:, None, 1]  # (n,1)
    m = rectangles.shape[1]
    
    xmin = rectangles[:, :, 0] - rectangles[:, :, 3] / 2
    xmax = rectangles[:, :, 0] + rectangles[:, :, 3] / 2
    ymin = rectangles[:, :, 1] - rectangles[:, :, 2] / 2
    ymax = rectangles[:, :, 1] + rectangles[:, :, 2] / 2

    # Current projection (clip to box, works for points outside)
    x_proj = np.clip(x, xmin, xmax)
    y_proj = np.clip(y, ymin, ymax)

    # Determine points inside the rectangles
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    # For inside points, project to nearest edge
    # Distance to each edge
    dx_left   = np.abs(x - xmin)
    dx_right  = np.abs(xmax - x)
    dy_bottom = np.abs(y - ymin)
    dy_top    = np.abs(ymax - y)

    # Stack distances to find minimum
    dists = np.stack([dx_left, dx_right, dy_bottom, dy_top], axis=-1)  # (n, m, 4)
    min_side = np.argmin(dists, axis=-1)  # (n, m)

    # Replace x_proj and y_proj for inside points
    x_proj_inside = np.where(min_side == 0, xmin,
                     np.where(min_side == 1, xmax, x))
    y_proj_inside = np.where(min_side == 2, ymin,
                     np.where(min_side == 3, ymax, y))

    # Apply inside point correction
    x_proj = np.where(inside, x_proj_inside, x_proj)
    y_proj = np.where(inside, y_proj_inside, y_proj)

    projections = np.stack([x_proj, y_proj], axis=-1)

    return projections, inside

def points_to_paired_rectangles_distance(points: np.ndarray, rectangles: np.ndarray) -> np.ndarray:
    """Finds the distance of points[i] to paired rectangles[i].
    Inputs:
    points: (n,2) points[i] = (x,y) in each row
    rectangles: (n,m,4) where rectangles[i] stores m rectangles to be paired against points[i]
    each rectangle i.e., rectangles[i,j] has (x_center, y_center, bbox_height, bbox_width)
    Outputs:
    distances: (n,m) where distances[i,j] = distance of i'th point to j'th paired rectangle
    """
    x, y = points[:,None,0], points[:,None,1] # (n,1)
    
    xmin = rectangles[:, :, 0] - rectangles[:, :, 3]/2 # (n,m)
    xmax = rectangles[:, :, 0] + rectangles[:, :, 3]/2
    ymin = rectangles[:, :, 1] - rectangles[:, :, 2]/2
    ymax = rectangles[:, :, 1] + rectangles[:, :, 2]/2

    dx = np.maximum(xmin - x, 0) + np.maximum(x - xmax, 0)
    dy = np.maximum(ymin - y, 0) + np.maximum(y - ymax, 0)

    distances = np.hypot(dx, dy) # (n,m)

    return distances