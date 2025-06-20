import numpy as np
from numba import njit
import math
from plotter import *
from typing import List, Tuple, Dict, Union
from numpy import linalg as LA

# helper function for pivot_candidates, kept outside of class for numba reasons
# @njit
def in_map(x_bound, y_bound, x, y, theta, range1=50, range2=100):
      new_x = x + range1 * np.cos(theta)
      new_y = y + range1 * np.sin(theta)
      new_x_ = x + range2 * np.cos(theta)
      new_y_ = y + range2 * np.sin(theta)
      return (
            new_x >= 0
            and new_x < x_bound
            and new_y >= 0
            and new_y < y_bound
      ) and (
            new_x_ >= 0
            and new_x_ < x_bound
            and new_y_ >= 0
            and new_y_ < y_bound
      )

# helper function for pivot_candidates, kept outside of class for numba reasons

def depth2grid(i, angle_range, pivot_, depth_slice, x0, y0):
    rad = angle_range[i]

    # @TODO: check why math is used instead of np
    del_x = depth_slice[i] * math.cos(np.pi / 2 + rad)
    del_y = depth_slice[i] * math.sin(np.pi / 2 + rad)

    x1, y1 = pivot_[0] + del_x, pivot_[2] - del_y

    # then convert to grid
    px = x0 + (del_x * 90)
    py = y0 - (del_y * 90)

    return (pivot_[0], pivot_[2], x1, y1), (x0, y0, px, py)

def arrow(x0, y0, length, angle, color):
    x1 = math.cos(np.pi / 2 + angle) * length
    y1 = math.sin(np.pi / 2 + angle) * length

    return Arrow(x0, y0, x1, -y1, width=100.0, color=color)


def update_map_ray_helper(
    explored_map: np.ndarray,
    stopping_map: np.ndarray,
    grid_pcl: Tuple[int,int,float,float],
    grid_pivot: List[int],
    step_size=1,
    view_range=80000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    helper function for update_map_ray and update_map_ray_check
    """
    grid_pcl = np.array(grid_pcl[2:4])
    grid_pivot = np.array(grid_pivot)

    # step size
    dist = LA.norm(grid_pcl - grid_pivot)
    cos_theta, sin_theta = np.divide(grid_pcl - grid_pivot, dist)

    steps = np.arange(step_size, dist, step_size)


    
    grid_rays = np.column_stack(
        [grid_pivot[0] + steps * cos_theta, grid_pivot[1] + steps * sin_theta]
    )

    # limit grid_rays to range of valid indices
    grid_rays = np.clip(grid_rays, [0, 0], [explored_map.shape[1] - 1, explored_map.shape[0] - 1])

    # check inbound and in view range
    inbound = (
        (grid_rays[:, 0] >= 0)
        & (grid_rays[:, 0] < explored_map.shape[1])
        & (grid_rays[:, 1] >= 0)
        & (grid_rays[:, 1] < explored_map.shape[0])
    )
    
    in_view_range = LA.norm(grid_rays - grid_pivot, axis=1) <= view_range

    # cast grid_rays to int
    grid_rays = grid_rays.astype(int)
    
    explored_mask = (
        explored_map[grid_rays[:, 1], grid_rays[:, 0]] == 1
    )

    # update explored map
    explored_mask = explored_mask & inbound & in_view_range
    explored_map[
        grid_rays[explored_mask, 1], grid_rays[explored_mask, 0]
    ] = 0.5
    stopping_map[
        grid_rays[explored_mask, 1], grid_rays[explored_mask, 0]
    ] = 0



    return explored_map, stopping_map, grid_rays, explored_mask, inbound, in_view_range