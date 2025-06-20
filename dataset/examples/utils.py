from test_runner import TestRunner
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
from matplotlib.patches import Arrow, Circle
from scipy.spatial.transform import Rotation as R
import habitat_sim
import math
import torch.nn.functional as F
import torch
from torchvision.transforms import ToTensor

def get_mask(obs1, pose1, obs2, pose2, hfov=90*(math.pi/180), W=1920, H=1080):
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])

        
        depths = [obs1["depth_sensor"], obs2["depth_sensor"]]
        rgbs = [obs1["color_sensor"], obs2["color_sensor"]]
        
        translation_0 = pose1[:3] # X Z Y
        translation_0[1], translation_0[2] = translation_0[2], translation_0[1] # X Y Z

        translation_1 = pose2[:3] # X Z Y
        translation_1[1], translation_1[2] = translation_1[2], translation_1[1] # X Y Z

        assert (translation_0[1] == translation_1[1]), "wrong poses with different floor heights{}, {}".format(translation_0, translation_1)

        rot_quat_0 = R.from_euler("xyz", pose1[3:], degrees=True).as_quat()
        rot_quat_1 = R.from_euler("xyz", pose2[3:], degrees=True).as_quat()

        rotation_0 = np.array(R.from_quat(rot_quat_0).as_matrix())
        rotation_1 = np.array(R.from_quat(rot_quat_1).as_matrix())


        # Now get an approximation for the true world coordinates -- see if they make sense
        # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
        xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,H))
        depth = depths[0].reshape(1, H, W).astype(np.float64)
        xs = xs.reshape(1,H,W)
        ys = ys.reshape(1,H,W)

        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        xy_c0 = np.matmul(np.linalg.inv(K), xys)

        T_world_camera0 = np.eye(4)
        T_world_camera0[0:3,0:3] = rotation_0
        T_world_camera0[0:3,3] = translation_0

        T_world_camera1 = np.eye(4)
        T_world_camera1[0:3,0:3] =  rotation_1
        T_world_camera1[0:3,3] = translation_1

        # Invert to get world --> camera
        T_camera1_world = np.linalg.inv(T_world_camera1)

        # Transformation matrix between views
        # Aka the position of camera0 in camera1's coordinate frame
        T_camera1_camera0 = np.matmul(T_camera1_world, T_world_camera0)

        # Finally transform actual points
        xy_c1 = np.matmul(T_camera1_camera0, xy_c0)
        xy_newimg = np.matmul(K, xy_c1)

        # Normalize by negative depth
        xys_newimg = xy_newimg[0:2,:] / -xy_newimg[2:3,:]
        # Flip back to y-down to match array indexing
        xys_newimg[1] *= -1

        sampler = torch.Tensor(xys_newimg).view(2, H, W).permute(1,2,0).unsqueeze(0)

        # Create generated image
        img1_tensor = ToTensor()(rgbs[0]).unsqueeze(0)
        img2_tensor = ToTensor()(rgbs[1]).unsqueeze(0)
        img2_warped = F.grid_sample(img2_tensor, sampler, align_corners=True)

        masked_image = np.abs(img2_warped.squeeze().permute(1,2,0) - img1_tensor.squeeze().permute(1,2,0)).numpy()

        return masked_image 

def pivot_candidates(original_map: np.ndarray) -> np.ndarray:
    pivot_array = []
    rows, cols = original_map.shape
    for i in range(rows):
        for j in range(cols):
            if original_map[i][j] == 1:
                is_top_row = i - 1 < 0
                is_bottom_row = i + 1 > rows - 1
                is_left_col = j - 1 < 0
                is_right_col = j + 1 > cols - 1
                if is_top_row:
                    if is_left_col:
                        if (
                            original_map[i][j + 1] == 0
                            or original_map[i + 1][j] == 0
                        ):
                            pivot_array.append((i, j))
                    elif is_right_col:
                        if (
                            original_map[i][j - 1] == 0
                            or original_map[i + 1][j] == 0
                        ):
                            pivot_array.append((i, j))
                    else:
                        if (
                            original_map[i + 1][j] == 0
                            or original_map[i][j - 1] == 0
                        ):
                            pivot_array.append((i, j))
                elif is_bottom_row:
                    if is_left_col:
                        if(
                            original_map[i][j + 1] == 0
                            or original_map[i - 1][j] == 0
                        ):
                            pivot_array.append((i, j))
                    elif is_right_col:
                        if (
                            original_map[i][j - 1] == 0
                            or original_map[i - 1][j] == 0
                        ):
                            pivot_array.append((i, j))
                    else:
                        if (
                            original_map[i - 1][j] == 0
                            or original_map[i][j - 1] == 0
                            or original_map[i][j + 1] == 0
                        ):
                            pivot_array.append((i, j))
                else:
                    if is_left_col:
                        if (
                            original_map[i][j + 1] == 0
                            or original_map[i + 1][j] == 0
                            or original_map[i - 1][j] == 0
                        ):
                            pivot_array.append((i, j))
                    elif is_right_col:
                        if (
                            original_map[i + 1][j] == 0
                            or original_map[i - 1][j] == 0
                            or original_map[i][j - 1] == 0
                        ):
                            pivot_array.append((i, j))
                    else:
                        if original_map[i - 1][j] == 0:
                            if original_map[i][j - 1] == 0:
                                pivot_array.append((i, j, 0))
                            elif original_map[i][j + 1] == 0:
                                pivot_array.append((i, j, 2))
                            else:
                                pivot_array.append((i, j, 5))
                        elif original_map[i + 1][j] == 0:
                            if original_map[i][j - 1] == 0:
                                pivot_array.append((i, j, 1))
                            elif original_map[i][j + 1] == 0:
                                pivot_array.append((i, j, 3))
                            else:
                                pivot_array.append((i, j, 7))
                        elif original_map[i][j - 1] == 0:
                            pivot_array.append((i, j, 4))
                        elif original_map[i][j + 1] == 0:
                            pivot_array.append((i, j, 6))
                        # check if any of the 8 neighbors are 0
                        elif (
                            original_map[i - 1][j - 1] == 0
                            or original_map[i - 1][j + 1] == 0
                            or original_map[i + 1][j - 1] == 0
                            or original_map[i + 1][j + 1] == 0
                        ):
                            pivot_array.append((i, j, 7))
    return set(pivot_array)  # pivot array is a list of [x,y,corner] where corner is 0,1,2,3,4,5,6,7

def get_candidate_svd(runner: TestRunner, height: float, pivot: Tuple[float, float, float], angle: float, original_map: np.ndarray,
                    scene_name: str, floor_no: int, iter_number: int, obs_ind=None, offset=0, plot_flag=False, pivot_list=None, pivot_indicies=None):
    """
    Parameters
    ----------
    pivot
          pose  : ([position],[rotation])
    n
          int   : Number of candidates
    """

    rotation = R.from_euler("xyz", [0, angle, 0], degrees=True)
    rotation = rotation.as_quat()

    # get pivot_grid
    pivot = np.array(pivot)
    pivot_grid = runner.get_grid_pos(pivot)

    new_pivot = pivot[:3]
    obs = runner.get_rgbd_from_pose(new_pivot, rotation)

    if plot_flag:
        plot_near_point(scene_name, pivot, original_map, pivot_grid, new_pivot_grid, floor_no, iter_number)

    candidate = (obs, new_pivot, rotation, angle)

    return candidate


def in_map(x, y, theta, original_map):
    new_x = x + 50 * np.cos(math.radians(theta))
    new_y = y + 50 * np.sin(math.radians(theta))
    new_x_ = x + 100 * np.cos(math.radians(theta))
    new_y_ = y + 100 * np.sin(math.radians(theta))

    return (
        new_x >= 0
        and new_x < original_map.shape[0]
        and new_y >= 0
        and new_y < original_map.shape[1]
    ) and (
        new_x_ >= 0
        and new_x_ < original_map.shape[0]
        and new_y_ >= 0
        and new_y_ < original_map.shape[1]
    )
