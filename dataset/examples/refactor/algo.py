from test_runner import TestRunner
from settings import default_sim_settings
from collections import *

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from matplotlib.patches import Arrow, Circle
import matplotlib.cm as cmx
import matplotlib.colors as color_
from scipy.spatial.transform import Rotation as R
import habitat_sim
from numpy import linalg as LA
from copy import copy
import argparse
from multiprocessing import cpu_count
import multiprocessing
import time
from PIL import Image
from pathlib import Path
from typing import Tuple, List

#mp
import multiprocessing as mp

# file handling
import os, shutil
import json
import math
from copy import copy
import cv2
from tqdm import tqdm
import glob

import concurrent.futures

from scipy.spatial import distance_matrix

from visualize_graphs import visualize_graph

colors = ["b", "r", "k", "m", "y", "g", "c"]
corners = ["lu", "ld", "ru", "rd", "l", "u", "r", "d"]

color_map = {
    "b": "blue",
    "r": "red",
    "k": "black",
    "m": "purple",
    "y": "yellow",
    "g": "green",
    "c": "cyan",
}
n_cores = cpu_count()
# print(f"Number of Logical CPU cores: {n_cores}")



def get_candidate_GA(
    runner: TestRunner,
    height: float,
    pivot: Tuple[float, float, float],
    angle: float,
    original_map: np.ndarray,
    scene_name: str,
    floor_no: int,
    iter_number: int,
    obs_ind=None,
    offset=0,
    plot_flag=False,
    pivot_list=None,
    pivot_indicies=None,
):
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

    # obtain new_pivot
    new_pivot, new_pivot_grid = runner.get_random_point_near(
        height,
        grid=original_map,
        circle_center=pivot,
        radius_grid=50,
        # pivot_list=pivot_list,
        # pivot_indicies=pivot_indicies,
        scalar_factor=100,
    )

    new_pivot = np.array(new_pivot)

    obs = runner.get_rgbd_from_pose(new_pivot, rotation)

    if plot_flag:
        fig, ax = plt.subplots()
        ax.set_title(f"scene_name {scene_name} {pivot[1]}")
        ax.imshow(original_map, cmap="gray")

        ax.add_patch(Circle((pivot_grid[0], pivot_grid[1]), radius=10, color="red"))
        ax.add_patch(
            Circle((new_pivot_grid[0], new_pivot_grid[1]), radius=10, color="blue")
        )
        fig.savefig(
            os.path.join(
            "temp", "More_vis", str(scene_name), str(floor_no), f"near_point_{iter_number}_GA.png"
            )
        )
        plt.close()

    candidate = (obs, new_pivot, rotation, angle)

    return candidate


def sift_features(cdt_color):
    """Checks for featureless image conditions.

    Args:
        cdt_color: Candidate color image.

    Returns:
        True if the image contains at least 100 SIFT features.
        False if the image contains fewer than 100 SIFT features.
    """
    gray_cdt = cv2.cvtColor(cdt_color, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()

    ##SIFT LOCALIZATION AND GETTING DESCRIPTOR THAT HAS HISTOGRAM ORIENTATION WITH SHAPE (#keyPoints*128):
    keyPoints, descriptor = sift.detectAndCompute(gray_cdt, None)
    total_features = len(keyPoints)
    print("------> Total Number of features: ", total_features)
    if total_features < 100:  ### should be a parameter and not just manual threshold
        drawKeypoints = cv2.drawKeypoints(
            gray_cdt,
            keyPoints,
            0,
            (255, 0, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return False
    return True


def intersect_score_simple(
    runner,
    original_map,
    explored_map,
    stopping_map,
    candidate,
    candidate_pose,
    saved_depth,
    saved_color,
    saved_pose,
    scene_name,
    iter_number,
    iter,
    init_unexplored,
    floor_time,
    scene_time,
    pivot_time,
    lower_threshold=0.15,
    upper_threshold=0.9,
    ALPHA=0.1,
    plot_flag=False,
    pivot_array=None,
    obs_array=None,
):
    max_overlap = -np.inf
    max_weight = -np.inf
    cam2base_h = 1.5
    bounds = runner._sim.pathfinder.get_bounds()
    p1 = candidate_pose  # xzyrpy
    c1 = candidate
    (obs_c1, pivot_c1, rotation_c1, candidates_rotation_rad_c1) = c1
    g1 = runner.get_grid_pos(pivot_c1)
    rotation_1 = [0, candidates_rotation_rad_c1, 0]

    c1_pose = list(pivot_c1.copy())
    c1_pose.extend(rotation_1)
    c1_pose[1] += cam2base_h

    #### xzy -> xyz
    temp = c1_pose[1]
    c1_pose[1] = c1_pose[2]
    c1_pose[2] = temp

    overlap_n = 0
    world_lines1, grid_lines1 = draw_sight_rays(
        runner, [c1], 0, pivot_c1[1], scene_name, iter_number, iter
    )
    # assert()
    temp_map_copy = original_map.copy()
    temp_map2_copy = original_map.copy()

    if plot_flag:
        fig, ax = plt.subplots()
        topdown = runner.get_top_down(height=pivot_c1[1])
        x0, y0, px, py = grid_lines1[0]
        ax.add_patch(Circle((x0, y0), radius=25, color="red"))
        ax.imshow(topdown)
        for grid_line in grid_lines1:
            x0, y0, px, py = grid_line
            ax.plot((x0, px), (y0, py), color="blue")

        fig.savefig(
            os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"update_grid_lines_test.png")
        )
        plt.close()

    temp_map_copy, temp_map2_copy, count_1 = update_map(
        runner,
        temp_map_copy,
        temp_map2_copy,
        world_lines1,
        grid_lines1,
        bounds,
        pivot_c1,
        scene_name,
        iter_number,
        init_unexplored,
        floor_time,
        scene_time,
        pivot_time,
        "fc",
    )
    explored_map_copy = explored_map.copy()
    stopping_map_copy = stopping_map.copy()
    _, _, count_2, o_count = update_map_check(
        runner,
        explored_map_copy,
        stopping_map_copy,
        world_lines1,
        grid_lines1,
        bounds,
        pivot_c1,
        scene_name,
        iter_number,
        iter,
    )
    diff_score = 100 * (count_2 - o_count) / np.sum(original_map)

    if plot_flag:
        fig, ax = plt.subplots()
        ax.set_title(scene_name + " " + str(pivot_c1[1]))
        ax.imshow(temp_map_copy, cmap="gray")

        fig.savefig(
            os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"update_map_{iter}_1.png")
        )
        plt.close()

    rel_ele = np.ones(1, dtype=np.int32)

    if len(saved_depth) == 0:
        return 0, 100 * count_1 / np.sum(original_map), obs_c1, candidate_pose, rel_ele

    # @TODO: Need optimization here
    for ind_2, svd_pose in enumerate(saved_pose):  ###### Need optimization here
        rel_ele = np.zeros(len(saved_pose) + 1, dtype=np.int32)
        rel_ele[-1] = 1
        p2 = copy(svd_pose)
        p2[2] -= 1.5
        temp = p2[2]
        p2[2] = p2[1]
        p2[1] = temp
        g2 = runner.get_grid_pos(p2)

        c2 = get_candidate_GA(
            runner, p2[1], p2, p2[4], original_map, scene_name, iter_number, iter
        )  # ,pivot_list=pivot_array, pivot_indicies=obs_array)
        (obs_c2, pivot_c2, rotation_c2, candidates_rotation_rad_c2) = c2

        rotation_2 = [0, candidates_rotation_rad_c2, 0]
        c2_pose = list(pivot_c2.copy())
        c2_pose.extend(rotation_2)

        world_lines2, grid_lines2 = draw_sight_rays(
            runner, [c2], 0, pivot_c2[1], scene_name, iter_number, iter + 0.5
        )
        if plot_flag:
            length = 100
            fig, ax = plt.subplots()
            ax.set_title(scene_name + " " + str(pivot_c1[1]))
            ax.imshow(runner.get_top_down(height=pivot_c1[1]))
            x0, y0 = int(g1[0]), int(g1[1])
            ax.add_patch(Circle((int(g1[0]), int(g1[1])), radius=25, color="blue"))
            ax.add_patch(
                arrow(x0, y0, length, math.radians(candidates_rotation_rad_c1), "green")
            )
            x1, y1 = int(g2[0]), int(g2[1])
            ax.add_patch(Circle((int(g2[0]), int(g2[1])), radius=25, color="red"))
            ax.add_patch(
                arrow(x1, y1, length, math.radians(candidates_rotation_rad_c2), "green")
            )
            fig.savefig(
                os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"step_{iter_number}_star.png")
            )
            plt.close()

        temp_map = temp_map_copy.copy()
        temp_map2 = temp_map2_copy.copy()
        temp_map, temp_map2, count_2, o_count = update_map_check(
            runner,
            temp_map,
            temp_map2,
            world_lines2,
            grid_lines2,
            bounds,
            pivot_c2,
            scene_name,
            iter_number,
            iter,
        )
        if plot_flag:
            fig, ax = plt.subplots()
            ax.set_title(scene_name + " " + str(pivot_c2[1]))
            ax.imshow(temp_map, cmap="gray")
            fig.savefig(
                os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"update_map_{iter}_2.png")
            )
            plt.close()
        overlap_score = o_count / count_2
        # print("ind_2_2nd:::"+str(ind_2))
        # print("overlap_score_2nd:::"+str(overlap_score))

        p1_copy = p1.copy()
        p2_copy = p2.copy()
        # print("p1_copy::"+str(p1_copy[-2]))
        # print("p2_copy::"+str(p2_copy[-2]))
        if p1_copy[-2] < 0:
            p1_copy[-2] += 360
        if p2_copy[-2] < 0:
            p2_copy[-2] += 360


        # while p1_copy[-2] - p2_copy[-2] >= 360:
        #     p2_copy[-2] += 360
        # while p1_copy[-2] - p2_copy[-2] <= -360:
        #     p2_copy[-2] -= 360
        if abs(p1_copy[-2] - p2_copy[-2]) > 150 and abs(p1_copy[-2] - p2_copy[-2]) < 210:#> 90:
            overlap_score = 0
            print("pivot: ", len(saved_pose) + 1)
            print("overlap_score: ", overlap_score)
            print("ind_2: ", ind_2)

        if overlap_score > 0.05: #and overlap_score < 0.7:#0.18:
            rel_ele[ind_2] = 1
        # if overlap_score > 0.05:#0.18:
            # rel_ele[ind_2] = 1
        # rel_ele[ind_2] = overlap_score
        if max_overlap < overlap_score:
            max_overlap = overlap_score

        weighted_score = (ALPHA * overlap_score) + (1 - ALPHA) * diff_score
        if max_weight < weighted_score:
            max_weight = weighted_score
            max_rel_ele = rel_ele
    return max_overlap, diff_score, obs_c1, candidate_pose, max_rel_ele


def pick_best_candidate(
    runner: TestRunner,
    original_map: np.ndarray,
    explored_map: np.ndarray,
    stopping_map: np.ndarray,
    candidates: list,
    saved_depth: list,
    saved_color: list,
    saved_pose: list,
    rel_mat: np.ndarray,
    stuck: bool,
    scene_name: str,
    iter_number: int,
    init_unexplored: np.ndarray,
    floor_time: float,
    scene_time: float,
    pivot_time: float,
    pivot_number=1,
    ALPHA=0.2,
    lower_bound=0.2,
    high_bound=0.4,#0.7,# @TODO: Test with 0.4
    plot_flag=False,
    pivot_array=None,
    obs_array=None,
):
    max_score = -np.inf
    max_index = -1
    best_obs = None
    best_pose = None
    best_rel = None

    #@TODO: Test this behavior
    # subset of candidates in cluster we have not explored too much
    if np.random.rand() < 0.01:
        candidates = [c for c in candidates if c[0]["depth_sensor"].shape[0] > 0]
 
    # @TODO: Need optimization here
    for i, candidate in enumerate(candidates):
        obs, pivot, rotation, yaw = candidate

        depth_img = obs["depth_sensor"]
        width, height = depth_img.shape[1], depth_img.shape[0]
        pcd = runner.convert_obs_to_pcd(
            obs,
            pivot,
            rotation,
            scene_name,
            iter_number,
            width=width,
            height=height,
            plot_flag=i,
        )

        point_cloud_array = np.asarray(pcd.points)

        focal_length = 364.86  ## in mm
        rotation_ = [0, yaw, 0]

        # Take slice of depth image at sensor height (center of image)
        slice_ind = height // 2
        depth_vis = np.zeros(depth_img.shape)
        depth_vis[slice_ind] = depth_img[slice_ind]
        array_len = len(depth_vis[slice_ind])

        offset = 4
        if (
            sum(depth_vis[slice_ind]) <= 1000 * offset
            or sum(depth_vis[slice_ind][int(array_len / 2) :]) <= 400 * offset
            or sum(depth_vis[slice_ind][: int(array_len / 2)]) <= 400 * offset
            or sum(depth_vis[slice_ind][int(array_len / 4) : int(array_len * 3 / 4)])
            <= 400 * offset
        ):
            continue

        if plot_flag:
            # Compare slice and original depth
            plt.imshow(depth_vis)
            plt.savefig(
                os.path.join(
                "temp", "More_vis", str(scene_name), str(iter_number), f"test_slice_{i}.png"
                )
            )
            plt.close()

        pose = list(pivot.copy().squeeze())
        pose.extend(rotation_)


        (
            overlap_score,
            diff_score,
            cloud_obs,
            cloud_pose,
            rel_ele,
        ) = intersect_score_simple(
            runner,
            original_map,
            explored_map,
            stopping_map,
            candidate,
            pose,
            saved_depth,
            saved_color,
            saved_pose,
            scene_name,
            iter_number,
            i,
            init_unexplored,
            floor_time,
            scene_time,
            pivot_time,
            upper_threshold=0.5,
            pivot_array=pivot_array,
            obs_array=obs_array,
        )
        #### Overlap not too large and not too small

        if len(saved_depth) != 0 and overlap_score > high_bound and not stuck:
            continue

        weighted_score = (ALPHA * overlap_score) + (1 - ALPHA) * diff_score


        if (max_score < weighted_score) and overlap_score < high_bound:
            max_score = weighted_score
            max_index = i
            best_obs = cloud_obs
            best_pose = cloud_pose
            best_rel = rel_ele

    # print("max_index::"+str(max_index))
    if max_index == -1 or diff_score == 0:
        return None, None, saved_depth, saved_color, saved_pose, rel_mat, False

    #### Modification before saved
    cam2base_h = 1.5
    best_pose[1] += cam2base_h
    temp = best_pose[1].copy()
    best_pose[1] = best_pose[2]
    best_pose[2] = temp
    best_obs["color_sensor"] = best_obs["color_sensor"][:, :, :3]

    saved_depth.append(best_obs["depth_sensor"])
    saved_color.append(best_obs["color_sensor"])
    saved_pose.append(best_pose)

    saved_obs_dir = os.path.join("temp", "More_vis", str(scene_name), str(iter_number), "saved_obs")

    os.makedirs(saved_obs_dir, exist_ok=True)

    if (np.sum(stopping_map) / np.sum(init_unexplored)) < 0.3:

        np.save(os.path.join(saved_obs_dir, "saved_dep.npy"),saved_depth) #

        np.save(os.path.join(saved_obs_dir, "saved_color.npy"), saved_color)#

        np.save(os.path.join(saved_obs_dir, "saved_pose.npy"),saved_pose)#


    ### saving for visualization
    saved_grid_pose = saved_pose.copy()
    for idx, svd_pose in enumerate(saved_grid_pose):
        pivot_ = svd_pose[:3]
        pivot_[2], pivot_[1] = pivot_[1], pivot_[2]
        grid_pivot = runner.get_grid_pos(pivot_)
        saved_grid_pose[idx] = grid_pivot


    if (np.sum(stopping_map) / np.sum(init_unexplored)) < 0.3:
        np.save(os.path.join(saved_obs_dir, "saved_grid_pose.npy"),saved_grid_pose)
        # print("------> Iter number: {} saved_grid_pose: {} with saved_pose:{}".format(str(iter_number), str(saved_grid_pose), str(saved_pose)))


    if len(rel_mat) != 0:
        rel_mat_copy = np.eye(len(saved_depth), dtype=np.double)
        rel_mat_copy[: len(rel_mat), : len(rel_mat)] = rel_mat
        print("best_rel: ", best_rel)
        print("max_score: ", max_score)
        rel_mat_copy[-1, :] = best_rel * max_score
        rel_mat_copy[:, -1] = best_rel.transpose() * max_score
        rel_mat = rel_mat_copy
    else:
        rel_mat = np.eye(1, dtype=np.double)


    # print("Candidate #:::" + str(len(rel_mat[0])))
    # print("rel_mat_after:::\n" + str(rel_mat))
#     print(saved_pose)
    # display using visualize_graph
    rel_mat_dir = os.path.join("temp", "More_vis", str(scene_name), str(iter_number),"rel_mats")
    os.makedirs(rel_mat_dir, exist_ok=True)
    visualize_graph(
        runner,
        rel_mat,
        runner.get_top_down(height=saved_pose[0][2] - 1.5),
        saved_pose,
        output_file=
        os.path.join(
        rel_mat_dir, f"rel_mat_{len(rel_mat[0])}.png"
        )
    )

    if (np.sum(stopping_map) / np.sum(init_unexplored)) < 0.3:
        np.save(os.path.join(saved_obs_dir, "rel_mat.npy"), rel_mat)

    plot_flag = True
    if plot_flag:
        image_pil = Image.fromarray(np.array(best_obs["color_sensor"]).astype(np.uint8))
        image_pil.save(
            os.path.join(
                "temp", "More_vis", str(scene_name), str(iter_number), "saved_obs", f"best_color_{len(saved_depth) - 1}.png"
            ), "PNG"
        )
        depth_array = np.array(best_obs["depth_sensor"])
        depth_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
        depth_scaled = (depth_normalized * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_scaled)
        depth_image.save(
            os.path.join(
                "temp", "More_vis", str(scene_name), str(iter_number), "depth", f"best_depth_{len(saved_depth) - 1}.png"
            ), "PNG"
        )
        fig, ax = plt.subplots()
        ax.imshow(np.array(best_obs["depth_sensor"]))

        fig.savefig(
            os.path.join(
                "temp", "More_vis", str(scene_name), str(iter_number), "depth", f"plt_best_depth_{len(saved_depth) - 1}.png"
            )
        )
        plt.close()

    if plot_flag:
        fig, ax = plt.subplots()
        turbo = plt.get_cmap("turbo")
        cNorm = color_.Normalize(vmin=0, vmax=len(saved_pose))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=turbo)
        scalarMap.set_array([])

        for idx, svd_pose in enumerate(saved_pose):
            # print("svd_pose::"+str(svd_pose))
            if idx == 0:
                ax.set_title(f"{scene_name} {svd_pose[2] - 1.5}")
                ax.imshow(runner.get_top_down(height=svd_pose[2] - 1.5))
            length = 100
            cand_radians = svd_pose[-2]
            # print("cand_radians:::"+str(cand_radians))
            # assert()
            pivot_ = svd_pose[:3]
            pivot_[2], pivot_[1] = pivot_[1], pivot_[2]
            grid_pivot = runner.get_grid_pos(pivot_)
            x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])
            colorVal = scalarMap.to_rgba(idx)
            ax.add_patch(
                Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red")
            )
            ax.add_patch(arrow(x0, y0, length, math.radians(cand_radians), colorVal))

        fig.colorbar(cmx.ScalarMappable(norm=cNorm, cmap=turbo), ax=ax)
        fig.savefig(
            os.path.join(
                "temp", "More_vis", str(scene_name), str(iter_number), "all_best_locs.png"
            )
        )
        plt.close()
    return max_index, max_score, saved_depth, saved_color, saved_pose, rel_mat, True

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

def update_map_ray(
    explored_map: np.ndarray,
    stopping_map: np.ndarray,
    count_set,
    grid_pcl: Tuple[int,int,float,float],
    grid_pivot: List[int],
    step_size=1,
    view_range=80000,
    plot_flag=True,
) -> Tuple[np.ndarray, np.ndarray, set]:
    """
    This function is used to update the map with a ray from the pivot point to the point cloud grid.
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
    # explored_map, stopping_map,grid_rays,explored_mask, _,_ = update_map_ray_helper(
    #     explored_map, stopping_map, grid_pcl, grid_pivot, step_size,view_range
    # )
    # grid_pcl = np.array(grid_pcl[2:4])
    # grid_pivot = np.array(grid_pivot)

    # update count map
    count_set.update(
        grid_rays[explored_mask, 0], grid_rays[explored_mask, 1]
    )

    #@TODO: check this
    # if explored_map == 0 then just pass

    # set in bounds to be a boolean mask
    in_bounds = (
        (grid_pcl[0] >= 0)
        & (grid_pcl[0] < explored_map.shape[1])
        & (grid_pcl[1] >= 0)
        & (grid_pcl[1] < explored_map.shape[0])
    )

    in_view_range = LA.norm(grid_pcl -grid_pivot) < view_range

    if in_bounds & in_view_range:
        grid_pcl = grid_pcl.astype(int)
        mask = explored_map[grid_pcl[1], grid_pcl[0]] == 1
        explored_map[grid_pcl[1], grid_pcl[0]] = 0.5 * mask
        stopping_map[grid_pcl[1], grid_pcl[0]] = 0 * mask
        count_set.update(
            grid_rays[explored_mask, 0],
            grid_rays[explored_mask, 1],
        )

    return explored_map, stopping_map, count_set


def update_map_ray_check(
    explored_map: np.ndarray,
    stopping_map: np.ndarray,
    explored_map_copy: np.ndarray,
    grid_pcl: Tuple[int,int,float,float],
    grid_pivot: List[int],
    step_size=1,
    view_range=80000,
    plot_flag=True,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    This function is used to update the map with a ray from the pivot point to the point cloud grid.
    """
    count = 0
    o_count = 0

    explored_map, stopping_map, grid_rays, explored_mask,inbound, in_view_range = update_map_ray_helper(
        explored_map, stopping_map, grid_pcl, grid_pivot,step_size,view_range
    )

    count = explored_mask.sum()

    mask_05 = (
        (explored_map[grid_rays[:, 1], grid_rays[:, 0]] == 0.5)
        & inbound
        & in_view_range
    )
    count += mask_05.sum()

    o_count = (
        (
            explored_map_copy[grid_rays[:, 1], grid_rays[:, 0]]
            == 0.5
        )
        & inbound
        #& in_view_range
    )
    o_count = o_count.sum()

    # clip indices to range of valid indices
    pcl_indices = np.clip(
        np.array([int(grid_pcl[1]), int(grid_pcl[0])]),
        0,
        np.subtract(explored_map.shape, 1),
    )

    if explored_map[pcl_indices[0]][pcl_indices[1]] == 1:
        explored_map[pcl_indices[0]][pcl_indices[1]] = 0.5
        stopping_map[pcl_indices[0]][pcl_indices[1]] = 0
        count += 1
    elif explored_map[pcl_indices[0]][pcl_indices[1]] == 0.5:
        count += 1
    if explored_map_copy[pcl_indices[0]][pcl_indices[1]] == 0.5:
        o_count += 1

    return explored_map, stopping_map, count, o_count



def update_map(
    runner,
    explored_map,
    stopping_map,
    world_lines,
    grid_lines,
    bounds,
    pivot,
    scene_name,
    iter_number,
    pivot_number,
    init_unexplored,
    floor_time,
    scene_time,
    pivot_time,
    plot_flag=False,
):
    count_set = set()
    pcd_grid = grid_lines
    grid_pivot = runner.get_grid_pos(pivot)
    explored_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0.5
    stopping_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0
    count_set.add((int(grid_pivot[1]), int(grid_pivot[0])))

    for idx, world_line in enumerate(world_lines
        #tqdm(world_lines, desc=f"update_map_{scene_name}_iter{iter_number}")
    ):
        explored_map, stopping_map, count_set = update_map_ray(
            explored_map,
            stopping_map,
            count_set,
            pcd_grid[idx],
            grid_pivot,
        )
    if plot_flag:
        fig, ax = plt.subplots()
        ax.set_title(scene_name + " " + str(pivot[1]))
        ax.imshow(explored_map, cmap="gray")
        ax.add_patch(
            Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red")
        )

        this_pivot_time = time.time() - pivot_time
        total_scene_time = time.time() - scene_time
        total_floor_time = time.time() - floor_time

        coverage = round(
            abs(1 - (np.sum(stopping_map) / np.sum(init_unexplored))) * 100, 3
        )

        # convert total_floor_time and total_scene_time to minutes
        total_floor_time, total_scene_time = (
            total_floor_time / 60,
            total_scene_time / 60,
        )

        metrics = f"pivot_time: {this_pivot_time:>3.2f}s\nscene_time: {total_scene_time:>3.2f}min\nfloor_time: {total_floor_time:>3.2f}min"

        # plt.text(1, -0.1, metrics, ha='right', va='bottom', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        plt.text(
            0.99,
            0.005,
            metrics,
            ha="right",
            va="bottom",
            transform=plt.gcf().transFigure,
            fontsize="small",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
        )

        # set title to be scene, pivot number, then coverage
        ax.set_title(f"{scene_name}  iter{pivot_number} | {coverage}%")


        fig.savefig(os.path.join(
            "temp", "More_vis", scene_name, str(iter_number), f"update_map_{pivot_number}_fc.png"
        ))

        plt.close()

    return explored_map, stopping_map, len(count_set)


def update_map_check(
    runner,
    explored_map,
    stopping_map,
    world_lines,
    grid_lines,
    bounds,
    pivot,
    scene_name,
    iter_number,
    pivot_number,
    plot_flag=False,
):
    count = 0
    o_count = 0
    pcd_grid = grid_lines
    grid_pivot = runner.get_grid_pos(pivot)
    explored_map_copy = copy(explored_map)

    if explored_map[int(grid_pivot[1])][int(grid_pivot[0])] == 1:
        explored_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0.5
        stopping_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0
    elif explored_map[int(grid_pivot[1])][int(grid_pivot[0])] == 0.5:
        # print("explored_map[int(grid_pivot[0])][int(grid_pivot[1])] ::"+str(explored_map[int(grid_pivot[0])][int(grid_pivot[1])] ))
        o_count += 1
    count += 1

    for idx, world_line in enumerate(world_lines
        #tqdm(world_lines, desc=f"update_map_check_{scene_name}_iter{iter_number}")
    ):  ##### Need optmization here
        explored_map, stopping_map, count_temp, o_count_temp = update_map_ray_check(
            explored_map,
            stopping_map,
            explored_map_copy,
            pcd_grid[idx],
            grid_pivot
        )
        # ax.add_patch(Circle((int(grid_pcl[0]), int(grid_pcl[1])), radius=5, color='red'))
        count += count_temp
        o_count += o_count_temp
    if plot_flag:
        fig, ax = plt.subplots()
        ax.set_title(scene_name + " " + str(pivot[1]))
        ax.imshow(explored_map, cmap="gray")

        fig.savefig(
            os.path.join(
            "temp", "More_vis", scene_name, str(iter_number), f"update_map_{pivot_number}.png"
            )
        )
        plt.close()

    return explored_map, stopping_map, count, o_count


def arrow(x0, y0, length, angle, color):
    x1 = math.cos(np.pi / 2 + angle) * length
    y1 = math.sin(np.pi / 2 + angle) * length

    return Arrow(x0, y0, x1, -y1, width=100.0, color=color)

def depth2grid(i, angle_range, pivot_, depth_slice, x0, y0):
    rad = angle_range[i]

    del_x = depth_slice[i] * math.cos(np.pi / 2 + rad)
    del_y = depth_slice[i] * math.sin(np.pi / 2 + rad)

    x1, y1 = pivot_[0] + del_x, pivot_[2] - del_y

    # then convert to grid
    px = x0 + (del_x * 90)
    py = y0 - (del_y * 90)

    return (pivot_[0], pivot_[2], x1, y1), (x0, y0, px, py)


def draw_sight_rays(
    runner,
    candidates,
    best_index,
    height,
    scene,
    iter_number,
    pivot_number,
    plot_flag=False,
):
    obs_, pivot_, rotation_, cand_rad = candidates[best_index]

    depth_img = obs_["depth_sensor"]

    # Take slice of depth image at sensor height (center of image)
    slice_ind = depth_img.shape[0] // 2
    depth_vis = np.zeros(depth_img.shape)
    depth_vis[slice_ind] = depth_img[slice_ind]

    # Assign each pixel to angle
    left_angle = math.radians(cand_rad - 45)
    right_angle = math.radians(cand_rad + 45)

    angle_range = np.linspace(left_angle, right_angle, depth_img.shape[1])
    depth_slice = depth_img[slice_ind]

    # Pivot
    x0, y0 = runner.get_grid_pos(pivot_)

    patch = Circle((x0, y0), radius=25, color="red")

    # grid conversion
    bounds = runner._sim.pathfinder.get_bounds()
    meters_per_pixel = 0.01

    # collect lines
    world_lines = []
    grid_lines = []

    # Plot on topdown
    if plot_flag:
        topdown = runner.get_top_down(height=height)
        ax.imshow(topdown)
        ax.add_patch(patch)

    angle_range = angle_range[::-1]

    i_range = np.arange(len(angle_range))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        ar_result = [
            executor.submit(
                depth2grid, i, angle_range, pivot_, depth_slice, x0, y0
            )
            for i in i_range
        ]
        # retrieve the return value results
        for ar in concurrent.futures.as_completed(ar_result):
            world_line, grid_line = ar.result()
            world_lines.append(world_line)
            grid_lines.append(grid_line)

    # with multiprocessing.Pool(processes=4) as pool:  # optimization example
    #     ar_result = [
    #         pool.apply_async(
    #             depth2grid, args=(i, angle_range, pivot_, depth_slice, x0, y0)
    #         )
    #         for i in i_range
    #     ]
    #     # retrieve the return value results
    #     for ar in ar_result:
    #         world_line, grid_line = ar.get()
    #         world_lines.append(world_line)
    #         grid_lines.append(grid_line)

    if plot_flag:
        fig, ax = plt.subplots()
        ax.imshow(topdown)
        ax.add_patch(arrow(x0, y0, 100, math.radians(cand_rad), "red"))
        for grid_line in grid_lines:
            x0, y0, px, py = grid_line
            ax.plot((x0, px), (y0, py), color="blue")

        fig.savefig(
            os.path.join(
                "temp", "More_vis", scene, str(iter_number), f"update_test_{pivot_number}.png"
            )
        )
        plt.close()
    return world_lines, grid_lines


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
                            pivot_array.append([i, j])
                    elif is_right_col:
                        if (
                            original_map[i][j - 1] == 0
                            or original_map[i + 1][j] == 0
                        ):
                            pivot_array.append([i, j])
                    else:
                        if (
                            original_map[i + 1][j] == 0
                            or original_map[i][j - 1] == 0
                        ):
                            pivot_array.append([i, j])
                elif is_bottom_row:
                    if is_left_col:
                        if(
                            original_map[i][j + 1] == 0
                            or original_map[i - 1][j] == 0
                        ):
                            pivot_array.append([i, j])
                    elif is_right_col:
                        if (
                            original_map[i][j - 1] == 0
                            or original_map[i - 1][j] == 0
                        ):
                            pivot_array.append([i, j])
                    else:
                        if (
                            original_map[i - 1][j] == 0
                            or original_map[i][j - 1] == 0
                            or original_map[i][j + 1] == 0
                        ):
                            pivot_array.append([i, j])
                else:
                    if is_left_col:
                        if (
                            original_map[i][j + 1] == 0
                            or original_map[i + 1][j] == 0
                            or original_map[i - 1][j] == 0
                        ):
                            pivot_array.append([i, j])
                    elif is_right_col:
                        if (
                            original_map[i + 1][j] == 0
                            or original_map[i - 1][j] == 0
                            or original_map[i][j - 1] == 0
                        ):
                            pivot_array.append([i, j])
                    else:
                        # corners = ['lu', 'ld', 'ru', 'rd', 'l', 'u', 'r', 'd']
                        if original_map[i - 1][j] == 0:
                            if original_map[i][j - 1] == 0:
                                pivot_array.append([i, j, 0])
                            elif original_map[i][j + 1] == 0:
                                pivot_array.append([i, j, 2])
                            else:
                                pivot_array.append([i, j, 5])
                        elif original_map[i + 1][j] == 0:
                            if original_map[i][j - 1] == 0:
                                pivot_array.append([i, j, 1])
                            elif original_map[i][j + 1] == 0:
                                pivot_array.append([i, j, 3])
                            else:
                                pivot_array.append([i, j, 7])
                        elif original_map[i][j - 1] == 0:
                            pivot_array.append([i, j, 4])
                        elif original_map[i][j + 1] == 0:
                            pivot_array.append([i, j, 6])
                        # check if any of the 8 neighbors are 0
                        elif (
                            original_map[i - 1][j - 1] == 0
                            or original_map[i - 1][j + 1] == 0
                            or original_map[i + 1][j - 1] == 0
                            or original_map[i + 1][j + 1] == 0
                        ):
                            pivot_array.append([i, j, 7])
    return np.array(pivot_array,dtype=int)  # pivot array is a list of [x,y,corner] where corner is 0,1,2,3,4,5,6,7


def algorithm(
    runner: TestRunner,
    floor_maps: list,
    heights: list,
    scene: str,
    SCENE_THRESHOLD: float,
    resume_flag=False,
    floor_flag=0,
) -> None:
    scene_time = time.time()
    sample_n_candidate = 6
    plot_flag = False

    # Keeping tabs on
    bounds = runner._sim.pathfinder.get_bounds()

    # Check directories existence
    vis_dir = os.path.join("temp/More_vis", str(scene))
    batch_dir = os.path.join("temp/batch_indx", str(scene))
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(batch_dir, exist_ok=True)

    # start/restart at floor_flag

    # @TODO: Need optimization here
    for i, floor in enumerate(floor_maps[floor_flag:]):
        floor_time = time.time()

        # Create directories
        os.makedirs(os.path.join(vis_dir, str(i)), exist_ok=True)
        os.makedirs(os.path.join(batch_dir, str(i)), exist_ok=True)

        # Get the floor map
        original_map = np.array(floor, copy=True,dtype=float)

        # @TODO: see if this is right
        init_unexplored = np.sum(original_map)

        if np.sum(init_unexplored) < 200:
            continue
        if resume_flag == False:
            explored_map = original_map.copy()
            stopping_map = original_map.copy()
            saved_depth = []
            saved_color = []
            saved_pose = []
            rel_mat = []
        else:
            saved_obs_dir = os.path.join(vis_dir, str(i), "saved_obs")
            try:
                explored_map = np.load(os.path.join(saved_obs_dir, "explored_map.npy"))

                stopping_map = np.load(os.path.join(saved_obs_dir, "stopping_map.npy"))

                saved_depth = np.load(os.path.join(saved_obs_dir, "saved_dep.npy"))

                saved_color= np.load(os.path.join(saved_obs_dir, "saved_color.npy"))

                saved_pose = np.load(os.path.join(saved_obs_dir, "saved_pose.npy"))

                rel_mat = np.load(os.path.join(saved_obs_dir, "rel_mat.npy"))


                if np.sum(stopping_map) / np.sum(init_unexplored) < SCENE_THRESHOLD:
                    continue
                if len(rel_mat[0]) > 20:
                    continue
            except:
                explored_map = original_map.copy()
                stopping_map = original_map.copy()
                saved_depth = np.array([])
                saved_color = np.array([])
                saved_pose = np.array([])
                rel_mat = np.array([])

        # skip if explore size is small
        if np.sum(init_unexplored) < 200:
            continue
        # Setup threshold calculations

        # When we explore, we turn the pixels
        # from true to false
        # So to check if we meet threshold,
        # sum(original_map) / init_unexplored >= 0.05
        step = 0
        trial_limit = 3
        stuck = False  # Initially unstuck
        rand = True  # Every new floor start by pick random
        pivot_number = -1

        ##########

        #
        if np.sum(stopping_map) / np.sum(init_unexplored) < SCENE_THRESHOLD:
            continue
        

        # print("starting pivot array")
        pivot_array = pivot_candidates(original_map)
        # print("finished creating pivot array")


        if plot_flag:
            fig, ax = plt.subplots()
            ax.imshow(original_map, cmap="gray")
            colors_ = {
            0: "red",
            1: "blue",
            2: "green",
            3: "yellow",
            4: "m",
            5: "k",
            6: "purple",
            7: "c",
            }
            # @TODO: Need optimization here
            for pivot_ in pivot_array:
                pivot_color = colors_.get(pivot_[2], "black")
                patch = Circle((int(pivot_[1]), int(pivot_[0])), radius=1, color=pivot_color)
                ax.add_patch(patch)
            fig.savefig(f"temp/More_vis/{scene}/{i}/potential_pivot_{pivot_number + 1}.png")
            plt.close()

        def in_map(x, y, theta):
            new_x = x + 50 * np.cos(theta)
            new_y = y + 50 * np.sin(theta)
            new_x_ = x + 100 * np.cos(theta)
            new_y_ = y + 100 * np.sin(theta)
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

        obs_array = set()
        downsample = 10  # degree
        # pivot_array = pivot_array[::5]
        scaler_factor = int(len(pivot_array) / 120)
        if scaler_factor > 50:
            scaler_factor = 50


        for p in pivot_array:  ##### Need optmization here
            # downsample=10
            angle_r = np.arange(0, 360, downsample)
            for ang in angle_r:  ##### Need optmization here
                if (
                    in_map(p[0], p[1], ang)
                    # and in_map(p[0], p[1], ang + 5)
                    # and in_map(p[0], p[1], ang - 5)
                ):
                    obs_array.add((p[0], p[1], ang))

        obs_array = np.array(list(set(obs_array)))  # Convert to NumPy array and remove duplicates

        obs_array_index = np.arange(len(obs_array), dtype=int)
        obs_array[:, [0, 1]] = obs_array[:, [1, 0]]  # Swap x and y positions directly in the NumPy array
        seen_array = []

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(original_map, cmap="gray")
        ax.scatter(obs_array[:, 0], obs_array[:, 1], c="r", s=1)

        # Save the plot to a file
        plt.savefig(os.path.join(vis_dir, str(i), "pivot_points.png"))
        plt.close()


        while (np.sum(stopping_map) / np.sum(init_unexplored) >= SCENE_THRESHOLD) and (
            len(obs_array_index) > 0
        ): #and (time.time() - floor_time) / 60 <= 90:
            # print(
            #     "(np.sum(stopping_map) / np.sum(init_unexplored)::"
            #     + str((np.sum(stopping_map) / np.sum(init_unexplored)))
            # )
            # print(f"\npicking new pivot | stuck={stuck} rand={rand}")
            pivot_time = time.time()

            pivot_number += 1


            # Picking Random Pivots
            final_obs_inds = []
            obs_inds = np.random.choice(
                obs_array_index, sample_n_candidate, replace=False
            )


            while len(final_obs_inds) < 6:
                obs_inds = obs_inds.astype(int)

                obs_coords = [obs_array[x][:3] for x in obs_inds if x not in seen_array]

            #     pivot_array_tmp = pivot_array[:, :3]
                pivot_array_tmp = [pivot_array[x][:3] for x in range(len(pivot_array))]

                dist_matrix = distance_matrix(obs_coords, pivot_array_tmp)

                dist_matrix = dist_matrix[:, :-1]
                mask = np.all(dist_matrix >= 100, axis=1)
                # randomly allow some obs_inds to be too close to pivot
                if np.random.rand() < 0.01:
                    mask = np.any(dist_matrix > 75, axis=1)

                obs_inds = obs_inds[mask]
                if len(obs_inds) > 0:
                    final_obs_inds.extend(obs_inds)
                    obs_inds = []

                obs_inds = np.random.choice(obs_array_index, 1, replace=False)
            obs_inds = final_obs_inds[:6]

            # print("finished sampling pivots")

            candidates = []

            def process_observation(
                runner: TestRunner,
                heights: list,
                i,
                obs_array: np.ndarray,
                obs_ind,
            ) -> Tuple[float, np.ndarray, np.ndarray, float]:
                g1 = obs_array[obs_ind]
                p1 = np.array(runner.get_pos_grid(g1[:2], heights[i], meters_per_pixel=0.01))

                rotation = R.from_euler("xyz", [0, g1[2], 0], degrees=True).as_quat() 
  
                return heights[i], p1, rotation, g1[2]


            with concurrent.futures.ThreadPoolExecutor(
                max_workers=(3 * n_cores) // 4
            ) as executor:
            #     map_copy = original_map.copy()

                features = [
                    executor.submit(
                        process_observation,
                        runner,
                        heights,
                        i,
                        obs_array,
                        obs_ind,
                    )
                    for obs_ind in obs_inds
                ]
                for f in concurrent.futures.as_completed(features):
                    (
                        height,
                        pivot,
                        rotation,
                        angle,
                    ) = f.result()
                    #@TODO: THis might increase gpu utilization but is not used
                    # pivot_grid = runner.get_grid_pos(pivot)
                    # ensure point is reachable and in the obs_array_index
                    new_pivot, _ = runner.get_random_point_near(
                        height,
                        grid=original_map,
                        circle_center=pivot,
                        radius_grid=50,
                        pivot_list=pivot_array,
                        pivot_indicies=obs_array_index,
                        scalar_factor=50,
                    )

                    obs = runner.get_rgbd_from_pose(new_pivot, rotation)
                    # convert new_pivot to proper data type
                    # list object has no attribute squeeze
                    new_pivot = np.asarray(new_pivot)

                    candidate = (obs, pivot, rotation, angle)
                    candidates.append(candidate)

            # Plot candidates on the map
            if plot_flag:
                  fig, ax = plt.subplots()
                  ax.set_title(f"Scene {scene} | Height {heights[i]:.2f}")
                  ax.imshow(explored_map, cmap="gray")
                  for ind, candidate in enumerate(candidates):
                        (_, pivot, _, candidates_rotation_angle) = candidate

                        candidates_rotation_rad = math.radians(candidates_rotation_angle)

                        grid_pivot = runner.get_grid_pos(pivot)
                        ax.add_patch(
                              Circle(
                                    (int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red"
                              )
                        )
                        x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])

                        ax.add_patch(
                              arrow(x0, y0, 100, candidates_rotation_rad, colors[ind])
                        ) # colors = ['b', 'r', 'k', 'm']
                  # make directory if not exist
                  directory_path = os.path.join("temp", "More_vis", scene, str(i), "candidates")
                  os.makedirs(directory_path, exist_ok=True)

                  # Save the figure
                  file_path = os.path.join(
                  directory_path, f"candidates_{pivot_number}.png"
                  )
                  fig.savefig(file_path)

                  # Close the figure to release resources
                  plt.close(fig)

            #plot candidates
            if plot_flag:
                fig, ax = plt.subplots()
                ax.set_title(scene + " " + str(heights[i]))
                ax.imshow(explored_map, cmap="gray")
                for ind, candidate in enumerate(candidates):
                    (_, pivot, _, candidates_rotation_angle) = candidate
                    candidates_rotation_rad = math.radians(candidates_rotation_angle)
                    grid_pivot = runner.get_grid_pos(pivot)

                    ax.add_patch(
                        Circle(
                            (int(grid_pivot[0]), int(grid_pivot[1])),
                            radius=25,
                            color="red",
                        )
                    )
                    x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])
                    ax.add_patch(
                        arrow(x0, y0, 100, candidates_rotation_rad, colors[ind])
                        )  # colors = ['b', 'r', 'k', 'm']

                fig.savefig(os.path.join("temp", "More_vis", scene, str(i), f"candidates_GA_{step}", "candidates.png"))
                plt.close()


                for ind, candidate in enumerate(candidates):
                    (obs, pivot, rotation, candidates_rotation_angle) = candidate
                    fig, ax = plt.subplots()

                    ax.set_title(f"{scene}_{color_map[colors[ind]]}")
                    ax.imshow(np.array(obs["color_sensor"]))

                    fig.savefig(
                        os.path.join("temp", "More_vis", scene, str(i), f"obs_GA_{step}_{color_map[colors[ind]]}.png")
                    )
                    plt.close()

            # Pick the best
            (
                best_index,
                best_score,
                saved_depth,
                saved_color,
                saved_pose,
                rel_mat,
                selected_flag,
            ) = pick_best_candidate(
                runner,
                original_map,
                explored_map,
                stopping_map,
                candidates,
                saved_depth,
                saved_color,
                saved_pose,
                rel_mat,
                stuck,
                scene_name=scene,
                iter_number=i,
                init_unexplored=init_unexplored,
                floor_time=floor_time,
                scene_time=scene_time,
                pivot_time=pivot_time,
                pivot_number=step,
                pivot_array=pivot_array,
                obs_array=obs_array_index,
            )
            # stuck = False
            # unstuck yourself
            if stuck == True:
                stuck = False

            # get only indicies of unexplored pixels (best_index is the index of the best candidate)
            # add to seen
            # @TODO Test this
            if best_index is not None:
                seen_array.append(obs_inds[best_index])
            else:
                trial_limit -= 1
                stuck = True if trial_limit == 0 else False
                continue


            if plot_flag:
                fig, ax = plt.subplots()
                ax.set_title(f"Scene {scene} {pivot[1]}")
                ax.imshow(runner.get_top_down(height=pivot[1]))
                _, pivot, _, cand_degree = candidates[best_index]
                grid_pivot = runner.get_grid_pos(pivot)

                cand_radians = math.radians(cand_degree)
                x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])
                # print("cand_degree_temp::"+str(cand_degree))
                ax.add_patch(
                    Circle(
                        (int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red"
                    )
                )
                ax.add_patch(arrow(x0, y0, 100, cand_radians, "green"))

                fig.savefig(
                    os.path.join(
                        "temp",
                        "More_vis",
                        scene,
                        str(i),
                        f"step_{step}_best.png",
                    )
                )
                plt.close()

            world_lines, grid_lines = draw_sight_rays(
                runner, candidates, best_index, pivot[1], scene, i, step
            )
            obs_, pivot_, rotation_, cand_rad = candidates[best_index]
            explored_map, stopping_map, _ = update_map(
                runner,
                explored_map,
                stopping_map,
                world_lines,
                grid_lines,
                bounds,
                pivot_,
                scene,
                i,
                step,
                init_unexplored,
                floor_time,
                scene_time,
                pivot_time,
                plot_flag=True,
            )
            # assert()

            # Save explored and stopping maps
            if (np.sum(stopping_map) / np.sum(init_unexplored)) < 0.3:
                np.save(os.path.join("temp", "More_vis", scene, str(i), "saved_obs", "explored_map.npy"), explored_map)
                np.save(os.path.join("temp", "More_vis", scene, str(i), "saved_obs", "stopping_map.npy"), stopping_map)

            ## Visualize the difference between init_unexplored and explored_map
            step += 1
            # obs_array_index = np.array([obs_inds[best_index]])


            #@TODO: See if this is correct
            exclude_array = np.arange(
                obs_inds[best_index] - scaler_factor * 36,
                obs_inds[best_index] + scaler_factor * 36,
                0.5,
            )
            obs_array_index = np.setdiff1d(obs_array_index, exclude_array)
            obs_array_index = np.setdiff1d(obs_array_index, obs_inds[best_index])
            # exclude all points around the best_index
            obs_array_index = np.setdiff1d(obs_array_index, [best_index])



def delete_folder_contents(folder: str) -> None:
    folder_path = Path(folder)
    for item in folder_path.glob("*"):
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"Failed to delete {item}. Reason: {e}")

def process_scene(scene, floor_heights,scene_dir,resume_flag, scene_flag=None, floor_flag=None,SCENE_THRESHOLD=0.15):
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs(os.path.join("temp/batch_indx", str(scene)), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "visualization"), exist_ok=True)
    # build the scene using scene_flag
    if scene_flag != None and scene != scene_flag:
        scene_name = scene_flag + ".glb"
    else:
        scene_name = scene + ".glb"

    default_sim_settings["scene"] = f"gibson/gibson/{scene_name}"
    
    runner = TestRunner(default_sim_settings)

    # Navmesh Settings
    navmesh_settings = habitat_sim.NavMeshSettings()
    use_custom_settings = False

    navmesh_settings.set_defaults()
    if use_custom_settings:
        navmesh_settings.agent_height = 1.5
        navmesh_settings.agent_max_climb = 0.2
        navmesh_settings.filter_walkable_low_height_spans = False
        navmesh_settings.region_merge_size = 10
        navmesh_settings.verts_per_poly = 8.0

    navmesh_success = runner._sim.recompute_navmesh(
        runner._sim.pathfinder, navmesh_settings
    )

    assert navmesh_success, "Failed to build navmesh"


    floor_maps = []  # topdown maps of each floor


    # Plot topdown maps
    fig, ax = plt.subplots()
    for i, height in enumerate(floor_heights):
        floor_maps.append(runner.get_top_down(height=height))
        ax.set_title(scene + " " + str(height))
        ax.imshow(floor_maps[i])
        fig.savefig(os.path.join(scene_dir, f"topdown_{i}.png"))
        plt.close()

        # saving full image size for visualization
        img_pil = Image.fromarray(floor_maps[i])
        img_pil.save(os.path.join(scene_dir, "visualization", f"topdown_{i}.png"))

    print("Working on Scene: " + str(scene))
    print("Seed: ", runner._sim_settings["seed"])

    algorithm(
        runner,
        floor_maps,
        floor_heights,
        scene,
        SCENE_THRESHOLD,
        resume_flag=resume_flag,
        floor_flag=floor_flag,
    )

    runner._sim.close()


def main(start=0, end=None, resume_flag=False, scene_flag=None, floor_flag=None):
    os.makedirs("temp/More_vis", exist_ok=True)
    os.makedirs("temp/batch_indx", exist_ok=True)

    SCENE_THRESHOLD = 0.15  # equivalent to 85% coverage
    gibson_fp = "igibson_floor_heights.json"

    with open(gibson_fp, "r") as j:
        json_file = json.load(j) #list of dicts

    # Determine which scenes are present in our dataset
    available_scenes = {filename[14:-4] for filename in glob.glob("gibson/gibson/*.glb")}

    available_scenes = sorted(list(available_scenes))
    # available_scenes.remove('Cooperstown')

    start = max(0, start)
    end = len(available_scenes) if end == None else min(end, len(available_scenes))
    available_scenes = available_scenes[start:end]
    
#     print("available_scenes::", available_scenes)

    start_time=time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context= mp.get_context('spawn')) as executor:
        futures=[]
        for scene in json_file:
            if scene in available_scenes:
                scene_dir = os.path.join("temp/More_vis", str(scene))
                futures.append(executor.submit(process_scene, scene, json_file[scene],scene_dir,resume_flag,scene_flag, floor_flag,SCENE_THRESHOLD))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(e)

    print("Total time taken: ", time.time()-start_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=bool,
        default=0,
        help="If present, restore checkpoint and resume training",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index of the scene")
    parser.add_argument("--end", type=int, default=None, help="End index of the scene")
    parser.add_argument("--scene", type=int, default=None, help="scene to run")
    parser.add_argument(
        "--floor",
        type=int,
        default=None,
        help="floor of a scene to run, indexed from 0",
    )
    FLAGS = parser.parse_args()
    main(FLAGS.start, FLAGS.end, FLAGS.resume, FLAGS.scene, FLAGS.floor)
