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
from visual import plot_topdown, process_observation, plot_obs_array
from visual import plot_candidate, plot_candidate_GA, plot_topdown
from visual import plot_update_map, plot_update_map_check, plot_update_candidate
from visual import plot_step_star, plot_update_map_iter, plot_intersect

from multiprocessing import cpu_count
import multiprocessing as mp
import time
from PIL import Image
from pathlib import Path
from typing import Tuple, List
from utils import pivot_candidates, get_candidate_svd, in_map

# file handling
import os, shutil
import json
import math
import cv2
from tqdm import tqdm
import glob
import torch

import glfw
from OpenGL.GL import *

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
print(f"Number of Logical CPU cores: {n_cores}")
# print("torch version",torch.__version__)

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
        drawKeypoints = cv2.drawKeypoints(gray_cdt, keyPoints, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return False
    return True


def intersect_score_simple(runner, original_map, explored_map, stopping_map, candidate, candidate_pose, saved_depth, saved_color, saved_pose,
                            scene_name, iter_number, iter, init_unexplored, floor_time, scene_time, pivot_time, ALPHA=0.1, plot_flag=False, index = 0, downsample_rate=8):
    
    max_overlap = -np.inf
    max_weight = -np.inf
    cam2base_h = 1.5 # The height of the camera
    bounds = runner._sim.pathfinder.get_bounds()
    
    (obs_candidate, pivot_candidate, rotation_candidate, candidates_rotation_rad_candidate) = candidate
    grid_candidate = runner.get_grid_pos(pivot_candidate)
    rotation_3d_candidate = [0, candidates_rotation_rad_candidate, 0]

    cand_pose = list(pivot_candidate.copy())
    cand_pose.extend(rotation_3d_candidate)
    cand_pose[1] += cam2base_h

    #### xzy -> xyz
    temp = cand_pose[1]
    cand_pose[1] = cand_pose[2]
    cand_pose[2] = temp

    overlap_n = 0
    world_lines_candidate, grid_lines_candidate = draw_sight_rays(runner, [candidate], 0, pivot_candidate[1], scene_name, iter_number, iter)  ## world lines: [piv[0], piv[1], wx, wy] and grid lines: [x0, y0, gx, gy]
    H, W = obs_candidate['depth_sensor'].shape

    pcd_candidate, pose_candidate = runner.convert_obs_to_pcd_v2(obs_candidate, cand_pose, W=W, H=H, downsample_rate=downsample_rate)
    lower_limit = ((H//2) - 1)*W
    upper_limit = lower_limit + W

    # This is the middle slice for calculating the score
    pcd_candidate_slice = list(pcd_candidate[lower_limit:upper_limit, :3])

    # One map is to calculate the score
    # The other map is to update the map
    temp_map_copy = original_map.copy()
    perm_map_copy = original_map.copy()


    if plot_flag:
        plot_topdown(floor_heights, runner) 
    
    ### Both temp_map_copy and perm_map_copy mean the map updated by the candidate
    ### The difference is temp_map_copy will have the current 0.5 of rays on the explored map, to distinguish the map updated by the saved pose
    ### The value: 1 means not yet explored, 0.5 means just explored by the candidate, 0 means occupied(either explored or non-navigable regions)
    ### count_1 represents how much new area is explored, and it will be used in the first iteration
    temp_map_copy, perm_map_copy, count_1 = update_map(runner, temp_map_copy, perm_map_copy, world_lines_candidate, grid_lines_candidate, bounds, pivot_candidate, scene_name, iter_number, init_unexplored, floor_time, scene_time, pivot_time, "fc")

    explored_map_copy = explored_map.copy()
    stopping_map_copy = stopping_map.copy()
    ### update_map_check is not real update, it is just to check how much new region is explored compared with the last iteration.
    ### total_count represents how many cells have been explored by the candidate
    ### new_count represents among the total maps, how many of them are newly explored
    _, _, total_count, new_count = update_map_check(runner, explored_map_copy, stopping_map_copy, world_lines_candidate, grid_lines_candidate, bounds, pivot_candidate, scene_name, iter_number, iter)
    
    explore_score = (total_count - new_count) / np.sum(stopping_map)

    if plot_flag:
        plot_update_candidate(scene_name, pivot_candidate, temp_map_copy, iter_number, iter)

    rel_ele = np.ones(1, dtype=np.int32)
    scores = np.ones(1, dtype=np.float32)

    if len(saved_depth) == 0:
        scores = scores
        return 0, count_1 / np.sum(original_map), obs_candidate, candidate_pose, rel_ele, rel_ele, scores, pcd_candidate_slice
    
    rel_ele = np.zeros(len(saved_pose) + 1, dtype=np.int32)
    rel_ele[-1] = 1
    scores = np.zeros(len(saved_pose) + 1, dtype=np.float32)
    scores[-1] = 1

    # @TODO: Need optimization here
    
    
    for ind_2, svd_pose in enumerate(saved_pose):  ###### Need optimization here
        svd_pose_copy = copy(svd_pose)
        svd_pose_copy_temp = copy(svd_pose)
        svd_pose_copy[2] -= 1.5 # X Z Y
        temp = svd_pose_copy[2] 
        svd_pose_copy[2] = svd_pose_copy[1] # X Y Z
        svd_pose_copy[1] = temp
        grid_saved = runner.get_grid_pos(svd_pose_copy)

        svd_candidate = get_candidate_svd(runner, svd_pose_copy[1], svd_pose_copy, svd_pose_copy[4], original_map, scene_name, iter_number, iter)
        (obs_saved, pivot_saved, rotation_save_pose, candidates_rotation_rad_save_pose) = svd_candidate

        #### Check here!!!
        pcd_saved, pose_saved = runner.convert_obs_to_pcd_v2(obs_saved, svd_pose_copy_temp, W=W, H=H, downsample_rate=downsample_rate)
        down_W, down_H = int(W/downsample_rate),  int(H/downsample_rate)
        #Added by Taarun
        pcd_intersection_score, pcd_union_score, iou, intersection_indices_cand, intersection_indices_svd =  runner.calculate_iou_point_cloud_v3(pcd_candidate, pcd_saved, pose_candidate, pose_saved, H=down_H, W=down_W)

        world_lines_saved, grid_lines_saved = draw_sight_rays(runner, [svd_candidate], 0, pivot_saved[1], scene_name, iter_number, iter + 0.5)
        
        if plot_flag:
            plot_step_star(scene_name, pivot_candidate, runner, grid_candidate, grid_saved, candidates_rotation_rad_candidate, candidates_rotation_rad_save_pose, iter_number)

                
        iou_threshold = 0.000 # a very small overlap will count

        if iou > iou_threshold:
            temp_map = temp_map_copy.copy()
            perm_map = perm_map_copy.copy()
            temp_map, perm_map, _, o_count = update_map_check(runner, temp_map, perm_map, world_lines_saved, grid_lines_saved, bounds, pivot_saved, scene_name, iter_number, iter)
            if plot_flag:
                plot_update_map_iter(scene_name, pivot_saved, temp_map, iter_number, iter)

            overlap_score = iou

            if overlap_score > 0.0:
                rel_ele[ind_2] = 1
                ind_cand = len(rel_ele) - 1
                ind_svd = ind_2
                plot_intersect(obs_candidate, obs_saved, intersection_indices_cand, intersection_indices_svd, downsample_rate, scene_name, iter_number, ind_cand, ind_svd)
        else:
            overlap_score = 0.0

        scores[ind_2] = overlap_score

        if max_overlap < overlap_score:
            max_overlap = overlap_score

        weighted_score = (ALPHA * overlap_score) + (1 - ALPHA) * explore_score
        if max_weight < weighted_score:
            max_weight = weighted_score
            max_rel_ele = rel_ele.copy()  ### bug: why use max_rel_ele. It will get updated only when a new high score comes. We can just send the rel_ele.
        
    return max_overlap, explore_score, obs_candidate, candidate_pose, max_rel_ele, rel_ele, scores, pcd_candidate_slice


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
    ALPHA=0.1,
    high_bound=0.9, # @TODO: Test with 0.4
    threshold_radius=25,
    plot_flag=False):

    max_score = -np.inf
    max_index = -1
    best_obs = None
    best_pose = None
    best_rel = None
    best_adj_row = None
    best_overlap_scores = None 
    best_explore_score = None
    pcd1_slice_max = []

    # @TODO: Need optimization here
    for i, candidate in enumerate(candidates):
        obs, pivot, rotation, yaw = candidate

        depth_img = obs["depth_sensor"]
        width, height = depth_img.shape[1], depth_img.shape[0]

        pcd = runner.convert_obs_to_pcd(obs, pivot, rotation, scene_name, iter_number, width=width, height=height, plot_flag=i)
        point_cloud_array = np.asarray(pcd.points)

        focal_length = 364.86  ## in mm
        rotation_ = [0, yaw, 0]

        # runner.get_set_depth_sensor_params()
    
        # Process here to downsample the depth image to avoid OOM issue, we were using it.
        slice_ind = height // 2
        depth_vis = np.zeros(depth_img.shape)
        depth_vis[slice_ind] = depth_img[slice_ind]
        array_len = len(depth_vis[slice_ind])

        # Make sure your depth image has some informations, not images like facing to the wall, or look outside the environment. Make sure it is a decent image
        offset = 4
        if (sum(depth_vis[slice_ind]) <= 1000 * offset
            or sum(depth_vis[slice_ind][int(array_len / 2) :]) <= 400 * offset
            or sum(depth_vis[slice_ind][: int(array_len / 2)]) <= 400 * offset
            or sum(depth_vis[slice_ind][int(array_len / 4) : int(array_len * 3 / 4)])
            <= 400 * offset):
            continue


        pose = list(pivot.copy().squeeze())
        pose.extend(rotation_)   ### pose-> [x, y, z, 0, yaw in angles, 0]
        
        #### This is the debugging function!!!!! <------cc write 
        ### Calculate each candidate's scoring
        # overlap_score: to measure the overlapping between the candidate with the existed observations.
        # explore_score: to measure the newly explored region
        (overlap_score, explore_score, cloud_obs, cloud_pose, rel_ele, adj_row, scores, pcd1_slice) = intersect_score_simple(runner, original_map, explored_map, stopping_map, candidate, pose,
                                                                                        saved_depth, saved_color, saved_pose, scene_name, iter_number, i, init_unexplored, floor_time,
                                                                                        scene_time, pivot_time, ALPHA = ALPHA)
        
        #### Overlap not too large and not too small
        # This is important. We want to calculate a scoring function. alpha value should be large to bias towards exploring new regions
        weighted_score = (ALPHA * overlap_score) + (1 - ALPHA) * explore_score 
        
        # save the best candidate
        if (max_score < weighted_score):
            max_score = weighted_score
            max_index = i
            best_obs = cloud_obs
            best_pose = cloud_pose
            best_rel = rel_ele
            best_adj_row = adj_row
            best_overlap_scores = scores
            best_explore_score = explore_score
            pcd1_slice_max = pcd1_slice

    # if max_index == -1 or explore_score == 0:  ### Bug: why care about explore_score == 0? This will of last observation's diff score anyways.
    if max_index == -1 or best_explore_score == 0 or best_explore_score == None:
        # print("No good candidates found at pivot: {} of scene: {}".format(len(saved_depth) - 1, scene_name))
        return None, None, saved_depth, saved_color, saved_pose, rel_mat, None, False, pcd1_slice_max

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

    #saving the pcd
    # print("saving pcd array")    
    # np.save(os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), "pcd", f"pcd_{len(saved_pose)-1}.npy"), new_pcd_array)

    saved_obs_dir = os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), "saved_obs")

    os.makedirs(saved_obs_dir, exist_ok=True)

    np.save(os.path.join(saved_obs_dir, "saved_dep.npy"), saved_depth)
    np.save(os.path.join(saved_obs_dir, "saved_color.npy"), saved_color)
    np.save(os.path.join(saved_obs_dir, "saved_pose.npy"), saved_pose)

    ### saving for visualization
    saved_grid_pose = saved_pose.copy()
    for idx, svd_pose in enumerate(saved_grid_pose):
        pivot_ = svd_pose[:3]
        pivot_[2], pivot_[1] = pivot_[1], pivot_[2]
        grid_pivot = runner.get_grid_pos(pivot_)
        saved_grid_pose[idx] = grid_pivot
    
    np.save(os.path.join(saved_obs_dir, "saved_grid_pose.npy"), saved_grid_pose)

    if len(rel_mat) != 0:
        rel_mat_copy = np.eye(len(saved_depth), dtype=np.double)
        rel_mat_copy[: len(rel_mat), : len(rel_mat)] = rel_mat
        assert len(best_adj_row) == len(best_overlap_scores), f" len of best_adj_row: {len(best_adj_row)} != len of best_overlap_scores: {len(best_overlap_scores)}"
        curr_scores = best_adj_row * best_overlap_scores
        curr_scores[-1] = 1.0
        rel_mat_copy[-1, :] = curr_scores
        rel_mat_copy[:, -1] = curr_scores.transpose()
        rel_mat = rel_mat_copy
    else:
        rel_mat = np.eye(1, dtype=np.double)


    # display using visualize_graph
    rel_mat_dir = os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number),"rel_mats")
    os.makedirs(rel_mat_dir, exist_ok=True)
    visualize_graph(runner, rel_mat, runner.get_top_down(height=saved_pose[0][2] - 1.5), saved_pose, output_file= os.path.join(rel_mat_dir, f"rel_mat_{len(rel_mat[0])}.png"))

    np.save(os.path.join(saved_obs_dir, "rel_mat.npy"), rel_mat)

    plot_flag = True
    if plot_flag:
        image_pil = Image.fromarray(np.array(best_obs["color_sensor"]).astype(np.uint8))
        image_pil.save(os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), "saved_obs", f"best_color_{len(saved_depth) - 1}.png"), "PNG")
  
        depth_array = np.array(best_obs["depth_sensor"])
        depth_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
        depth_scaled = (depth_normalized * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_scaled)
        depth_image.save(os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), "depth", f"best_depth_{len(saved_depth) - 1}.png"), "PNG")
        fig, ax = plt.subplots()
        ax.imshow(np.array(best_obs["depth_sensor"]))

        fig.savefig(os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), "depth", f"plt_best_depth_{len(saved_depth) - 1}.png"))
        plt.close()

    if plot_flag:
        fig, ax = plt.subplots()
        turbo = plt.get_cmap("gist_rainbow")
        cNorm = color_.Normalize(vmin=0, vmax=len(saved_pose))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=turbo)
        scalarMap.set_array([])

        for idx, svd_pose in enumerate(saved_pose):
            if idx == 0:
                ax.set_title(f"{scene_name} {svd_pose[2] - 1.5}")
                ax.imshow(runner.get_top_down(height=svd_pose[2] - 1.5))
            length = 100
            cand_radians = svd_pose[-2]

            pivot_ = svd_pose[:3]
            pivot_[2], pivot_[1] = pivot_[1], pivot_[2]
            grid_pivot = runner.get_grid_pos(pivot_)
            x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])
            colorVal = scalarMap.to_rgba(idx)
            # Add transparent patch to visualize the radius
            ax.add_patch(Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=threshold_radius, color="red", alpha = 0.2))
            ax.add_patch(arrow(x0, y0, length, math.radians(cand_radians), colorVal))

        fig.colorbar(cmx.ScalarMappable(norm=cNorm, cmap=turbo), ax=ax)
        fig.savefig(os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), "all_best_locs.png"))
        plt.close()
    
    svd_pose = saved_pose[-1]
    pivot_ = svd_pose[:3]
    pivot_[2], pivot_[1] = pivot_[1], pivot_[2]
    grid_pivot = runner.get_grid_pos(pivot_)

    

    return max_index, max_score, saved_depth, saved_color, saved_pose, rel_mat, grid_pivot, True, pcd1_slice_max


def update_map_ray(explored_map: np.ndarray, stopping_map: np.ndarray, count_set, grid_pcl: Tuple[int,int,float,float], grid_pivot: List[int], step_size=1, view_range=80000, plot_flag=True) -> Tuple[np.ndarray, np.ndarray, set]:
    """
    This function is used to update the map with a ray from the pivot point to the point cloud grid.
    """
    grid_pcl = np.array(grid_pcl[2:4])
    grid_pivot = np.array(grid_pivot)

    # step size
    dist = LA.norm(grid_pcl - grid_pivot)

    assert dist!=0, f"distance between grid_pcl{grid_pcl} and grid_pivot{grid_pivot} equal to 0"
    
    cos_theta, sin_theta = np.divide(grid_pcl - grid_pivot, dist)

    steps = np.arange(step_size, dist, step_size)

    grid_rays = np.column_stack([grid_pivot[0] + steps * cos_theta, grid_pivot[1] + steps * sin_theta])

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
    
    explored_mask = (explored_map[grid_rays[:, 1], grid_rays[:, 0]] == 1)

    # update explored map
    explored_mask = explored_mask & inbound & in_view_range
    explored_map[grid_rays[explored_mask, 1], grid_rays[explored_mask, 0]] = 0.5
    stopping_map[grid_rays[explored_mask, 1], grid_rays[explored_mask, 0]] = 0
    
    # update count map
    count_set.update(map(tuple, grid_rays[explored_mask]))  ### Fixed bug: It is now updating the count_set with the grid rays' positions.

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
        count_set.update(map(tuple, grid_rays[explored_mask]))

    return explored_map, stopping_map, count_set


def update_map_ray_check(explored_map: np.ndarray, stopping_map: np.ndarray, explored_map_copy: np.ndarray, grid_pcl: Tuple[int,int,float,float], grid_pivot: List[int], step_size=1, view_range=80000, plot_flag=True,) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    This function is used to update the map with a ray from the pivot point to the point cloud grid.
    """
    count = 0
    o_count = 0

    grid_pcl = np.array(grid_pcl[2:4])
    grid_pivot = np.array(grid_pivot)

    # step size
    dist = LA.norm(grid_pcl - grid_pivot)

    if dist==0:
        return explored_map, stopping_map, count, o_count
    
    cos_theta, sin_theta = np.divide(grid_pcl - grid_pivot, dist)

    steps = np.arange(step_size, dist, step_size)
    
    grid_rays = np.column_stack([grid_pivot[0] + steps * cos_theta, grid_pivot[1] + steps * sin_theta])

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
    
    explored_mask = (explored_map[grid_rays[:, 1], grid_rays[:, 0]] == 1)

    explored_mask = explored_mask & inbound & in_view_range

    count = explored_mask.sum()

    mask_05 = ((explored_map[grid_rays[:, 1], grid_rays[:, 0]] == 0.5) & inbound & in_view_range)
    count += mask_05.sum()

    explored_map[grid_rays[explored_mask, 1], grid_rays[explored_mask, 0]] = 0.5
    stopping_map[grid_rays[explored_mask, 1], grid_rays[explored_mask, 0]] = 0

    o_count = ((explored_map_copy[grid_rays[:, 1], grid_rays[:, 0]] == 0.5) & inbound & in_view_range)
    o_count = o_count.sum()

    # clip indices to range of valid indices
    pcl_indices = np.clip(np.array([int(grid_pcl[1]), int(grid_pcl[0])]), 0, np.subtract(explored_map.shape, 1))

    if explored_map[pcl_indices[0]][pcl_indices[1]] == 1:
        explored_map[pcl_indices[0]][pcl_indices[1]] = 0.5
        stopping_map[pcl_indices[0]][pcl_indices[1]] = 0
        count += 1
    elif explored_map[pcl_indices[0]][pcl_indices[1]] == 0.5:
        count += 1
    if explored_map_copy[pcl_indices[0]][pcl_indices[1]] == 0.5:
        o_count += 1

    return explored_map, stopping_map, count, o_count


def update_map(runner, explored_map, stopping_map, world_lines, grid_lines, bounds, pivot, scene_name, iter_number, pivot_number, init_unexplored, floor_time, scene_time, pivot_time, pcd_candidate_slice = None, plot_flag=False):
    ## This is to update the map by the candidate, for calculating how much area are newly explore and how much area are previously explored
    count_set = set()
    pcd_grid = grid_lines
    grid_pivot = runner.get_grid_pos(pivot)
    explored_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0.5
    stopping_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0
    count_set.add((int(grid_pivot[1]), int(grid_pivot[0])))
    
    for idx, world_line in enumerate(world_lines):
        try:
            explored_map, stopping_map, count_set = update_map_ray(explored_map, stopping_map, count_set, pcd_grid[idx], grid_pivot)
        except:
            continue

    if plot_flag:
        plot_update_map(runner, pcd_candidate_slice, scene_name, pivot, explored_map, grid_pivot, pivot_time, scene_time, floor_time, stopping_map, init_unexplored, pivot_number, iter_number)
    
    return explored_map, stopping_map, len(count_set)


def update_map_check(runner, explored_map, stopping_map, world_lines, grid_lines, bounds, pivot, scene_name, iter_number, pivot_number, plot_flag=False):
    count = 0
    o_count = 0
    pcd_grid = grid_lines
    grid_pivot = runner.get_grid_pos(pivot)
    explored_map_copy = copy(explored_map)

    # Update pivot point location as 0.5, o_count counts how many new cells are updated
    # count counts up the total number of cells updated
    # Thus count - o_count = the number of cells that have been explored before
    if explored_map[int(grid_pivot[1])][int(grid_pivot[0])] == 1:
        explored_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0.5
        stopping_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0
    elif explored_map[int(grid_pivot[1])][int(grid_pivot[0])] == 0.5:
        o_count += 1
    count += 1

    for idx, world_line in enumerate(world_lines):
        ##### Need optmization here
        explored_map, stopping_map, count_per_ray, o_count_per_ray = update_map_ray_check(explored_map, stopping_map, explored_map_copy, pcd_grid[idx], grid_pivot)
        count += count_per_ray
        o_count += o_count_per_ray
    if plot_flag:
        plot_update_map_check(scene_name, pivot, explored_map, iter_number, pivot_number)

    return explored_map, stopping_map, count, o_count


def arrow(x0, y0, length, angle, color):
    x1 = math.cos(np.pi / 2 + angle) * length
    y1 = math.sin(np.pi / 2 + angle) * length

    return Arrow(x0, y0, x1, -y1, width=100.0, color=color)

def depth2grid(runner, i, angle_range, pivot_, depth_slice, x0, y0, yaw_deg):
    rad = angle_range[i]
    yaw_rad = math.radians(yaw_deg)
    sign_x = 1 if math.cos((np.pi / 2) + rad) >= 0 else -1
    sign_y = 1 if math.sin(np.pi / 2 + rad) >= 0 else -1
    del_x = depth_slice[i] * abs(math.cos((np.pi/2) + rad)/math.sin((np.pi / 2) - yaw_rad + rad))*sign_x
    del_y = depth_slice[i] * abs(math.sin((np.pi/2) + rad)/math.sin((np.pi / 2) - yaw_rad + rad))*sign_y

    x1, y1 = pivot_[0] + del_x, pivot_[2] - del_y

    # then convert to grid
    # px = x0 + (del_x * 100)
    # py = y0 + (del_y * 100)
    px, py = runner.get_grid_pos((x1, 1.5, y1))

    return (pivot_[0], pivot_[2], x1, y1), (x0, y0, px, py)


def draw_sight_rays(runner, candidates, best_index, height, scene, iter_number, pivot_number, plot_flag=False):
    obs_, pivot_, rotation_, cand_rad = candidates[best_index]

    depth_img = obs_["depth_sensor"]

    # Take slice of depth image at sensor height (center of image) to update the map
    slice_ind = depth_img.shape[0] // 2
    depth_vis = np.zeros(depth_img.shape)
    depth_vis[slice_ind] = depth_img[slice_ind]

    # Assign each pixel to angle, # perceptive field is +-45 degree
    left_angle = math.radians(cand_rad - 45)
    right_angle = math.radians(cand_rad + 45)
    yaw_deg = cand_rad
    
    angle_range = np.linspace(left_angle, right_angle, depth_img.shape[1])  ### could be a minor bug: maybe discretize angle range with max(depth_img.shape[1], depth_img.shape[0]) instead of depth_img.shape[1]
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
        plot_topdown(runner, height, patch)

    angle_range = angle_range[::-1]
    i_range = np.arange(len(angle_range))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        ar_result = [executor.submit(depth2grid, runner, i, angle_range, pivot_, depth_slice, x0, y0, yaw_deg) for i in i_range]
        # retrieve the return value results
        for ar in concurrent.futures.as_completed(ar_result):
            world_line, grid_line = ar.result()
            world_lines.append(world_line)
            grid_lines.append(grid_line)
    

    if plot_flag:
        plot_update_map(topdown, x0, y0, cand_rad, grid_lines, scene, iter_number, pivot_number)
    
    return world_lines, grid_lines


def algorithm(runner: TestRunner, floor_maps: list, heights: list, scene: str, SCENE_THRESHOLD: float, resume_flag=False, floor_flag=0) -> None:
    # print("INSIDE ALGORITHM FUNCTION")
    scene_time = time.time()
    sample_n_candidate = 6
    plot_flag = False

    # Keeping tabs on
    bounds = runner._sim.pathfinder.get_bounds()
    ## Check here -> TS
    seed_random = (runner._sim_settings["seed"])
    print("######### debug seed number: ", seed_random)
    np.random.seed(seed_random)

    # Check directories existence
    vis_dir = os.path.join("data_folder/More_vis", str(scene))
    batch_dir = os.path.join("data_folder/batch_indx", str(scene))
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(batch_dir, exist_ok=True)

    # TS -> starting from 0 floor 
    # Start from floor to floor
    for i, floor in enumerate(floor_maps[floor_flag:]):
        floor_time = time.time()

        # Create directories
        os.makedirs(os.path.join(vis_dir, str(i)), exist_ok=True)
        os.makedirs(os.path.join(batch_dir, str(i)), exist_ok=True)

        # Get the floor map, To decide a stop criteria, we want to check the percentage of the area
        # explored. The original map refers to the initial unexplored region.
        # We use explore_ratio = (explore_region + non_navigable area)/ (all vacant space + non_navigable area)
        # to define if we have explored enough regions
        original_map = np.array(floor, copy=True, dtype=float)

        init_unexplored = np.sum(original_map)
        if np.sum(floor) < 200:
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
                # explore_ratio is here < SCENE_THRESHOLD
                if np.sum(stopping_map) / np.sum(init_unexplored) < SCENE_THRESHOLD:
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

        # When we explore, we turn the pixels from true to false. So to check if we meet threshold, sum(original_map) / init_unexplored >= 0.05
        step = 0
        trial_limit = 250
        total_trials = 250
        stuck = False  # Initially unstuck
        rand = True  # Every new floor start by pick random
        pivot_number = -1
        threshold_radius = max(original_map.shape[0], original_map.shape[1]) / 20

        if np.sum(stopping_map) / np.sum(init_unexplored) < SCENE_THRESHOLD:
            continue
        
        # Create a set of all possible pivot candidate along the scene boundary and facing to room space
        pivot_array = list(map(list, pivot_candidates(original_map)))

        obs_array = []
        downsample = 30  # degree

        scaler_factor = int(len(pivot_array) / 120)
        if scaler_factor > 50:
            scaler_factor = 50

        for p in pivot_array:  ##### Need optmization here
            angle_r = np.arange(0, 360, downsample)
            for ang in angle_r:  ##### Need optmization here
                if in_map(p[0], p[1], ang, original_map):
                    obs_array.append([p[0], p[1], ang]) # [... [x, y, ang] ...]

        obs_array = np.array(obs_array)  # Convert to NumPy array and remove duplicates
        obs_array_index = np.arange(len(obs_array), dtype=int)
        obs_array_sample_index = obs_array_index.copy()
        obs_array[:, [0, 1]] = obs_array[:, [1, 0]]  # Swap x and y positions directly in the NumPy array
        obs_all_coords = [obs[:2] for obs in obs_array]

        assert len(obs_array) == len(obs_all_coords), f"length of observation all coords {len(obs_all_coords)} does not match length of observation array {len(obs_array)}"
        
        seen_pivot_coords = []
        plot_obs_array(original_map, obs_array, vis_dir, i)

        # Start the interation, observation is selected from obs_array
        while (np.sum(stopping_map) / np.sum(init_unexplored) >= SCENE_THRESHOLD) and (len(obs_array_sample_index) > 0) and total_trials > 0:
            pivot_time = time.time()
            pivot_number += 1

            # Picking Random Pivots
            final_obs_inds = []
            # TS -> Picking 6 random candidates out of all the available pivot candidates
            obs_inds = np.random.choice(obs_array_sample_index, sample_n_candidate, replace=False)  ## [40, 80, 65, 98, 100, 149]

            # Choose 6 candidates from the possible pool
            while len(final_obs_inds) < 6:
                obs_inds = obs_inds.astype(int)
                if len(seen_pivot_coords)!=0:
                    final_obs_inds += [x for x in obs_inds if sum(LA.norm(np.array(obs_array[x][:2]) - np.array(seen_pivot_coords), axis = 1).reshape(-1,) < 2.1*threshold_radius) == 0]  ## twice of threshold is done so that the areas of two pivots dont overlap.
                else:
                    final_obs_inds += [x for x in obs_inds]

                obs_inds = np.random.choice(obs_array_sample_index, 1, replace=False)
            obs_inds = final_obs_inds[:6]

            candidates = []

            ## Take the observation at the candidate nearby location
            with concurrent.futures.ThreadPoolExecutor(max_workers=(3 * n_cores) // 4) as executor:
                features = [executor.submit(process_observation, runner, heights, i, obs_array, obs_ind) for obs_ind in obs_inds]

                for f in concurrent.futures.as_completed(features):
                    (height, pivot, rotation, angle) = f.result()
                    new_pivot, _ = runner.get_random_point_near(height, grid=original_map, circle_center=pivot, radius_grid=2, pivot_list=pivot_array, pivot_indicies=obs_array_index,scalar_factor=10)
                    obs = runner.get_rgbd_from_pose(new_pivot, rotation)
                    # Draw covariant images
                    if len(candidates) != 0:
                        (_, pivot1, rotation1, _) = candidates[-1]
                        runner.get_covariant_image(str(scene), len(candidates)-1,pivot1,rotation1,len(candidates),pivot,rotation)

                    new_pivot = np.asarray(new_pivot)

                    candidate = (obs, pivot, rotation, angle) ### HERE pivot is a 3D point in the world #Changed pivot -> new_pivot
                    candidates.append(candidate)


            # Dont do rest algo for now.
            # return
            # Plot candidates on the map
            if plot_flag:
                plot_candidate(scene, heights, explored_map, candidates, colors, i, runner, pivot_number)

            #plot candidates
            if plot_flag:
                plot_candidate_GA(scene, heights, explored_map, candidates, color_map, i, runner, step)

            # Pick the best # Troubleshoot here
            (best_index, best_score, saved_depth, saved_color, saved_pose, rel_mat, grid_pivot, selected_flag, pcd1_slice) = pick_best_candidate(runner, original_map,  explored_map, stopping_map, candidates,
                                                                                                saved_depth, saved_color, saved_pose, rel_mat, stuck, scene_name=scene,
                                                                                                iter_number=i, init_unexplored=init_unexplored, floor_time=floor_time,
                                                                                                scene_time=scene_time, pivot_time=pivot_time,pivot_number=step,
                                                                                                threshold_radius = threshold_radius)
            # unstuck yourself
            # Stuck means if you tried many round and cannot make 80% of exploration, then stuck flag turns on.
            if stuck == True:
                stuck = False
            
            # get only indicies of unexplored pixels (best_index is the index of the best candidate)
            if best_index is not None:
                seen_pivot_coords.append(grid_pivot[:2])
                distances = LA.norm(np.array(obs_all_coords) - np.array(grid_pivot[:2]), axis=1)
                exclude_indices = obs_array_index[distances < 2.1*threshold_radius].copy()
                obs_array_sample_index = np.setdiff1d(obs_array_sample_index, exclude_indices)

            else:
                trial_limit -= 1 ### Trial limit is for candidate selection limit, we will resample candidates N times if no candidates work.
                total_trials -= 1 ### Trial limit is for total image selection limit, we will keep N images for a scene if they have not reached 80% of the coverage.
                stuck = True if trial_limit <= 0 else False
                continue

            if plot_flag:
                plot_best_step(scene, pivot, runner, candidates, step)

            world_lines, grid_lines = draw_sight_rays(runner, candidates, best_index, pivot[1], scene, i, step)
            obs_, pivot_, rotation_, cand_rad = candidates[best_index]
            explored_map, stopping_map, _ = update_map(runner, explored_map, stopping_map, world_lines, grid_lines, bounds, pivot_, scene, i, step,
                                            init_unexplored, floor_time, scene_time, pivot_time, pcd1_slice, plot_flag=True)

            
            np.save(os.path.join("data_folder", "More_vis", scene, str(i), "saved_obs", "explored_map.npy"), explored_map)
            np.save(os.path.join("data_folder", "More_vis", scene, str(i), "saved_obs", "stopping_map.npy"), stopping_map)

            ## Visualize the difference between init_unexplored and explored_map
            step += 1


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

def process_scene(dataset_flag, scene, floor_heights,scene_dir,resume_flag, scene_flag=None, floor_flag=None,SCENE_THRESHOLD=0.20):
    # TS -> Creating scene specific directories
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs(os.path.join("data_folder/batch_indx", str(scene)), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "visualization"), exist_ok=True)
    # build the scene using scene_flag
    if scene_flag != None and scene != scene_flag:
        scene_name = scene_flag + ".glb"
    else:
        scene_name = scene + ".glb"

    ## load Glb, runner and habitat navmeash
    if dataset_flag == "gibson" or dataset_flag == "test":
        default_sim_settings["scene"] = f"gibson/gibson/{scene_name}"

    if dataset_flag == "hm3d":
        default_sim_settings["scene"] = f"hm3dDataset_full/{scene_name}"
    
    runner = TestRunner(default_sim_settings)
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

    algorithm(runner, floor_maps, floor_heights, scene, SCENE_THRESHOLD, resume_flag=resume_flag, floor_flag=floor_flag)

    runner._sim.close()


def main(start=0, end=None, resume_flag=False, scene_flag=None, floor_flag=None):
    # TS -> Creating folders to store results
    os.makedirs("data_folder/More_vis", exist_ok=True)
    os.makedirs("data_folder/batch_indx", exist_ok=True)

    # TS -> To cover 80% area of the scenes
    SCENE_THRESHOLD = 0.20  # equivalent to 80% coverage

    #TS -> adding dataset_flag to determine the dataset type
    dataset_flag = "test"

    if dataset_flag == "gibson":
        print("The gibson dataset is processed")
        gibson_fp = "igibson_floor_heights.json"
        with open(gibson_fp, "r") as j:
            json_file = json.load(j)
        available_scenes = {filename[14:-4] for filename in glob.glob("gibson/gibson/*.glb")}

    elif dataset_flag == "test":
        print("Test one scene of the gibson dataset")
        # available_scenes = {'Albertville'}
        # available_scenes = {'Roxboro', 'Albertville', 'Eagerville', 'Sodaville', 'Rosser', 'Sawpit', 'Monson', 'Greigsville', 'Avonia', 'Kerrtown', 'Stilwell', 'Oyens','Sisters','Bowlus'}
        available_scenes = {'Sanctuary', 'Angiola', 'Maryhill', 'Sasakwa', 'Bowlus', 'Anaheim', 'Mesic', 'Sargents', 'Capistrano', 'test', 'Edgemere', 'Seward', 'Crandon', 'Sands', 'Delton'}
        #available_scenes = {'Hambleton', 'Arkansaw', 'Bolton', 'Cooperstown', 'Soldier', 'Micanopy', 'Woonsocket', 'Andover', 'Spencerville', 'Scottsmoor'}
        # available_scenes = {'Haxtun', 'Rancocas', 'Silas', 'Colebrook', 'Mobridge', 'Stanleyville', 'Scioto','Mosquito', 'Convoy', 'Mosinee', 'Ballou', 'Azusa', 'Ribera'}
        # available_scenes = {'Eastville', 'Sumas', 'Spotswood', 'Quantico', 'Hominy', 'Parole', 'Shelbiana', 'Swormville', 'Placida', 'Nimmons', 'Mifflintown', 'Nicut', 'Pablo'}
        # available_scenes = {'Goffs', 'Hillsdale', 'Applewold', 'Nuevo', 'Hometown', 'Cantwell', 'Pettigrew', 'Superior', 'Beach', 'Annawan', 'Elmira', 'Stokes', 'Reyno'}
        # available_scenes = {'Pleasant', 'Dunmor', 'Springhill', 'Eudora', 'Brevort', 'Denmark', 'Hainesburg', 'Nemacolin', 'Dryville', 'Roane', 'Roeville', 'Adrian'}
        gibson_fp = "igibson_floor_heights.json"
        with open(gibson_fp, "r") as j:
            json_file = json.load(j)

    elif dataset_flag == "hm3d":
        print("the HM3D dataset is processed")
        hm3d_fp = "hm3d_floor_heights.json"
        with open(hm3d_fp, "r") as j:
            json_file = json.load(j)
        available_scenes = {filename[12:-4] for filename in glob.glob("hm3dDataset_full/*.glb")}
        available_scenes = ["7GvCP12M9fi.basis"]

    print("available_scenes::", available_scenes)
    
    start_time = time.time()

    ## Multi process to deal with it.
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context= mp.get_context('spawn')) as executor:
        futures=[]
        for scene in json_file:
            if scene in available_scenes:
                scene_dir = os.path.join("data_folder/More_vis", str(scene))
                # TS -> Calling the process scene function 
                futures.append(executor.submit(process_scene, dataset_flag, scene, json_file[scene], scene_dir, resume_flag, scene_flag, floor_flag, SCENE_THRESHOLD))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(e)

    # TS -> Execution ends here
    print("Total time taken: ", time.time()-start_time)

if __name__ == "__main__":
    #TS -> Starting point of the program
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

    # TS -> Calling the main function 
    main(FLAGS.start, FLAGS.end, FLAGS.resume, FLAGS.scene, FLAGS.floor)
