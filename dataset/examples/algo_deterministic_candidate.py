from test_runner import TestRunner
from settings import default_sim_settings
from collections import *
import random
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
from utils import pivot_candidates, get_candidate_GA, in_map

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
print(f"Number of Logical CPU cores: {n_cores}")


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
                            scene_name, iter_number, iter, init_unexplored, floor_time, scene_time, pivot_time, ALPHA=0.1, plot_flag=False, index = 0):
   
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
    world_lines1, grid_lines1 = draw_sight_rays(runner, [c1], 0, pivot_c1[1], scene_name, iter_number, iter)  ## world lines: [piv[0], piv[1], wx, wy] and grid lines: [x0, y0, gx, gy]
    # grid_lines1_set = set([(int(gridline[2]), int(gridline[3])) for gridline in grid_lines1])
    

    # c1_ind = my_function_c1()
    pcd_c1 = runner.convert_obs_to_pcd_v2(obs_c1, c1_pose)
    lower_limit = ((1080//2) - 1)*1920
    upper_limit = lower_limit + 1920

    pcd1_slice = list(pcd_c1[lower_limit:upper_limit, :3])

    # r = int(0.01*(min(original_map.shape[0], original_map.shape[1]))) # add the nearby pixels (within radius r) to the set to incorporate the roundedness and accuracy of grid lines.
    # # add nearby grid lines to the set
    # for gridline in grid_lines1:
    #     _, _, x0, y0 = gridline
    #     for i in range(-r, r):
    #         for j in range(-r, r):
    #             grid_lines1_set.add((int(x0) + i, int(y0) + j))

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

        fig.savefig(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"update_grid_lines_test.png"))
        plt.close()

    temp_map_copy, temp_map2_copy, count_1 = update_map(runner, temp_map_copy, temp_map2_copy, world_lines1, grid_lines1, bounds, pivot_c1, scene_name, iter_number, init_unexplored, floor_time, scene_time, pivot_time, "fc")
    ### temp copy will have the current 0.5 of rays on the explored map.

    explored_map_copy = explored_map.copy()
    stopping_map_copy = stopping_map.copy()
    _, _, count_2, o_count = update_map_check(runner, explored_map_copy, stopping_map_copy, world_lines1, grid_lines1, bounds, pivot_c1, scene_name, iter_number, iter)
    
    diff_score = (count_2 - o_count) / np.sum(stopping_map)

    if plot_flag:
        fig, ax = plt.subplots()
        ax.set_title(scene_name + " " + str(pivot_c1[1]))
        ax.imshow(temp_map_copy, cmap="gray")

        fig.savefig(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"update_map_{iter}_1.png"))
        plt.close()

    rel_ele = np.ones(1, dtype=np.int32)
    scores = np.ones(1, dtype=np.float32)

    if len(saved_depth) == 0:
        scores = scores
        return 0, count_1 / np.sum(original_map), obs_c1, candidate_pose, rel_ele, rel_ele, scores, pcd1_slice
    
    rel_ele = np.zeros(len(saved_pose) + 1, dtype=np.int32)
    rel_ele[-1] = 1
    scores = np.zeros(len(saved_pose) + 1, dtype=np.float32)
    scores[-1] = 1

    # @TODO: Need optimization here
    
    for ind_2, svd_pose in enumerate(saved_pose):  ###### Need optimization here
        p2 = copy(svd_pose)
        p2_temp = copy(svd_pose)
        p2[2] -= 1.5 # X Z Y
        temp = p2[2] 
        p2[2] = p2[1] # X Y Z
        p2[1] = temp
        g2 = runner.get_grid_pos(p2)

        c2 = get_candidate_GA(runner, p2[1], p2, p2[4], original_map, scene_name, iter_number, iter)
        (obs_c2, pivot_c2, rotation_c2, candidates_rotation_rad_c2) = c2

        rotation_2 = [0, candidates_rotation_rad_c2, 0]
        c2_pose = list(pivot_c2.copy())
        c2_pose.extend(rotation_2)

  
        pcd_c2 = runner.convert_obs_to_pcd_v2(obs_c2, p2_temp)
        pcd2_slice = pcd_c2[lower_limit:upper_limit, :3]
  
        # pcd_intersection_score, pcd_union_score, iou = runner.calculate_iou_point_cloud(pcd_c1, pcd_c2)
        #Added by Taarun
        pcd_intersection_score, pcd_union_score, iou =  runner.calculate_iou_point_cloud_v2(pcd_c1, pcd_c2)
        # pcd_intersection_score, pcd_union_score, iou =  runner.calculate_iou_point_cloud_v3(pcd_c1, pcd_c2)

        world_lines2, grid_lines2 = draw_sight_rays(runner, [c2], 0, pivot_c2[1], scene_name, iter_number, iter + 0.5)
        
        if plot_flag:
            length = 100
            fig, ax = plt.subplots()
            ax.set_title(scene_name + " " + str(pivot_c1[1]))
            ax.imshow(runner.get_top_down(height=pivot_c1[1]))
            x0, y0 = int(g1[0]), int(g1[1])
            ax.add_patch(Circle((int(g1[0]), int(g1[1])), radius=25, color="blue"))
            ax.add_patch(arrow(x0, y0, length, math.radians(candidates_rotation_rad_c1), "green"))
            x1, y1 = int(g2[0]), int(g2[1])
            ax.add_patch(Circle((int(g2[0]), int(g2[1])), radius=25, color="red"))
            ax.add_patch(arrow(x1, y1, length, math.radians(candidates_rotation_rad_c2), "green"))
            fig.savefig(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"step_{iter_number}_star.png"))
            plt.close()

        # # Check if the gridlines2 and gridlines1 have some overlap
        # grid_lines2_set = set([(int(gridline[2]), int(gridline[3])) for gridline in grid_lines2])
        # end_points_inter = grid_lines1_set.intersection(grid_lines2_set)

        # assert len(grid_lines1) == 1920 and len(grid_lines2) == 1920, f"Length of either grid_lines1 or grid_lines2 is not 1920"

        # beta_1 = sum([1 for gridpoint in grid_lines1 if (int(gridpoint[2]), int(gridpoint[3])) in end_points_inter])/len(grid_lines1)
        # beta_2 = sum([1 for gridpoint in grid_lines2 if (int(gridpoint[2]), int(gridpoint[3])) in end_points_inter])/len(grid_lines2)

        # root_score = math.sqrt(beta_1 * beta_2)
                
        iou_threshold = 0.000
        print(f"iou threshold: {iou_threshold}")

        if iou >= iou_threshold:
        
            temp_map = temp_map_copy.copy()
            temp_map2 = temp_map2_copy.copy()
            temp_map, temp_map2, count_2, o_count = update_map_check(runner, temp_map, temp_map2, world_lines2, grid_lines2, bounds, pivot_c2, scene_name, iter_number, iter)
            if plot_flag:
                fig, ax = plt.subplots()
                ax.set_title(scene_name + " " + str(pivot_c2[1]))
                ax.imshow(temp_map, cmap="gray")
                fig.savefig(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"update_map_{iter}_2.png"))
                plt.close()
            # overlap_score = o_count / count_2
            overlap_score = iou

            p1_copy = p1.copy()
            p2_copy = p2.copy()

            if overlap_score > 0.0:
                rel_ele[ind_2] = 1
        else:
            overlap_score = 0.0

        scores[ind_2] = overlap_score

     
        # print("beta1: {}, beta2: {}, SCORE: {}, overlap score: {} with grid lines 2: {} for index: {}".format(beta_1, beta_2, root_score, overlap_score, grid_lines2[:5], index))

        if max_overlap < overlap_score:
            max_overlap = overlap_score

        weighted_score = (ALPHA * overlap_score) + (1 - ALPHA) * diff_score
        if max_weight < weighted_score:
            max_weight = weighted_score
            max_rel_ele = rel_ele.copy()  ### bug: why use max_rel_ele. It will get updated only when a new high score comes. We can just send the rel_ele.
        
    return max_overlap, diff_score, obs_c1, candidate_pose, max_rel_ele, rel_ele, scores, pcd1_slice


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
    best_diff_score = None
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

        # Take slice of depth image at sensor height (center of image)
        slice_ind = height // 2
        depth_vis = np.zeros(depth_img.shape)
        depth_vis[slice_ind] = depth_img[slice_ind]
        array_len = len(depth_vis[slice_ind])

        offset = 4
        if (sum(depth_vis[slice_ind]) <= 1000 * offset
            or sum(depth_vis[slice_ind][int(array_len / 2) :]) <= 400 * offset
            or sum(depth_vis[slice_ind][: int(array_len / 2)]) <= 400 * offset
            or sum(depth_vis[slice_ind][int(array_len / 4) : int(array_len * 3 / 4)])
            <= 400 * offset):
            continue

        if plot_flag:
            # Compare slice and original depth
            plt.imshow(depth_vis)
            plt.savefig(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"test_slice_{i}.png"))
            plt.close()

        pose = list(pivot.copy().squeeze())
        pose.extend(rotation_)   ### pose-> [x, y, z, 0, yaw in angles, 0]
        
        (overlap_score, diff_score, cloud_obs, cloud_pose, rel_ele, adj_row, scores, pcd1_slice) = intersect_score_simple(runner, original_map, explored_map, stopping_map, candidate, pose,
                                                                                        saved_depth, saved_color, saved_pose, scene_name, iter_number, i, init_unexplored, floor_time,
                                                                                        scene_time, pivot_time, ALPHA = ALPHA)
        #### Overlap not too large and not too small
        if len(saved_depth) != 0 and not stuck:
            continue

        weighted_score = (ALPHA * overlap_score) + (1 - ALPHA) * diff_score

        if (max_score < weighted_score):
            max_score = weighted_score
            max_index = i
            best_obs = cloud_obs
            best_pose = cloud_pose
            best_rel = rel_ele
            best_adj_row = adj_row
            best_overlap_scores = scores
            best_diff_score = diff_score
            pcd1_slice_max = pcd1_slice

    # if max_index == -1 or diff_score == 0:  ### Bug: why care about diff_score == 0? This will of last observation's diff score anyways.
    if max_index == -1 or best_diff_score == 0 or best_diff_score == None:
        print("No good candidates found at pivot: {} of scene: {}".format(len(saved_depth) - 1, scene_name))
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
    # np.save(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), "pcd", f"pcd_{len(saved_pose)-1}.npy"), new_pcd_array)

    saved_obs_dir = os.path.join("temp", "More_vis", str(scene_name), str(iter_number), "saved_obs")

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
    rel_mat_dir = os.path.join("temp", "More_vis", str(scene_name), str(iter_number),"rel_mats")
    os.makedirs(rel_mat_dir, exist_ok=True)
    visualize_graph(runner, rel_mat, runner.get_top_down(height=saved_pose[0][2] - 1.5), saved_pose, output_file= os.path.join(rel_mat_dir, f"rel_mat_{len(rel_mat[0])}.png"))

    np.save(os.path.join(saved_obs_dir, "rel_mat.npy"), rel_mat)

    plot_flag = True
    if plot_flag:
        image_pil = Image.fromarray(np.array(best_obs["color_sensor"]).astype(np.uint8))
        image_pil.save(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), "saved_obs", f"best_color_{len(saved_depth) - 1}.png"), "PNG")
  
        depth_array = np.array(best_obs["depth_sensor"])
        depth_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
        depth_scaled = (depth_normalized * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_scaled)
        depth_image.save(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), "depth", f"best_depth_{len(saved_depth) - 1}.png"), "PNG")
        fig, ax = plt.subplots()
        ax.imshow(np.array(best_obs["depth_sensor"]))

        fig.savefig(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), "depth", f"plt_best_depth_{len(saved_depth) - 1}.png"))
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
        fig.savefig(os.path.join("temp", "More_vis", str(scene_name), str(iter_number), "all_best_locs.png"))
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
    # count_set.update(grid_rays[explored_mask, 0], grid_rays[explored_mask, 1])  ### bug: It is not updating the count_set with the grid rays' positions.
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
        # count_set.update(grid_rays[explored_mask, 0], grid_rays[explored_mask, 1])
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


def update_map(runner, explored_map, stopping_map, world_lines, grid_lines, bounds, pivot, scene_name, iter_number, pivot_number, init_unexplored, floor_time, scene_time, pivot_time, plot_flag=False):
    count_set = set()
    pcd_grid = grid_lines
    grid_pivot = runner.get_grid_pos(pivot)
    explored_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0.5
    stopping_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0
    count_set.add((int(grid_pivot[1]), int(grid_pivot[0])))
    
    for idx, world_line in enumerate(world_lines):
        explored_map, stopping_map, count_set = update_map_ray(explored_map, stopping_map, count_set, pcd_grid[idx], grid_pivot)
    xs = [int(i[2]) for i in grid_lines]
    ys = [int(i[3]) for i in grid_lines]
           
    if plot_flag:
        fig, ax = plt.subplots()
        # ax.scatter(xs, ys, s=1, c="red")
        ax.set_title(scene_name + " " + str(pivot[1]))
        ax.imshow(explored_map, cmap="gray")
        ax.add_patch(Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red"))

        this_pivot_time = time.time() - pivot_time
        total_scene_time = time.time() - scene_time
        total_floor_time = time.time() - floor_time

        coverage = round(abs(1 - (np.sum(stopping_map) / np.sum(init_unexplored))) * 100, 3)

        # convert total_floor_time and total_scene_time to minutes
        total_floor_time, total_scene_time = (total_floor_time / 60, total_scene_time / 60)
        metrics = f"pivot_time: {this_pivot_time:>3.2f}s\nscene_time: {total_scene_time:>3.2f}min\nfloor_time: {total_floor_time:>3.2f}min"
        
        plt.text(0.99, 0.005, metrics, ha="right", va="bottom", transform=plt.gcf().transFigure, fontsize="small", bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"))
        ax.set_title(f"{scene_name}  iter{pivot_number} | {coverage}%")

        fig.savefig(os.path.join("temp", "More_vis", scene_name, str(iter_number), f"update_map_{pivot_number}_fc.png"))

        plt.close()
    
    return explored_map, stopping_map, len(count_set)


def update_map_check(runner, explored_map, stopping_map, world_lines, grid_lines, bounds, pivot, scene_name, iter_number, pivot_number, plot_flag=False):
    count = 0
    o_count = 0
    pcd_grid = grid_lines
    grid_pivot = runner.get_grid_pos(pivot)
    explored_map_copy = copy(explored_map)

    if explored_map[int(grid_pivot[1])][int(grid_pivot[0])] == 1:
        explored_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0.5
        stopping_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0
    elif explored_map[int(grid_pivot[1])][int(grid_pivot[0])] == 0.5:
        o_count += 1
    count += 1

    for idx, world_line in enumerate(world_lines):
        ##### Need optmization here
        explored_map, stopping_map, count_temp, o_count_temp = update_map_ray_check(explored_map, stopping_map, explored_map_copy, pcd_grid[idx], grid_pivot)
        count += count_temp
        o_count += o_count_temp
    if plot_flag:
        fig, ax = plt.subplots()
        ax.set_title(scene_name + " " + str(pivot[1]))
        ax.imshow(explored_map, cmap="gray")

        fig.savefig(os.path.join("temp", "More_vis", scene_name, str(iter_number), f"update_map_{pivot_number}.png"))
        plt.close()

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

    # Take slice of depth image at sensor height (center of image)
    slice_ind = depth_img.shape[0] // 2
    depth_vis = np.zeros(depth_img.shape)
    depth_vis[slice_ind] = depth_img[slice_ind]

    # print("Draw sight pivot: ", pivot_)

    # Assign each pixel to angle
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
        topdown = runner.get_top_down(height=height)
        ax.imshow(topdown)
        ax.add_patch(patch)

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
        fig, ax = plt.subplots()
        ax.imshow(topdown)
        ax.add_patch(arrow(x0, y0, 100, math.radians(cand_rad), "red"))
        for grid_line in grid_lines:
            x0, y0, px, py = grid_line
            ax.plot((x0, px), (y0, py), color="blue")

        fig.savefig(os.path.join("temp", "More_vis", scene, str(iter_number), f"update_test_{pivot_number}.png"))
        plt.close()
    return world_lines, grid_lines


def algorithm(runner: TestRunner, floor_maps: list, heights: list, scene: str, SCENE_THRESHOLD: float, resume_flag=False, floor_flag=0) -> None:
    scene_time = time.time()
    sample_n_candidate = 6
    plot_flag = False

    # Keeping tabs on
    bounds = runner._sim.pathfinder.get_bounds()
    
    np.random.seed(runner._sim_settings['seed'])
    random.seed(runner._sim_settings["seed"])

    print("Numpy Seed: {}, Random Seed: {}".format(runner._sim_settings['seed'], runner._sim_settings['seed']))

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
        trial_limit = 3
        total_trials = 250
        stuck = False  # Initially unstuck
        rand = True  # Every new floor start by pick random
        pivot_number = -1
        threshold_radius = max(original_map.shape[0], original_map.shape[1]) / 20

        if np.sum(stopping_map) / np.sum(init_unexplored) < SCENE_THRESHOLD:
            continue
        
        # print("starting pivot array")
        pivot_array = list(map(list, pivot_candidates(original_map)))

        # print("finished creating pivot array")

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

        # Plotting pivot locations from obs_array
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(original_map, cmap="gray")
        ax.scatter(obs_array[:, 0], obs_array[:, 1], c="r", s=1)
        plt.savefig(os.path.join(vis_dir, str(i), "pivot_points.png"))
        plt.close()

        saved_depths = np.load(os.path.join("SpatialReasoningDataset1", "temp", "More_vis", scene, str(i), "saved_obs", "saved_dep.npy"), allow_pickle=True)

        while (np.sum(stopping_map) / np.sum(init_unexplored) >= SCENE_THRESHOLD) and (len(obs_array_sample_index) > 0) and total_trials > 0 and step < saved_depths.shape[0]:
            pivot_time = time.time()
            pivot_number += 1

            # Picking Random Pivots
            final_obs_inds = []
            obs_inds = np.random.choice(obs_array_sample_index, sample_n_candidate, replace=False)  ## [40, 80, 65, 98, 100, 149]

            while len(final_obs_inds) < 6:
                obs_inds = obs_inds.astype(int)
                if len(seen_pivot_coords)!=0:
                    final_obs_inds += [x for x in obs_inds if sum(LA.norm(np.array(obs_array[x][:2]) - np.array(seen_pivot_coords), axis = 1).reshape(-1,) < 2.1*threshold_radius) == 0]  ## twice of threshold is done so that the areas of two pivots dont overlap.
                    
                else:
                    final_obs_inds += [x for x in obs_inds]

                obs_inds = np.random.choice(obs_array_sample_index, 1, replace=False)
            obs_inds = final_obs_inds[:6]

            # print("finished sampling pivots")

            candidates = []

            def process_observation(runner: TestRunner, heights: list, i, obs_array: np.ndarray, obs_ind) -> Tuple[float, np.ndarray, np.ndarray, float]:
                g1 = obs_array[obs_ind]
                p1 = np.array(runner.get_pos_grid(g1[:2], heights[i], meters_per_pixel=0.01))

                rotation = R.from_euler("xyz", [0, g1[2], 0], degrees=True).as_quat() 
  
                return heights[i], p1, rotation, g1[2]

            with concurrent.futures.ThreadPoolExecutor(max_workers=(3 * n_cores) // 4) as executor:

                features = [executor.submit(process_observation, runner, heights, i, obs_array, obs_ind) for obs_ind in obs_inds]

                for f in concurrent.futures.as_completed(features):
                    (height, pivot, rotation, angle) = f.result()
                    
                    #@TODO: This might increase gpu utilization but is not used
                    # pivot_grid = runner.get_grid_pos(pivot)
                    # ensure point is reachable and in the obs_array_index
                    new_pivot, _ = runner.get_random_point_near(height, grid=original_map, circle_center=pivot, radius_grid=1, pivot_list=pivot_array, pivot_indicies=obs_array_index,scalar_factor=10)
                    obs = runner.get_rgbd_from_pose(new_pivot, rotation)

                    # convert new_pivot to proper data type
                    # list object has no attribute squeeze
                    new_pivot = np.asarray(new_pivot)

                    candidate = (obs, pivot, rotation, angle) ### HERE pivot is a 3D point in the world
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
                        ax.add_patch(Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red"))
                        x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])

                        ax.add_patch(arrow(x0, y0, 100, candidates_rotation_rad, colors[ind]))
                  # make directory if not exist
                  directory_path = os.path.join("temp", "More_vis", scene, str(i), "candidates")
                  os.makedirs(directory_path, exist_ok=True)

                  # Save the figure
                  file_path = os.path.join(directory_path, f"candidates_{pivot_number}.png")
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

                    ax.add_patch(Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red"))
                    x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])
                    ax.add_patch(arrow(x0, y0, 100, candidates_rotation_rad, colors[ind]))

                fig.savefig(os.path.join("temp", "More_vis", scene, str(i), f"candidates_GA_{step}", "candidates.png"))
                plt.close()

                for ind, candidate in enumerate(candidates):
                    (obs, pivot, rotation, candidates_rotation_angle) = candidate
                    fig, ax = plt.subplots()

                    ax.set_title(f"{scene}_{color_map[colors[ind]]}")
                    ax.imshow(np.array(obs["color_sensor"]))

                    fig.savefig(os.path.join("temp", "More_vis", scene, str(i), f"obs_GA_{step}_{color_map[colors[ind]]}.png"))
                    plt.close()

            ##### HERE SKIPPING CANDIDATE SELECTION ########################################################################################################################

            candidates = []

            saved_depths = np.load(os.path.join("SpatialReasoningDataset1", "temp", "More_vis", scene, str(i), "saved_obs", "saved_dep.npy"), allow_pickle=True)
            saved_colors = np.load(os.path.join("SpatialReasoningDataset1", "temp", "More_vis", scene, str(i), "saved_obs", "saved_color.npy"), allow_pickle=True)
            saved_poses = np.load(os.path.join("SpatialReasoningDataset1", "temp", "More_vis", scene, str(i), "saved_obs", "saved_pose.npy"), allow_pickle=True)

            obs = {"depth_sensor": saved_depths[step].copy(),
                    "color_sensor": saved_colors[step].copy()}

            saved_pose_cand = saved_poses[step].copy()
            
            yaw = saved_pose_cand[-2]
            saved_pose_cand = saved_pose_cand[:3]

            saved_pose_cand[2] -= 1.50
            saved_pose_cand[1], saved_pose_cand[2] = saved_pose_cand[2], saved_pose_cand[1]

            rotation = R.from_euler("xyz", [0, yaw, 0], degrees=True).as_quat()

            candidate = (obs, saved_pose_cand, rotation, yaw)

            candidates.append(candidate)


            #################################################################################################################################################################

            # Pick the best
            (best_index, best_score, saved_depth, saved_color, saved_pose, rel_mat, grid_pivot, selected_flag, pcd1_slice) = pick_best_candidate(runner, original_map,  explored_map, stopping_map, candidates,
                                                                                                    saved_depth, saved_color, saved_pose, rel_mat, stuck, scene_name=scene,
                                                                                                    iter_number=i, init_unexplored=init_unexplored, floor_time=floor_time,
                                                                                                    scene_time=scene_time, pivot_time=pivot_time, pivot_number=step,
                                                                                                    threshold_radius = threshold_radius)
            # stuck = False
            # unstuck yourself
            if stuck == True:
                stuck = False

            # get only indicies of unexplored pixels (best_index is the index of the best candidate)
            if best_index is not None:
                seen_pivot_coords.append(grid_pivot[:2])
                distances = LA.norm(np.array(obs_all_coords) - np.array(grid_pivot[:2]), axis=1)
                exclude_indices = obs_array_index[distances < 2.1*threshold_radius].copy()
                obs_array_sample_index = np.setdiff1d(obs_array_sample_index, exclude_indices)

            else:
                trial_limit -= 1
                total_trials -= 1
                stuck = True if trial_limit <= 0 else False
                continue

            if plot_flag:
                fig, ax = plt.subplots()
                ax.set_title(f"Scene {scene} {pivot[1]}")
                ax.imshow(runner.get_top_down(height=pivot[1]))
                _, pivot, _, cand_degree = candidates[best_index]
                grid_pivot = runner.get_grid_pos(pivot)

                cand_radians = math.radians(cand_degree)
                x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])
                ax.add_patch(Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red"))
                ax.add_patch(arrow(x0, y0, 100, cand_radians, "green"))

                fig.savefig(os.path.join("temp", "More_vis", scene, str(i), f"step_{step}_best.png"))
                plt.close()

            world_lines, grid_lines = draw_sight_rays(runner, candidates, best_index, pivot[1], scene, i, step)
            obs_, pivot_, rotation_, cand_rad = candidates[best_index]
            explored_map, stopping_map, _ = update_map(runner, explored_map, stopping_map, world_lines, grid_lines, bounds, pivot_, scene, i, step,
                                            init_unexplored, floor_time, scene_time, pivot_time, plot_flag=True)

            
            np.save(os.path.join("temp", "More_vis", scene, str(i), "saved_obs", "explored_map.npy"), explored_map)
            np.save(os.path.join("temp", "More_vis", scene, str(i), "saved_obs", "stopping_map.npy"), stopping_map)

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

def process_scene(scene, floor_heights,scene_dir,resume_flag, scene_flag=None, floor_flag=None,SCENE_THRESHOLD=0.20):
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

    algorithm(runner, floor_maps, floor_heights, scene, SCENE_THRESHOLD, resume_flag=resume_flag, floor_flag=floor_flag)

    runner._sim.close()


def main(start=0, end=None, resume_flag=False, scene_flag=None, floor_flag=None):
    os.makedirs("temp/More_vis", exist_ok=True)
    os.makedirs("temp/batch_indx", exist_ok=True)

    SCENE_THRESHOLD = 0.20  # equivalent to 80% coverage
    gibson_fp = "igibson_floor_heights.json"

    with open(gibson_fp, "r") as j:
        json_file = json.load(j) #list of dicts

    # Determine which scenes are present in our dataset
    available_scenes = {filename[14:-4] for filename in glob.glob("gibson/gibson/*.glb")}
    available_scenes = sorted(list(available_scenes))

    start = max(0, start)
    end = len(available_scenes) if end == None else min(end, len(available_scenes))
    available_scenes = available_scenes[start:end]
    # available_scenes = ["Anaheim", "Applewood", "Goffs", "Mesic", "Sanctuary", "Silas"]
    available_scenes = ["Goffs"]
    
    print("available_scenes::", available_scenes)

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context= mp.get_context('spawn')) as executor:
        futures=[]
        for scene in json_file:
            if scene in available_scenes:
                scene_dir = os.path.join("temp/More_vis", str(scene))
                futures.append(executor.submit(process_scene, scene, json_file[scene],scene_dir,resume_flag,scene_flag, floor_flag, SCENE_THRESHOLD))

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