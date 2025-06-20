import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import math
import time
from matplotlib.patches import Arrow, Circle
import cv2 
import pickle
from scipy.spatial import ConvexHull

colors = ["b", "r", "k", "m", "y", "g", "c"]

def arrow(x0, y0, length, angle, color):
    x1 = math.cos(np.pi / 2 + angle) * length
    y1 = math.sin(np.pi / 2 + angle) * length

    return Arrow(x0, y0, x1, -y1, width=100.0, color=color)

def plot_topdown(floor_heights, runner):
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

def process_observation(runner, heights, i, obs_array, obs_ind):
    g1 = obs_array[obs_ind]
    p1 = np.array(runner.get_pos_grid(g1[:2], heights[i], meters_per_pixel=0.01))

    rotation = R.from_euler("xyz", [0, g1[2], 0], degrees=True).as_quat() 

    return heights[i], p1, rotation, g1[2]

def plot_obs_array(original_map, obs_array, vis_dir, i):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_map, cmap="gray")
    ax.scatter(obs_array[:, 0], obs_array[:, 1], c="r", s=1)
    plt.savefig(os.path.join(vis_dir, str(i), "pivot_points.png"))
    plt.close()

def plot_candidate(scene, heights, explored_map, candidates, colors, i, runner, pivot_number):
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
    directory_path = os.path.join("data_folder", "More_vis", scene, str(i), "candidates")
    os.makedirs(directory_path, exist_ok=True)

    # Save the figure
    file_path = os.path.join(directory_path, f"candidates_{pivot_number}.png")
    fig.savefig(file_path)

    # Close the figure to release resources
    plt.close(fig)

def plot_candidate_GA(scene, heights, explored_map, candidates, color_map, i, runner, step):
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
    
    os.makedirs(os.path.join("data_folder", "More_vis", scene, str(i), f"candidates_GA_{step}"), exist_ok=True)
    fig.savefig(os.path.join("data_folder", "More_vis", scene, str(i), f"candidates_GA_{step}", "candidates.png"))
    plt.close()

    for ind, candidate in enumerate(candidates):
        (obs, pivot, rotation, candidates_rotation_angle) = candidate
        fig, ax = plt.subplots()

        ax.set_title(f"{scene}_{color_map[colors[ind]]}")
        ax.imshow(np.array(obs["color_sensor"]))
        
        fig.savefig(os.path.join("data_folder", "More_vis", scene, str(i), f"obs_GA_{step}_{color_map[colors[ind]]}.png"))
        plt.close()

def plot_best_step(scene, pivot, runner, candidates, step):
    fig, ax = plt.subplots()
    ax.set_title(f"Scene {scene} {pivot[1]}")
    ax.imshow(runner.get_top_down(height=pivot[1]))
    _, pivot, _, cand_degree = candidates[best_index]
    grid_pivot = runner.get_grid_pos(pivot)

    cand_radians = math.radians(cand_degree)
    x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])
    ax.add_patch(Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red"))
    ax.add_patch(arrow(x0, y0, 100, cand_radians, "green"))

    fig.savefig(os.path.join("data_folder", "More_vis", scene, str(i), f"step_{step}_best.png"))
    plt.close()

def plot_update_map(topdown, x0, y0, cand_rad, grid_lines, scene, iter_number, pivot_number):
    fig, ax = plt.subplots()
    ax.imshow(topdown)
    ax.add_patch(arrow(x0, y0, 100, math.radians(cand_rad), "red"))
    for grid_line in grid_lines:
        x0, y0, px, py = grid_line
        ax.plot((x0, px), (y0, py), color="blue")

    fig.savefig(os.path.join("data_folder", "More_vis", scene, str(iter_number), f"update_test_{pivot_number}.png"))
    plt.close()

def plot_test_slice(depth_vis, scene_name, iter_number, i):
    plt.imshow(depth_vis)
    plt.savefig(os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), f"test_slice_{i}.png"))
    plt.close()

def plot_topdown(runner, height, patch):
    topdown = runner.get_top_down(height=height)
    ax.imshow(topdown)
    ax.add_patch(patch)

def plot_update_map(runner, pcd_candidate_slice, scene_name, pivot, explored_map, grid_pivot, pivot_time, scene_time, floor_time, stopping_map, init_unexplored, pivot_number, iter_number):
    points1 = [runner.get_grid_pos(point) for point in pcd_candidate_slice]
    xs = [int(i[0]) for i in points1]
    ys = [int(i[1]) for i in points1]

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

    fig.savefig(os.path.join("data_folder", "More_vis", scene_name, str(iter_number), f"update_map_{pivot_number}_fc.png"))

    plt.close()

def plot_update_map_check(scene_name, pivot, explored_map, iter_number, pivot_number):
    fig, ax = plt.subplots()
    ax.set_title(scene_name + " " + str(pivot[1]))
    ax.imshow(explored_map, cmap="gray")

    fig.savefig(os.path.join("data_folder", "More_vis", scene_name, str(iter_number), f"update_map_{pivot_number}.png"))
    plt.close()

def plot_update_candidate(scene_name, pivot_c1, temp_map_copy, iter_number, iter):
    fig, ax = plt.subplots()
    ax.set_title(scene_name + " " + str(pivot_c1[1]))
    ax.imshow(temp_map_copy, cmap="gray")

    fig.savefig(os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), f"update_map_{iter}_1.png"))
    plt.close()

def plot_step_star(scene_name, pivot_candidate, runner, g1, g2, candidates_rotation_rad_candidate, candidates_rotation_rad_save_pose, iter_number):
    length = 100
    fig, ax = plt.subplots()
    ax.set_title(scene_name + " " + str(pivot_candidate[1]))
    ax.imshow(runner.get_top_down(height=pivot_candidate[1]))
    x0, y0 = int(g1[0]), int(g1[1])
    ax.add_patch(Circle((int(g1[0]), int(g1[1])), radius=25, color="blue"))
    ax.add_patch(arrow(x0, y0, length, math.radians(candidates_rotation_rad_candidate), "green"))
    x1, y1 = int(g2[0]), int(g2[1])
    ax.add_patch(Circle((int(g2[0]), int(g2[1])), radius=25, color="red"))
    ax.add_patch(arrow(x1, y1, length, math.radians(candidates_rotation_rad_save_pose), "green"))
    fig.savefig(os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), f"step_{iter_number}_star.png"))
    plt.close()

def plot_update_map_iter(scene_name, pivot_saved, temp_map, iter_number, iter):
    fig, ax = plt.subplots()
    ax.set_title(scene_name + " " + str(pivot_saved[1]))
    ax.imshow(temp_map, cmap="gray")
    fig.savefig(os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number), f"update_map_{iter}_2.png"))
    plt.close()

def plot_near_point(scene_name, pivot, original_map, pivot_grid, new_pivot_grid, floor_no, iter_number):
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

def index_to_coords(indices, height_ds):
    x_coords = indices // height_ds
    y_coords = indices % height_ds
    return x_coords, y_coords

def highlight_regions(image, x_coords, y_coords, color, alpha=0.5):
    overlay = image.copy()
    for x, y in zip(x_coords, y_coords):
        cv2.circle(overlay, (int(y), int(x)), radius=10, color=color, thickness=-1)
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

def indices_to_original(indices, height_ds, rate):
    x_ds, y_ds = index_to_coords(indices, height_ds)
    x_orig = x_ds * rate
    y_orig = y_ds * rate
    return x_orig, y_orig
    
def plot_depth_intersect(obs_candidate, obs_saved, intersection_indices_cand, intersection_indices_svd, downsample_rate, scene_name, iter_number):
    H, W, C = obs_candidate['color_sensor'].shape
    down_H, down_W = int(H/downsample_rate), int(W/downsample_rate)

    cand_x, cand_y = index_to_coords(intersection_indices_cand, down_H)
    svd_x, svd_y = index_to_coords(intersection_indices_svd, down_H)   

    cand_x_orig, cand_y_orig = indices_to_original(intersection_indices_cand, down_H, downsample_rate)
    svd_x_orig, svd_y_orig = indices_to_original(intersection_indices_svd, down_H, downsample_rate)
    
    obs_candidate_highlighted = highlight_regions(obs_candidate['color_sensor'], cand_x_orig, cand_y_orig, (0, 255, 0))  # green
    obs_saved_highlighted = highlight_regions(obs_saved['color_sensor'], svd_x_orig, svd_y_orig, (0, 255, 0))  # green
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(obs_candidate_highlighted)
    axes[0].set_title('Highlighted Candidate Image')
    axes[0].axis('off') 

    axes[1].imshow(obs_saved_highlighted)
    axes[1].set_title('Highlighted Saved Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(
        "data_folder", "More_vis", str(scene_name), str(iter_number), "temp.png"
        ), bbox_inches='tight', pad_inches=0)
    plt.close()
 
def generate_convex_mask_indices(W, H, x_coords, y_coords):
    # Combine x and y into points
    points = np.column_stack((x_coords, y_coords))

    # Compute convex
    hull = ConvexHull(points)
    hull_points = points[hull.vertices].astype(int)  # 凸包的顶点

    # Creat all zero mask
    mask = np.zeros((H, W), dtype=np.uint8)

    # within convex is 255
    cv2.fillPoly(mask, [hull_points], 255)

    # find all 255
    mask_indices = np.argwhere(mask == 255)

    # take indices
    y_indices, x_indices = mask_indices[:, 0], mask_indices[:, 1]

    return x_indices, y_indices

def plot_intersect(obs_candidate, obs_saved, intersection_indices_cand, intersection_indices_svd, downsample_rate, scene_name, iter_number, ind_cand, ind_svd):
    H, W, C = obs_candidate['color_sensor'].shape
    down_H, down_W = int(H/downsample_rate), int(W/downsample_rate)

    cand_x, cand_y = index_to_coords(intersection_indices_cand, down_W)
    svd_x, svd_y = index_to_coords(intersection_indices_svd, down_W)   

    cand_x_orig, cand_y_orig = indices_to_original(intersection_indices_cand, down_W, downsample_rate)
    svd_x_orig, svd_y_orig = indices_to_original(intersection_indices_svd, down_W, downsample_rate)

    try:
        cand_x_orig, cand_y_orig = generate_convex_mask_indices(W, H, cand_x_orig, cand_y_orig)
    except Exception as e:
        print(f'error: {e}')
        print(f'cannot generate convex cand mask indices. cand_x_orig:{cand_x_orig}, cand_y_orig:{cand_y_orig}, {str(ind_cand)}, {str(ind_svd)} ')
    try:
        svd_x_orig, svd_y_orig = generate_convex_mask_indices(W, H, svd_x_orig, svd_y_orig)
    except Exception as e:
        print(f'error: {e}')
        print(f'cannot generate convex svg mask indices. svd_x_orig:{svd_x_orig}, svd_y_orig:{svd_y_orig},{str(ind_cand)}, {str(ind_svd)}')
    # print("cand_x_orig::::",cand_x_orig[:10])

    coordinate_dict = {
        "cand_x_orig": cand_x_orig,
        "cand_y_orig": cand_y_orig,
        "svd_x_orig": svd_x_orig,
        "svd_y_orig": svd_y_orig
    }
    with open(os.path.join("data_folder", "batch_indx", str(scene_name), str(iter_number), "Correspondance_"+str(ind_cand)+"_"+str(ind_svd)+".pkl"), "wb") as f:
        pickle.dump(coordinate_dict, f)

    obs_candidate_highlighted = highlight_regions(obs_candidate['color_sensor'], cand_x_orig, cand_y_orig, (0, 255, 0))  # green
    obs_saved_highlighted = highlight_regions(obs_saved['color_sensor'], svd_x_orig, svd_y_orig, (0, 255, 0))  # green
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(obs_candidate_highlighted)
    axes[0].set_title('Highlighted Candidate Image')
    axes[0].axis('off') 

    axes[1].imshow(obs_saved_highlighted)
    axes[1].set_title('Highlighted Saved Image')
    axes[1].axis('off')

    plt.tight_layout()

    # display using visualize_graph
    rel_mat_img_dir = os.path.join("data_folder", "More_vis", str(scene_name), str(iter_number),"rel_mats_imgs")
    os.makedirs(rel_mat_img_dir, exist_ok=True)
    plt.savefig(os.path.join(
        "data_folder", "More_vis", str(scene_name), str(iter_number), "rel_mats_imgs", "Correspondance_"+str(ind_cand)+"_"+str(ind_svd)+".png"
        ), bbox_inches='tight', pad_inches=0)
    
    plt.close()
 