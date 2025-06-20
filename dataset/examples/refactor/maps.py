import numpy as np
from folder_structure import FolderStructure
from typing import List, Tuple, Dict, Union
from time import time
import os
from config import *
from pivot_candidates import pivot_candidates
from plotter import *
from test_runner import TestRunner
import math
import concurrent.futures
from copy import copy
from numpy import linalg as LA

from helper import *
from scipy.spatial import distance_matrix
from floor import TimeWrapper
from scipy.spatial.transform import Rotation as R



class Maps(FolderStructure):
      def __init__(self, scene_name: str, floor_index: int, original_map: np.ndarray, height: float, resume_flag: bool = False) -> None:
            # initialize the FolderStructure directories
            super().__init__(scene_name, floor_index)

            # initialize the config variables
            self.SCENE_THRESHOLD = floor["SCENE_THRESHOLD"]

            # initialize the maps
            self.init_maps_candidates_lines()
            self.original_map = original_map
            self.init_unexplored = np.sum(self.original_map == 0)
            self.sum_original_map=np.sum(original_map)


            # Extra variables
            self.height = height
            self.pivot_number = 0
            self.scene_floor_string = f"Scene {scene_name} Floor {floor_index} "
            self.sight_rays_max_workers = 8
            self.is_valid = True
            self.pivot_time=-1


            # skip map if it is too small
            if not self.size_check():
                  print(self.scene_floor_string, "is too small, skipping")
                  self.is_valid = False
                  return
            
            # initialize the explored and stopping maps
            if resume_flag:
                  # if we can't load the explored and stopping maps, initialize them to the original map
                  if not self.load_explored_stopping_maps():
                        self.init_explored_stopping_maps()

                  # ensure mapsize is good
                  if self.coverage_sanity_check():
                        self.is_valid = False
                        return
            else:
                  # initialize the explored and stopping maps
                  self.init_explored_stopping_maps()

            # ensure mapsize is good
            if self.coverage_sanity_check():
                  self.is_valid = False
                  return

            # start the timer
            self._init_start_time()

            # initialize the pivot array
            self._init_pivot_array()


            # DONE HERE
            # rest of logic done in Floor class


      def sample_pivots(self, sample_n_candidate=6, distance_from_existing_pivots=100, plot_flag=False) -> np.ndarray:
            """
            Sample sample_n_candidate pivots=6 from obs_array_index. Ensures all sampled pivots are at least 100 pixels away from previously selected pivots other (or 75 with a very low probability).

            Arguments:
                  - sample_n_candidate: number of pivots to sample, default 6
                  - distance_from_existing_pivots: minimum distance between pivots, default 100
                  - plot_flag: if True, plots the pivots on the original map @TODO: add this logic

            Returns:
                  - obs_inds: NumPy array of indices of the obs_array, shape (sample_n_candidate,), each element in [0..len(pivot_array))

            """
            init_pivot_time=time()
            self.pivot_number += 1
            final_obs_inds =[]


            # Replace should be false, we manually remove the selected indicie from self.obs_array_index of the selected candidate
            obs_inds=np.random.choice(
                  self.obs_array_index,sample_n_candidate, replace=False
            ).astype(int)

            while len(final_obs_inds) < 6:

                  # Get the coordinates of the selected candidates
                  obs_coords = self.obs_array[obs_inds][:, :3]

                  # get the coordinates of the previously selected candidates
                  prev_obs_coords = self.obs_array[self.seen_array][:, :3]
                  

                  # Calculate distances
                  dist_matrix=distance_matrix(obs_coords, prev_obs_coords)

                  dist_matrix=dist_matrix[:,:-1]

                  # get only the indicies whre distance is greater than 100
                  mask=np.all(dist_matrix>distance_from_existing_pivots, axis=1)

                  # randomly allow some obs_inds to be less than 100
                  if np.random.rand() < 0.01:
                        mask = np.any(dist_matrix > 75, axis=1)

                  
                  obs_inds=obs_inds[mask]

                  if len(obs_inds) > 0:
                        final_obs_inds.append(obs_inds[0])
                        obs_inds=np.array([], dtype=int)
                  obs_inds=np.random.choice(
                        self.obs_array_index, 1, replace=False
                  ).astype(int)
            obs_inds=np.array(final_obs_inds[:6], dtype=int)
            self.pivot_time=time()-init_pivot_time



            return obs_inds




      def _init_pivot_array(self, plot_flag=False) -> None:
            """
            Initializes:
                  - pivot_array: NumPy array of pivot candidates [[x,y,corner_index], ...], where corner_index is in [0..7]
                  - obs_array_index: NumPy array of indices of the obs_array
                  - obs_array: NumPy array of candidate points [[x,y,theta], ...], where theta is in [0..360)
                  - seen_array: NumPy array of seen indices of the obs_array


            If plot_flag is True,
            Plots:
                  - pivot_candidates: pivot candidates on the original map [x,y,corner_index]
                  - pivot_points: pivot points on the original map [x,y]
            """
            # NumPy array of pivot candidates [[x,y,corner_index], ...], where corner_index is in [0..7]
            self.pivot_array = pivot_candidates(self.original_map)
            if plot_flag:
                  # plot the pivot_candidates
                  plot_pivot_candidates(self.original_map,self.pivot_array, self.vis_floor_dir,self.pivot_number)

            # generate the pivot_candidates
            obs_array=set()
            downsample=10 #degree
            scaler_factor=int(len(self.pivot_array/120))
            if scaler_factor >50:
                  scaler_factor = 50


            for pivot_ in self.pivot_array:
                  angle_r = np.arange(0,360, downsample)
                  for angle in angle_r:
                        if in_map(self.original_map.shape[0], self.original_map.shape[1], pivot_[0], pivot_[1], angle):
                              obs_array.add((int(pivot_[0]), int(pivot_[1]), angle))
            obs_array = np.array(list(set(obs_array)))

            self.obs_array_index = np.arange(len(obs_array), dtype=int)
            obs_array[:, [0,1]]= obs_array[:, [1,0]] # swap x and y positions directly in the NumPy array
            self.obs_array = obs_array

            self.seen_array = np.array([], dtype=int)
            if plot_flag:
                  plot_pivot_poins(self.original_map, obs_array, self.vis_floor_dir)



      def init_explored_stopping_maps(self) -> None:
            """
            Initializes the explored_map and stopping_map attributes to the original_map.
            """
            self.explored_map = self.original_map.copy()
            self.stopping_map = self.original_map.copy()

      def coverage_sanity_check(self) -> bool:
            """
            Returns True if the current floor's explored area is larger than the threshold.
            """
            if self.get_current_floor_coverage() >= self.SCENE_THRESHOLD:
                  print(self.scene_floor_string, "is already explored, skipping")
                  return True
            return False

      def init_maps_candidates_lines(self) -> None:
            """
            Initializes the explored_map, stopping_map, pivot_candidates, world_lines, and grid_lines attributes to empty NumPy arrays.
            """
            self.explored_map = np.array([], dtype=np.float64)
            self.stopping_map = np.array([], dtype=np.float64)
            
            self.pivot_candidates = np.array([])
            # self.world_lines = np.array([])
            # self.grid_lines = np.array([])


      def size_check(self, size=200) -> bool:
            """
            Returns True if the map's explorable area is larger than the given size.
            """
            return self.init_unexplored > size
      

      def load_explored_stopping_maps(self) -> bool:
            """
            Loads the explored_map and stopping_map for a given scene and floor index.
            """
            try:
                  self.explored_map = np.load(os.path.join(self.saved_obs_dir, "explored_map.npy"))
                  self.stopping_map = np.load(os.path.join(self.saved_obs_dir, "stopping_map.npy"))
                  return True
            except:
                  print(self.scene_floor_string, "failed to load explored_map.npy and stopping_map.npy")
                  pass
            return False

      def _init_start_time(self) -> None:
            """
            Initializes the start_time and end_time attributes to the current time and -1, respectively.
            """
            self.start_time = time()
            self.end_time=-1

      def set_end_time(self) -> None:
            self.end_time = time()

      def get_floor_generation_time(self) -> float:
            """
            Returns the time it took to generate the floor, or the time taken from the start of generating the floor until now.
            """
            if self.end_time == -1:
                  return time() - self.start_time
            return self.end_time - self.start_time










      def _depth2grid(self, i, angle_range, pivot_, depth_slice, x0, y0) -> Tuple:
            return depth2grid(i, angle_range, pivot_, depth_slice, x0, y0)
            
      # @TODO: Debug this function, concurrency could cause problems with the raycast.
      def draw_sight_rays(self, runner_grid_pos, runner_bounds, runner_top_down, candidate_at_best_index, pivot_number,plot_flag=False) -> Tuple[np.ndarray, np.ndarray]:
            """
            Draws sight rays (does raycast) for a single pivot and returns the resulting grid and top-down maps.

            Parameters:
                  - runner_grid_pos: runner's grid position [x,y]
                  - runner_bounds: runner's bounds [x_min, x_max, y_min, y_max]
                  - runner_top_down: runner's top-down map
                  - candidate_at_best_index: candidate at the best index [obs, pivot, rotation, candidate_radius]
                  - pivot_number: pivot number
                  - plot_flag: whether to plot the resulting maps
            Returns:
                  - world_lines: world lines (np.ndarray)
                  - grid_lines: grid lines (np.ndarray) 
            """
            obs_, pivot_, rotation_, cand_rad = candidate_at_best_index

            depth_img = obs_["depth_sensor"]

            # take slice of depth image at sensor height (center of image)
            slice_ind = depth_img.shape[0] // 2
            depth_vis = np.zeros(depth_img.shape)

            # @TODO: see if this is right
            depth_vis[slice_ind] = depth_img[slice_ind]

            # Assign each pixel to an angle
            fov_2=45

            left_angle = math.radians(cand_rad - fov_2)
            right_angle = math.radians(cand_rad + fov_2)

            angle_range=np.linspace(left_angle, right_angle, depth_img.shape[1])
            # @TODO: check this logic (why depth_img not depth_vis?)
            depth_slice = depth_img[slice_ind]

            # Pivot
            x0, y0 = runner_grid_pos

            bounds = runner_bounds
            meters_per_pixel=0.01

            # collect lines
            world_lines = np.array([])
            grid_lines = np.array([])

            if plot_flag:
                  patch = Circle((x0, y0), radius=25, color="red")
                  plot_sight_rays(runner_top_down,patch,self.vis_floor_dir,pivot_number)

            # reverse the order of angles so that the rays are drawn from left to right
            angle_range = angle_range[::-1]

            i_range = np.arange(len(angle_range))
            tmp_args = [(i, angle_range, pivot_, depth_slice, x0, y0) for i in i_range]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.sight_rays_max_workers) as executor:
                  # use map to preserve order
                  for ar in executor.map(depth2grid,tmp_args):
                        world_lines = np.append(world_lines, ar[0])
                        grid_lines = np.append(grid_lines, ar[1])

            if plot_flag:
                  plot_sight_rays_all(runner_top_down,arrow(x0,y0,100,math.radians(cand_rad),color="red"),grid_lines,self.vis_floor_dir,pivot_number)

            return world_lines, grid_lines


      def update_map_check(self, explored_map,stopping_map,runner_grid_pos, pivot,grid_lines,world_lines, plot_flag=False) -> Tuple[np.ndarray, np.ndarray, int, int]:
            """
            Updates the explored_map and stopping_map for a given pivot and returns the resulting maps and counts of obstacles and open spaces.
            Called by intersect_score_simple().
            Calls update_map_ray_check().

            Parameters:
                  - runner_grid_pos: runner's grid position [x,y]
                  - pivot: pivot @TODO: check data structure
                  - plot_flag: whether to plot the resulting maps

            Returns:
                  - explored_map: explored map (np.ndarray)
                  - stopping_map: stopping map (np.ndarray)
                  - count: count of obstacles
                  - o_count: count of open spaces
            """
            count=0
            o_count=0
            # pcd_grid=self.grid_lines
            grid_pivot = runner_grid_pos
            explored_map_copy = copy(explored_map)
            explored_map = explored_map_copy
            stopping_map = copy(stopping_map)

            # @TODO: check this logic
            if explored_map[int(grid_pivot[1])][int(grid_pivot[0])] == 1:
                  explored_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0.5
                  stopping_map[int(grid_pivot[1])][int(grid_pivot[0])] = 0
            elif explored_map[int(grid_pivot[1])][int(grid_pivot[0])] == 0.5:
                  o_count += 1
            count += 1

            for idx, _ in enumerate(world_lines):
                  explored_map, stopping_map, count_temp, o_count_temp = self.update_map_ray_check(explored_map_copy, idx,grid_pivot, grid_lines)
                  count += count_temp
                  o_count += o_count_temp

            if plot_flag:
                  plot_update_map_check(self.scene_name,pivot[1],self.explored_map,self.vis_floor_dir, len(world_lines))
            
            return explored_map, stopping_map, count, o_count



      # def update_map_ray_helper(self, explored_map, stopping_map, grid_pcl, grid_pivot, step_size, view_range) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
      #       return explored_map, stopping_map, grid_pcl, grid_pivot, False
      

      def update_map_ray_check(self, explored_map_copy: np.ndarray, grid_line_index:int, grid_pivot, grid_lines, step_size=1, view_range=80000, plot_flag=True) -> Tuple[np.ndarray, np.ndarray, int, int]:
            """
            Updates the explored_map and stopping_map with a ray from the pivot point to the point cloud grid, and returns the resulting maps and counts of obstacles and open spaces.
            Called by update_map_check().


            Parameters:
                  - explored_map_copy: copy of the explored map (np.ndarray), needs to be a copy because it is modified only to check for overlap
                  - grid_line_index: index of the grid line to check
                  - grid_pivot: pivot point of the grid @TODO: check this parameter
                  - step_size: step size for the ray
                  - view_range: view range for the ray
                  - plot_flag: whether to plot the resulting maps
            
            Returns:
                  - explored_map: explored map (np.ndarray)
                  - stopping_map: stopping map (np.ndarray)
                  - count: count of obstacles
                  - o_count: count of open spaces
            """
            count, o_count=0,0

            explored_map, stopping_map, grid_rays, explored_mask, inbound, in_view_range = update_map_ray_helper(explored_map_copy, self.stopping_map.copy(), grid_lines[grid_line_index], grid_pivot, step_size, view_range)
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
                  np.array([int(grid_lines[1]), int(grid_lines[0])]),
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

      def update_map_ray(self, explored_map, stopping_map, count_set, pcl_grid_idx, runner_grid_pos,grid_lines, step_size=1, view_range=80000) -> Tuple[np.ndarray, np.ndarray, set]:
            """
            Updates the explored_map and stopping_map with a single ray from the pivot point to the point cloud grid, and returns the resulting maps and counts of obstacles and open spaces. NOTE: this is different from update_map_ray_check() in that it does not check for overlap. We also don't do this in-place as the first call is used for the intersect_score calculation, the second is to update the map.
            Called by update_map(). 

            Parameters:
                  - explored_map: explored map (np.ndarray)
                  - stopping_map: stopping map (np.ndarray)
                  - count_set: set of counts of obstacles and open spaces
                  - pcl_grid_idx: index of the grid line to check
                  - runner_grid_pos: pivot point of the grid @TODO: check this parameter
                  - step_size: step size for the ray
                  - view_range: view range for the ray

            Returns:
                  - explored_map: explored map (np.ndarray)
                  - stopping_map: stopping map (np.ndarray)
                  - count_set: set of counts of obstacles and open spaces


            """
            grid_pcl = np.array(grid_lines[pcl_grid_idx][2:4])
            grid_pivot = np.array(runner_grid_pos)

            dist = LA.norm(grid_pcl-grid_pivot)
            cos_theta,sin_theta=np.divide(grid_pcl-grid_pivot,dist)

            steps = np.arange(step_size, dist, step_size)

            grid_rays = np.column_stack(
                  [grid_pivot[0] + steps * cos_theta, grid_pivot[1] + steps * sin_theta]
            )

            # limit grid_rays to range of valid indices
            grid_rays = np.clip(grid_rays, [0, 0], [self.explored_map.shape[1] - 1, self.explored_map.shape[0] - 1])


            # check inbound and in view range
            inbound = (
                  (grid_rays[:, 0] >= 0)
                  & (grid_rays[:, 0] < self.explored_map.shape[1])
                  & (grid_rays[:, 1] >= 0)
                  & (grid_rays[:, 1] < self.explored_map.shape[0])
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
      
      def update_map(self, explored_map: np.ndarray, stopping_map: np.ndarray, world_lines: np.ndarray, grid_lines: np.ndarray, runner_grid_pos, pivot, scene_time: float,floor_time:float, pivot_time: float, iter_number: int, plot_flag=False) -> Tuple[np.ndarray, np.ndarray, int]:
            """
            Updates the explored_map and stopping_map with all rays for a pivot point, and returns the number of new open spaces. Wrapper for update_map_ray(), calls it for each grid line.
            Called by intersect_score_simple(), and algorithm.py.

            NOTE: modifications are not made in-place to the maps, but rather to copies of the maps. This is because the first call to update_map_ray() is used for the intersect_score calculation, the second is to update the map.

            Parameters:
                  - explored_map: explored map (np.ndarray)
                  - stopping_map: stopping map (np.ndarray)
                  - world_lines: world lines (np.ndarray)
                  - runner_grid_pos: pivot point of the grid
                  - pivot: pivot point of the runner
                  - scene_time: time of all floors
                  - floor_time: floor of this floor
                  - pivot_time: time of the pivot
                  - iter_number: iteration number of pivot selection
                  - plot_flag: flag to plot the maps
            
            Returns:
                  - explored_map: explored map (np.ndarray)
                  - stopping_map: stopping map (np.ndarray)
                  - len(count_set): number of new open spaces
            """
            count_set = set()

            pcd_grid = grid_lines
            grid_pivot = runner_grid_pos.astype(int)
            explored_map[grid_pivot[1], grid_pivot[0]] = 0.5
            stopping_map[grid_pivot[1], grid_pivot[0]] = 0
            count_set.add((grid_pivot[1],grid_pivot[0])) #@TODO: Check this logic, why swap?

            for idx in range(len(world_lines)):
                  explored_map, stopping_map, count_set = self.update_map_ray(
                        explored_map, stopping_map, count_set, pcd_grid[idx], grid_pivot,grid_lines)
                  

            if plot_flag:
                  # @TODO: fix scene time
                  plot_update_map(explored_map, pivot, grid_pivot,self.scene_name,self.pivot_number, pivot_time, scene_time, floor_time, self.get_current_floor_coverage(), self.vis_floor_dir,iter_number)

            return explored_map, stopping_map, len(count_set)
      
      def save_explored_stopping_maps(self, explored_map, stopping_map, scene_time, pivot_time) -> None:
            try:
                  np.save(os.path.join(self.saved_obs_dir, f"explored_map_{scene_time}_{pivot_time}.npy"), explored_map)
                  np.save(os.path.join(self.saved_obs_dir, f"stopping_map_{scene_time}_{pivot_time}.npy"), stopping_map)
            except:
                  print("Error saving maps")
                  



      def get_current_floor_coverage(self) -> float:
            """
            Returns the current floor coverage as a percentage. (0.0 - 1.0)
            """
            return (1- (np.sum(self.stopping_map) / np.sum(self.init_unexplored)))
      
