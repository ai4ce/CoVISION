from settings import default_sim_settings, make_cfg
import os

import numpy as np
import torch
from open3d import geometry, camera

import habitat_sim
import habitat_sim.utils.datasets_download as data_downloader
import matplotlib.pyplot as plt
import random
import time
# import quaternion as q

class TestRunner:
    def __init__(self, sim_settings) -> None:
        self._sim_settings = sim_settings
        self._cfg = make_cfg(self._sim_settings)
        scene_file = self._sim_settings["scene"]
        if (
            not os.path.exists(scene_file)
            and scene_file == default_sim_settings["scene"]
        ):
            print(
                "Test scenes not downloaded locally, downloading and extracting now..."
            )
            data_downloader.main(["--uiplot_flag::ds", "habitat_test_scenes"])
            print("Downloaded and extracted test scenes data.")

        # create a simulator (Simulator python class object, not the backend simulator)
        self._sim = habitat_sim.Simulator(self._cfg)
        
        # set seeds for np, random, simulator
        random.seed(self._sim_settings["seed"])
        np.random.seed(self._sim_settings['seed'])
        self._sim.seed(self._sim_settings["seed"])
        


    def get_top_down(self, meters_per_pixel=0.01, height=0) -> np.ndarray:
        return self._sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    
    def get_islands(self, meters_per_pixel=0.01, height=0):
        return self._sim.pathfinder.get_topdown_island_view(meters_per_pixel, height)

    def get_grid_pos(self, pos, meters_per_pixel=0.01):
        '''
        pos -> grid
        '''
        bounds = self._sim.pathfinder.get_bounds() # returns minimum and maximum coordinates of the navigation mesh
        
        # given that index 0 is x, index 2 is y
        px = (pos[0] - bounds[0][0]) / meters_per_pixel        
        py = (pos[2] - bounds[0][2]) / meters_per_pixel
        return [int(px), int(py)]
    
    def get_pos_grid(self, grid_point, height, meters_per_pixel=0.01):
        
        bounds = self._sim.pathfinder.get_bounds() # returns minimum and maximum coordinates of the navigation mesh
        
        # print("grid", grid)
        # print("height", height)
        
        # given that index 0 is x, index 2 is y
        posX = grid_point[0] * meters_per_pixel + bounds[0][0]
        posY = grid_point[1] * meters_per_pixel + bounds[0][2]
        return [posX, height, posY]


    def is_navigable_grid(self, grid, x, y):
        
        '''
        given gridpoints, check if navigable inside grid
        
        note that grid has inverted x and y so account for that in call
        
        returns True if navigable, False otherwise
        '''
        valid_indices = (x >= 0) & (x < grid.shape[1]) & (y >= 0) & (y < grid.shape[0])
        return grid[y[valid_indices], x[valid_indices]] == 1.0

        
        # return x >= 0 and x < grid.shape[1] and y >= 0 and y < grid.shape[0] and grid[y][x] == 1.0

    
    def get_random_point_near(self, height, grid, circle_center,radius_grid=1, meters_per_pixel=0.01,pivot_list=None,pivot_indicies=None,scalar_factor=200):
        '''
        returns a random navigable point within a square matrix with radius near the given point
        
        circle_center: [x,z,y] pos
        radius_grid       : grid
        
        returns:
            random_point_pos  : [x, height, y]
            random_point_grid : [y, x]
        '''
        circle_center_grid = self.get_grid_pos(circle_center)
        circleX, circleY = circle_center_grid
        x_dim, y_dim = grid.shape[1], grid.shape[0]

        pivot_array_tmp =np.array([])
        if pivot_list is not None:
            pivot_array = np.array(pivot_list)[:, :2]
            mask = (pivot_array[:, 0] >= circleX-radius_grid) & (pivot_array[:, 0] <= circleX+(radius_grid+21)) & (pivot_array[:, 1] >= circleY-radius_grid) & (pivot_array[:, 1] <= circleY+(radius_grid+21))
            pivot_array_tmp = pivot_array[mask]

        x_range = np.clip(
            np.arange(circleX-radius_grid, circleX+radius_grid+1),
            0,
            x_dim-1
        )
        y_range = np.clip(
            np.arange(circleY - radius_grid, circleY + radius_grid + 1),
            0,
            y_dim-1
        )

        xx, yy = np.meshgrid(x_range, y_range)
        xx, yy = xx.flatten(), yy.flatten()

        valid_mask = self.is_navigable_grid(grid, xx, yy)


        if pivot_list is not None:
            for pivot in pivot_array_tmp:
                res_=np.linalg.norm(np.stack([xx,yy], axis=1) - np.array(pivot), ord=1, axis=1)
                valid_mask &= (res_ >= 200)
            valid_points = np.column_stack([xx[valid_mask],yy[valid_mask]])

        else:
            valid_points = [[i,j] for i in x_range for j in y_range if self.is_navigable_grid(grid, i, j)]
        
        if len(valid_points) == 0:
            # print("radius increased to", radius_grid+2)
            return self.get_random_point_near(height, grid, circle_center,radius_grid=radius_grid+2)
        else:
            # print("len", len(valid_points))
            random_point_grid = random.choice(valid_points)
            if pivot_list is not None:
                    # ensure that the minimim distance to all indicies is greater than 100
                    min_tmp = np.inf
                    min_point = None
                    # generate random 
                    for pivot in pivot_array_tmp:
                        random_point_grid = random.choice(valid_points)
                        old_min_tmp = min_tmp
                        min_tmp = np.min([np.linalg.norm(np.array(random_point_grid) - np.array(pivot)),min_tmp])
                        #print("min_tmp",min_tmp)
                        if min_tmp < old_min_tmp:
                            min_point = random_point_grid

                    if min_point is not None:
                        random_point_grid=min_point
                        
                    if min_tmp >= 200 and min_tmp != np.inf:
                        # print("min_tmp_",min_tmp)

                        return self.get_pos_grid(random_point_grid, height), random_point_grid
                    # print("radius increased to", radius_grid+2)
                    return self.get_random_point_near(height, grid, circle_center,radius_grid=radius_grid+2)

            return self.get_pos_grid(random_point_grid, height), random_point_grid
    
    def get_rgbd_from_pose(self, position, rotation):
        agent = self._sim.initialize_agent(0)
        state = agent.get_state()
        state.position = position
        state.rotation = rotation
        agent.set_state(state)
        observation = self._sim.get_sensor_observations()
        return observation

    def convert_obs_to_pcd(self, obs, position, rotation, scene_name,iter_number, stride=2, width=256, height=256, hfov=np.pi/2, plot_flag=None):
        depth = geometry.Image(obs["depth_sensor"])
        #depth = geometry.Image(obs["equirect_depth_sensor"])
        fx = width / (2 * np.tan(hfov / 2))
        fy = fx / width * height
        intrinsic = camera.PinholeCameraIntrinsic(width, height, fx, fy, width/2, height/2)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = geometry.get_rotation_matrix_from_quaternion(rotation)
        # extrinsic[:3, :3] = self.quaternion_rotation_matrix(rotation)
        extrinsic[:3, 3] = position

        if not os.path.exists("temp/More_vis/"+scene_name+"/"+str(iter_number)+"/depth"):
            os.mkdir("temp/More_vis/"+scene_name+"/"+str(iter_number)+"/depth")

            
        pcd = geometry.PointCloud.create_from_depth_image(depth, intrinsic, extrinsic, depth_scale=1, stride=stride)
        return pcd

    def chamfer_dst(self, pcd1, pcd2):
        pcd1 = torch.tensor(np.asarray(pcd1.points), dtype=torch.float16)
        pcd2 = torch.tensor(np.asarray(pcd2.points), dtype=torch.float16)

        pcd1 = pcd1.unsqueeze_(0)
        pcd2 = pcd2.unsqueeze_(1)
        D = torch.min(torch.norm(pcd1 - pcd2, dim=-1), dim=0)
        return D

    def get_pivot_point(self, prevPivot=None, height=None, random_flag=True, explored_map=None, original_map=None, stuck=False):#, pivot_coordinates=[]):
        # Purple = False (not_navigable), Yellow = True (navigable)
        
        if height == None:
            return self._sim.pathfinder.get_random_navigable_point()
        # Need to make sure point is navigable
        # at both agent height and sensor height
        # random_flag only enables at the first epoch, search pivot point in unexplore area 
        # stuck enables if the threshold does not increase for several epochs, search pivot point in unexplore area 
        # explored_map is the explored grid
        # total_map is the total map
        
        # force choosing from given floor
        lower = height - 0.25
        higher = height + 0.25

        num_tries = 0
        exploredRegion = np.count_nonzero(explored_map == 0.5)
        # print("exploredRegion::", exploredRegion)
        
        while True:
            if num_tries % 100 == 0: print("num_tries:",(num_tries))
            res = self._sim.pathfinder.get_random_navigable_point()
            grid_res = self.get_grid_pos(res)
            # print(random_flag, stuck)
            
            if (random_flag == True) or (stuck == True): # if we want to get a random new point
                if (explored_map[int(grid_res[1])][int(grid_res[0])] == True): # and it's in explored map, go to unexplored area
                    print("...picked a random pivot not in explored area")
                    break
            else: 
                if exploredRegion < 5000: # modify this threshold
                    # get within radius 
                    res = self._sim.pathfinder.get_random_navigable_point_near(circle_center=prevPivot,radius=1.0)
                    grid_res = self.get_grid_pos(res)
                    explored = explored_map[int(grid_res[1])][int(grid_res[0])] 
                    # print("prevPivot:", self.get_grid_pos(prevPivot))
                    # print("new grid:", grid_res)
                    break
                else: 
                    # get within explored region
                    explored = explored_map[int(grid_res[1])][int(grid_res[0])] 
                    if (explored==0.5): 
                        # we want to pick pivot within explored region, and it must be navigable in the original topdown map
                        print("...picked a pivot within explored region that is navigable in topdown map")
                        break
                
            num_tries += 1
        assert(self._sim.pathfinder.is_navigable(res))
        
        print("gridPivot", grid_res)
        
        return res, grid_res

    def get_correspondence(self, pcd1, pcd2, threshold=0.1):
        num_points = len(pcd1)
        dist = self.chamfer_dst(pcd1, pcd2)
        corr = dist[dist < threshold]
        return corr / num_points
