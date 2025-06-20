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
import math
from scipy.spatial.transform import Rotation as R
import open3d as o3d

import glfw
from OpenGL.GL import *

# import quaternion as q

class TestRunner:
    def __init__(self, sim_settings) -> None:
        #Clearing OpenGL depth buffer
        glClear(GL_DEPTH_BUFFER_BIT)
        print("Clearing OpenGL Depth Buffer")
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, 1920, 1080)
        print("Increasing depth buffer precision")
        # glEnable(GL_DEPTH_CLAMP)
        # print("Enabling depth clamp")

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

        ####Added by Taarun
        ####Checking and setting projection matrix values
        # print("inside get set depth sensor params")
        depth_sensor_obj = self._sim.agents[0]._sensors["depth_sensor"]
        # # color_sensor_obj = self._sim.agents[0]._sensors["color_sensor"]

        render_cam = depth_sensor_obj.render_camera
        # print("Before modifying zfar and znear values")
        # print(f"projection matrix:\n {np.array(render_cam.projection_matrix)}")
        # print(f"near: {depth_sensor_obj.near_plane_dist}, far: {depth_sensor_obj.far_plane_dist}")

        render_cam.set_projection_matrix(self._sim_settings["width"],
                                         self._sim_settings["height"],
                                         depth_sensor_obj.near_plane_dist+ 0.0001,  
                                         depth_sensor_obj.far_plane_dist, 
                                         depth_sensor_obj.hfov)

        # print("After modifying zfar and znear values")
        # print(f"Projection Matrix updated:\n {np.array(render_cam.projection_matrix)}")
 
        
        # # test creating an orthographic camera
        # self._sceneGraph = habitat_sim.SceneGraph()
        # self.rootNode = self._sceneGraph.get_root_node()
        # self._sceneNode = habitat_sim.SceneNode(self.rootNode)
        # self._sceneNode2 = habitat_sim.SceneNode(self._sceneNode)
        # self._sceneNode2.set_parent(self._sceneNode)
        
        # self._sceneNode.create_child()
        # self._cameraSensorSpec = habitat_sim.CameraSensorSpec()
        # self._cameraSensor = habitat_sim.sensor.CameraSensor(self._sceneNode2, self._cameraSensorSpec)
        # assert()

    def get_top_down(self, meters_per_pixel=0.01, height=0) -> np.ndarray:
        return self._sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    
    def get_islands(self, meters_per_pixel=0.01, height=0):
        return self._sim.pathfinder.get_topdown_island_view(meters_per_pixel, height)

    def get_grid_pos(self, pos, meters_per_pixel=0.01):
        '''
        pos -> grid
        '''
        bounds = self._sim.pathfinder.get_bounds() # returns minimum and maximum coordinates of the navigation mesh
        
        # given that index 0 is x, index 2 is z
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

        # np.random.seed(self._sim_settings["seed"])
        # random.seed(self._sim_settings["seed"])

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
        
        # valid_points =  [] # valid gridpoints in (y,x)
        # if pivot_list is not None:
        #     for i in x_range:
        #         for j in y_range:
        #             if not self.is_navigable_grid(grid, i, j): continue
        #             # ensure point exists in pivot list that are reachable by the pivot indicies 
        #             # pivot indicies need to get rid of angle (last column)
        #             if pivot_list is not None:
        #                 # ensure that the minimim distance to all indicies is greater than 100
        #                 for pivot in pivot_array_tmp:
        #                     res_=np.linalg.norm(np.array([i,j]) - np.array(pivot), ord=1)
        #                     if res_ < 200:
        #                         continue
        #             valid_points.append([i,j])
        else:
            valid_points = [[i,j] for i in x_range for j in y_range if self.is_navigable_grid(grid, i, j)]
        
        if len(valid_points) == 0:
            # print("radius increased to", radius_grid+2)
            return self.get_random_point_near(height, grid, circle_center,radius_grid=radius_grid+2)
        else:
            # print("len", len(valid_points))
            # random.seed(self._sim_settings["seed"])
            random_point_grid = random.choice(valid_points)
            if pivot_list is not None:
                    # ensure that the minimim distance to all indicies is greater than 100
                    min_tmp = np.inf
                    min_point = None
                    # generate random 
                    for pivot in pivot_array_tmp:
                        # random.seed(self._sim_settings["seed"])
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
        agent.set_use_mask(False)
        agent.set_state(state)
        observation = self._sim.get_sensor_observations()
        return observation

    def get_covariant_image(self, scene_name, id1, position1, rotation1, id2, position2, rotation2):
        os.makedirs(os.path.join("data_folder", "More_vis", "covariant_images"), exist_ok=True)
        agent = self._sim.initialize_agent(0)

        state = agent.get_state()
        state.sensor_states['color_sensor'].position = position1
        state.sensor_states['color_sensor'].rotation = rotation1
        agent.set_use_mask(False)
        agent.set_state(state, infer_sensor_states=False)
        compObs = self._sim.get_sensor_observations()

        state = agent.get_state()
        state.sensor_states['depth_sensor'].position = position1
        state.sensor_states['depth_sensor'].rotation = rotation1
        state.sensor_states['color_sensor'].position = position2
        state.sensor_states['color_sensor'].rotation = rotation2
        agent.set_use_mask(True)
        agent.set_state(state, infer_sensor_states=False)
        observation = self._sim.get_sensor_observations()
        # Draw and save images
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title(f"{scene_name}_origin_{id1}")
        ax[0].imshow(np.array(compObs["color_sensor"]))
        ax[1].set_title(f"{scene_name}_depth_{id1}_color_{id2}")
        ax[1].imshow(np.array(observation["color_sensor"]))
        fig.savefig(os.path.join("data_folder", "More_vis", "covariant_images", f"{scene_name}_depth_{id1}_color_{id2}.png"))
        plt.close()

    def convert_obs_to_pcd(self, obs, position, rotation, scene_name,iter_number, stride=2, width=1920, height=1080, hfov=np.pi/2, plot_flag=None):
        depth = geometry.Image(obs["depth_sensor"])
        #depth = geometry.Image(obs["equirect_depth_sensor"])
        fx = width / (2 * np.tan(hfov / 2))
        fy = fx / width * height
        intrinsic = camera.PinholeCameraIntrinsic(width, height, fx, fy, width/2, height/2)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = geometry.get_rotation_matrix_from_quaternion(rotation)
        # extrinsic[:3, :3] = self.quaternion_rotation_matrix(rotation)
        extrinsic[:3, 3] = position

        if not os.path.exists("data_folder/More_vis/"+scene_name+"/"+str(iter_number)+"/depth"):
            os.mkdir("data_folder/More_vis/"+scene_name+"/"+str(iter_number)+"/depth")
        
        # if plot_flag != None:
        #     fig, ax = plt.subplots()
        #     ax.set_title(scene_name)
        #     ax.imshow(np.array(depth))
        #         ##########

        #     fig.savefig("data_folder/More_vis/"+scene_name+"/"+str(iter_number)+"/depth/"+str(plot_flag)+"_depth.png")
        #     plt.close()
        pcd = geometry.PointCloud.create_from_depth_image(depth, intrinsic, extrinsic, depth_scale=1, stride=stride)
        return pcd

    def convert_obs_to_pcd_v2(self, obs, pose, hfov=90*(math.pi/180), W=1920, H=1080, downsample_rate=8): #scene_name, iter_number

        # print(f"pose: {pose}")
        x,y = np.meshgrid(np.linspace(-1,1,H), np.linspace(1,-1,W))
        x = x.reshape(1, H, W).astype(np.float64)
        y = y.reshape(1, H, W).astype(np.float64)

        depth = obs["depth_sensor"] # (1080, 1920)
        depth = depth.reshape(1, H, W).astype(np.float64)

        xyd = np.vstack((x * depth, y * depth, -depth, np.ones(depth.shape)))
        xyd = xyd[:, ::downsample_rate, ::downsample_rate]
        
        xyd = xyd.reshape(4,-1)

        K = np.array([
            [1/ np.tan(hfov/2.), 0., 0., 0.],
            [0., 1/ np.tan(hfov/2.), 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]], dtype = np.float64)

        xyd_cs = np.matmul(np.linalg.inv(K), xyd)
        
        translation = pose[:3] # X Z Y
        translation[1], translation[2] = translation[2], translation[1] # X Y Z
        # translation[1] = 1.5 + translation[1]

        rot_quat = R.from_euler("xyz", pose[3:], degrees=True).as_quat()
        rot_mat = np.array(R.from_quat(rot_quat).as_matrix())
        ext_mat = np.identity(4)
        ext_mat[:3, :3] = rot_mat
        ext_mat[:3, 3] = translation
        ext_mat = np.array(ext_mat, dtype=np.float64)

        xyd_ws = np.matmul(ext_mat, xyd_cs)
        xyd_ws = np.matmul(K, xyd_ws)
        xyd_ws = xyd_ws.T

        point_cloud = xyd_ws[:, :3].reshape(-1,3)

        return np.round(point_cloud, decimals = 2), pose

    # def calculate_iou_point_cloud(self, pcd1, pcd2):

    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     lower_limit = ((1080//2) - 1)*1920
    #     upper_limit = lower_limit + 1920

    #     pcd1_slice = pcd1[lower_limit:upper_limit, :3]
    #     pcd2_slice = pcd2[lower_limit:upper_limit, :3]

    #     pcd1_tensor = torch.tensor(pcd1_slice).to(device)
    #     pcd2_tensor = torch.tensor(pcd2_slice).to(device)
        
    #     intersection = torch.sum(torch.eq(pcd1_tensor[:, None, :], pcd2_tensor).all(dim = 2))
        
    #     union = pcd1_tensor.shape[0] + pcd2_tensor.shape[0] - intersection 
        
    #     iou = intersection.item()/union.item() if union.item() > 0 else 0.0
        
    #     # print("Intersection: {}, union: {}, iou: {}".format(intersection, union, iou))
        
    #     return intersection, union, iou

    # def calculate_iou_point_cloud_v2(self, pcd1, pcd2,pose1, pose2, thresh_val = 0.02):
    #     # print(f"Pose1: {pose1}")
    #     # print(f"Pose2: {pose2}")
    #     # print(f"IoU tolerance: {thresh_val}, V2")
    #     # torch.cuda.empty_cache()
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     # device = "cpu"
    #     # print("devices: ", torch.cuda.device_count())
    #     # print("device: ", device)
        
    #     # print(f"device count: {torch.cuda.device_count()}")

    #     pcd1_np = np.reshape(pcd1, (1080, 1920, 3))
    #     pcd2_np = np.reshape(pcd2, (1080, 1920, 3))

    #     pcd1_slice = pcd1_np[1080 // 2 - 5: 1080 // 2 + 6, :, :]
    #     pcd2_slice = pcd2_np[1080 // 2 - 5: 1080 // 2 + 6, :, :]

    #     pcd1_slice = pcd1_slice.reshape(-1, 3)
    #     pcd2_slice = pcd2_slice.reshape(-1, 3)

    #     pcd1_gpu = torch.tensor(pcd1_slice).to(device)
    #     pcd2_gpu = torch.tensor(pcd2_slice).to(device)

    #     chunk_size = 1000
    #     # Split the data into chunks
    #     num_chunks = (pcd1_gpu.size(0) + chunk_size - 1) // chunk_size
       
    #     # Initialize an empty tensor to store the results
    #     # euclidean_distance = torch.zeros(pcd1_gpu.size(0), pcd2_gpu.size(0)).cuda()

    #     intersection_points = []
    #     for i in range(num_chunks):
    #         start_idx = i * chunk_size
    #         end_idx = min((i + 1) * chunk_size, pcd1_gpu.size(0))
    #         chunk_pcd1 = pcd1_gpu[start_idx:end_idx]

    #         # Broadcasting still works as before, but now for smaller chunks
    #         chunk_distance = torch.sqrt(torch.sum((chunk_pcd1.unsqueeze(1) - pcd2_gpu) ** 2, dim=2))

    #         ###this piece of code is for intersection point viz
    #         for ri, r in enumerate(chunk_distance):
    #             indices = torch.nonzero(r < thresh_val).squeeze(dim=1)
    #             if indices.numel() > 0:
    #                 for j in indices:
    #                     intersection_points.append(pcd2_gpu[j, :])
    #         ###Intersection point viz ends here

    #         # euclidean_distance[start_idx:end_idx, :] = chunk_distance

    #     intersection_points = np.array([tensor.cpu().numpy() for tensor in intersection_points])
    #     # print("before: pcd3_int_points", pcd3_intersection_points.shape)
    #     intersection_points = np.asarray(list(set(tuple(point) for point in intersection_points)))
    #     # print("after: pcd3_int_points", pcd3_intersection_points.shape)

    #     intersection_count = intersection_points.shape[0]
    #     # print(intersection_count)

    #     union = pcd1_gpu.shape[0] + pcd2_gpu.shape[0] - intersection_count
    #     # print(union, type(union))
    #     iou = intersection_count / np.abs(union)
    #     # print(iou)

    #     # print(f"pose1: {pose1}, pose2: {pose2}, intersection: {intersection_count}, union: {union}, iou: {iou}")

    #     return intersection_count, union, iou

    # def calculate_iou_all_pcd(self, pcd1, pcd2,pose1,pose2, thresh_val=0.02):
    #     print(f"iou tolerance: {thresh_val}")
    #     # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    #     device = "cpu"
    
    #     pcd1_tl = pcd1.copy()
    #     pcd2_tl = pcd2.copy()

    #     pcd1_bl = pcd1.copy()
    #     pcd2_bl = pcd2.copy()

    #     pcd1_ml = pcd1.copy()
    #     pcd2_ml = pcd2.copy()

    #     pcd1_tl = np.reshape(pcd1_tl, (1080, 1920, 3))
    #     pcd1_sort_tl = np.flipud(np.sort(pcd1_tl, axis=0))
    #     pcd1_ftl = pcd1_sort_tl[:5, :, :]
    #     pcd1_ftl = np.reshape(pcd1_ftl, (-1, 3))

    #     pcd2_tl = np.reshape(pcd2_tl, (1080, 1920, 3))
    #     pcd2_sort_tl = np.flipud(np.sort(pcd2_tl, axis=0))
    #     pcd2_ftl = pcd2_sort_tl[:5, :, :]
    #     pcd2_ftl = np.reshape(pcd2_ftl, (-1, 3))

    #     pcd1_bl = np.reshape(pcd1_bl, (1080, 1920, 3))
    #     pcd1_sort_bl = np.sort(pcd1_bl, axis=0)
    #     pcd1_fbl = pcd1_sort_bl[:5, :, :]
    #     pcd1_fbl = np.reshape(pcd1_fbl, (-1, 3))

    #     pcd2_bl = np.reshape(pcd2_bl, (1080, 1920, 3))
    #     pcd2_sort_bl = np.sort(pcd2_bl, axis=0)
    #     pcd2_fbl = pcd2_sort_bl[:5, :, :]
    #     pcd2_fbl = np.reshape(pcd2_fbl, (-1, 3))

    #     pcd1_ml = np.reshape(pcd1_ml, (1080, 1920, 3))
    #     pcd2_ml = np.reshape(pcd2_ml, (1080, 1920, 3))

    #     # pcd1_slice_ml = pcd1_ml[1080 // 2 - 5: 1080 // 2 + 6, :, :]
    #     # pcd2_slice_ml = pcd2_ml[1080 // 2 - 5: 1080 // 2 + 6, :, :]

    #     pcd1_fml = pcd1_ml.reshape(-1, 3)
    #     pcd2_fml = pcd2_ml.reshape(-1, 3)

    #     # pcd1_final = np.vstack((pcd1_ftl, pcd1_fml, pcd1_fbl))
    #     # pcd2_final = np.vstack((pcd2_ftl, pcd2_fml, pcd2_fbl))
    #     pcd1_final = pcd1_fml
    #     pcd2_final = pcd2_fml
    #     pcd1_gpu = pcd1_final.tolist()
    #     pcd2_gpu = pcd2_final.tolist()

    #     pcd1_gpu = list(map(lambda x: tuple(x), pcd1_gpu))
    #     pcd1_gpu = set(pcd1_gpu)
        
    #     pcd2_gpu = list(map(lambda x: tuple(x), pcd2_gpu))
    #     pcd2_gpu = set(pcd2_gpu)

        
    #     intersection_points = pcd1_gpu.intersection(pcd2_gpu)
    #     intersection_count = len(intersection_points)
    #     union = len(pcd1_gpu.union(pcd2_gpu))
    #     iou = intersection_count / union
    #     if iou < 0:
    #         print("Something is wrong as iou cant be negative")
    #         exit()
    #     print(f"pose1: {pose1}, pose2: {pose2}, intersection: {intersection_count}, union: {union}, iou: {iou}")

    def calculate_iou_point_cloud_v3(self, pcd1, pcd2, pose1, pose2, thresh_val=0.05, H=1080, W=1920):
        print(f"iou tolerance: {thresh_val}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # pcd1_tl = pcd1.copy()
        # pcd2_tl = pcd2.copy()

        # pcd1_bl = pcd1.copy()
        # pcd2_bl = pcd2.copy()

        # pcd1_ml = pcd1.copy()
        # pcd2_ml = pcd2.copy()

        # pcd1_tl = np.reshape(pcd1_tl, (H, W, 3))
        # pcd1_sort_tl = np.flipud(np.sort(pcd1_tl, axis=0))
        # pcd1_ftl = pcd1_sort_tl[:5, :, :]
        # pcd1_ftl = np.reshape(pcd1_ftl, (-1, 3))

        # pcd2_tl = np.reshape(pcd2_tl, (H, W, 3))
        # pcd2_sort_tl = np.flipud(np.sort(pcd2_tl, axis=0))
        # pcd2_ftl = pcd2_sort_tl[:5, :, :]
        # pcd2_ftl = np.reshape(pcd2_ftl, (-1, 3))

        # pcd1_bl = np.reshape(pcd1_bl, (H, W, 3))
        # pcd1_sort_bl = np.sort(pcd1_bl, axis=0)
        # pcd1_fbl = pcd1_sort_bl[:5, :, :]
        # pcd1_fbl = np.reshape(pcd1_fbl, (-1, 3))

        # pcd2_bl = np.reshape(pcd2_bl, (H, W, 3))
        # pcd2_sort_bl = np.sort(pcd2_bl, axis=0)
        # pcd2_fbl = pcd2_sort_bl[:5, :, :]
        # pcd2_fbl = np.reshape(pcd2_fbl, (-1, 3))

        # pcd1_ml = np.reshape(pcd1_ml, (H, W, 3))
        # pcd2_ml = np.reshape(pcd2_ml, (H, W, 3))

        pcd1_final = pcd1.reshape(-1, 3)
        pcd2_final = pcd2.reshape(-1, 3)

        pcd1_gpu = torch.tensor(pcd1_final).to(device)
        pcd2_gpu = torch.tensor(pcd2_final).to(device)

        chunk_size = 1
        # Split the data into chunks
        num_chunks = (pcd1_gpu.size(0) + chunk_size - 1) // chunk_size
        
        # Initialize an empty tensor to store the results
        # euclidean_distance = torch.zeros(pcd1_gpu.size(0), pcd2_gpu.size(0)).cuda()

        # intersection_points = []
        intersection_indices_pcd1 = []
        intersection_indices_pcd2 = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, pcd1_gpu.size(0))
            chunk_pcd1 = pcd1_gpu[start_idx:end_idx]
        
            # Broadcasting still works as before, but now for smaller chunks
            chunk_distance = torch.sqrt(torch.sum((chunk_pcd1.unsqueeze(1) - pcd2_gpu) ** 2, dim=2))

            ###this piece of code is for intersection point viz
            for ri, r in enumerate(chunk_distance):
                pcd1_point = chunk_pcd1[ri]  # 获取对应的 pcd1_gpu 的点
                pcd1_index = start_idx + ri
                indices = torch.nonzero(r < thresh_val).squeeze(dim=1)
                if indices.numel() > 0:
                    for j in indices:
                        # intersection_points.append(pcd2_gpu[j, :])
                        intersection_indices_pcd1.append(pcd1_index)
                        intersection_indices_pcd2.append(j.item())
            ###Intersection point viz ends here

            # euclidean_distance[start_idx:end_idx, :] = chunk_distance

        # intersection_points = np.array([tensor.cpu().numpy() for tensor in intersection_points])
        # intersection_points = np.asarray(list(set(tuple(point) for point in intersection_points)))
        intersection_indices_pcd1 = np.array(intersection_indices_pcd1)

        intersection_indices_pcd1 = np.asarray(list(set(intersection_indices_pcd1)))
        intersection_indices_pcd2 = np.asarray(list(set(intersection_indices_pcd2)))

        intersection_count = math.sqrt(intersection_indices_pcd1.shape[0] * intersection_indices_pcd2.shape[0]) # a small modification
        
        union = pcd1_gpu.shape[0] + pcd2_gpu.shape[0] - intersection_count
        
        iou = intersection_count / np.abs(union)
        # print("intersection_indices_pcd1::", len(intersection_indices_pcd1))
        # print("intersection_indices_pcd2::", len(intersection_indices_pcd2))

        print(f"pose1: {pose1}, pose2: {pose2}, intersection: {intersection_count}, union: {union}, iou: {iou}")

        return intersection_count, union, iou, intersection_indices_pcd1, intersection_indices_pcd2

    def chamfer_dst(self, pcd1, pcd2):
        pcd1 = torch.tensor(np.asarray(pcd1.points), dtype=torch.float16)
        pcd2 = torch.tensor(np.asarray(pcd2.points), dtype=torch.float16)

        pcd1 = pcd1.unsqueeze_(0)
        pcd2 = pcd2.unsqueeze_(1)
        D = torch.min(torch.norm(pcd1 - pcd2, dim=-1), dim=0)
        return D

    # def get_random_navigable_pose(self, height=None, explored_pcd=None):
    #     if height == None:
    #         return self._sim.pathfinder.get_random_navigable_point()
    #     # Need to make sure point is navigable
    #     # at both agent height and sensor height

    #     # force choosing from given floor
    #     lower = height - 0.25
    #     heigher = height + 0.25

    #     # If explored_pcd, use floor pixels of explored_pcd to check
    #     # todo: change total_map stuff to explored_pcd

    #     num_start_tries = 0
    #     mapCheck = False
    #     res = self._sim.pathfinder.get_random_navigable_point()
    #     while res.position[1] < lower and res.position[1] > heigher and mapCheck and num_start_tries < 100:
    #         res = self._sim.pathfinder.get_random_navigable_point()
    #         if total_map != None:
    #             grid_res = self.get_grid_pos(res)
    #             x,y = grid_res
    #             mapCheck = total_map[x][y]
    #         else:
    #             mapCheck = True
                
    #         num_start_tries += 1
        
    #     assert(self._sim.pathfinder.is_navigable(res))
    #     return res

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
