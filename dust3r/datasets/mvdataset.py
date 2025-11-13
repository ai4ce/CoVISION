import sys, os
import os.path as osp
import json
import itertools
from collections import deque
import imageio
from copy import deepcopy

import cv2
import numpy as np
import random
import h5py
import PIL

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

if 'META_INTERNAL' in os.environ.keys() and os.environ['META_INTERNAL'] == "False":
    from dust3r.dummy_io import *
else:
    from meta_internal.io import *
    
class MVDataset(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, from_tar = False, random_order = False, random_render_order = False, debug = False, *args, ROOT, n_test=1000, num_render_views = 0, n_vis_test = 8, n_vis_train = 8, split_thres = 0.9, render_start = None, tb_name = None, ref_all = False, n_ref = 1, random_nv_nr = None, dps_name = 'dps.h5', n_all = None, single_id = None, reverse = False, **kwargs):
        self.ROOT = ROOT
        self.num_render_views = num_render_views
        self.from_tar = from_tar
        self.random_order = random_order
        self.random_render_order = random_render_order
        super().__init__(*args, **kwargs) # self.num_views, split set inside
        if "test" in dps_name:
            self.test = True
        else:
            self.test = False
        self.num_inference_views = self.num_views - self.num_render_views
        self.num_vis_test = n_vis_test
        self.num_vis_train = n_vis_train
        if render_start is None:
            render_start = self.num_inference_views
        self.render_start = render_start
        self.tb_name = tb_name if tb_name is not None else self.split
        self.ref_all = ref_all
        self.n_ref = n_ref
        if random_nv_nr is None:
            random_nv_nr = [[self.num_views, self.num_render_views]]
        self.random_nv_nr = random_nv_nr
        self.dps_name = dps_name
        self.single_id = single_id

        # load all scenes
        self.data_name = osp.basename(self.ROOT)
        self.json_path = g_pathmgr.get_local_path(osp.join(self.ROOT, self.dps_name))

        
        with open(self.json_path, "r") as f:
            data = json.load(f)
        self.dps = [i for i in range(len(data))]
        # with h5py.File(self.h5f_path, 'r') as h5f:
        #     self.dps = [i for i in range(len(h5f['json_strs']))]
        
        if self.split != "all":
            split_ind = int(len(self.dps) * split_thres)
            test_id_list = [x for x in range(split_ind, len(self.dps), (len(self.dps) - split_ind) // n_test)]
            train_id_list = np.setdiff1d(np.arange(split_ind), test_id_list)
            train_test_id_list = [x + 1 for x in range(0, split_ind, split_ind // n_test)]
            vis_list = [x for x in range(split_ind, len(self.dps), (len(self.dps) - split_ind) // n_vis_test)][-n_vis_test:] + [train_test_id_list[x] for x in range(0, len(train_test_id_list), len(train_test_id_list) // n_vis_train)][-n_vis_train:]
            # vis_list = [x for x in range(split_ind, len(self.dps), (len(self.dps) - split_ind) // n_vis_test)][-n_vis_test:] + [train_test_id_list[x] for x in range(0, len(train_test_id_list), len(train_test_id_list) // n_vis_train)][-n_vis_train:]
            print('vis list', len(vis_list), vis_list)
        
        if self.split == "all":
            if n_all is not None:
                self.dps = [self.dps[int(id / n_all * len(self.dps))] for id in range(n_all)]
            pass
        elif self.split == "train":
            self.dps = [self.dps[x] for x in train_id_list]
        elif self.split == "train_test":
            self.dps = [self.dps[x] for x in train_test_id_list]
        elif self.split == "vis":
            self.dps = [self.dps[x] for x in vis_list]
        elif self.split == "test":
            self.dps = [self.dps[x] for x in test_id_list]
        if self.single_id is not None:
            self.dps = [self.dps[self.single_id]]
        if reverse:
            self.dps = list(reversed(self.dps))
        
    def __len__(self):
        
        print('len in dataset', len(self.dps), self.split)
        if self.ref_all:
            return len(self.dps) * self.num_inference_views
        return len(self.dps)

    def _downsample_to_target(self, image, depth, target_resolution=(224, 224)):
        # 0.3 - 0.4s
        assert image.shape[:2] == depth.shape, "shape error"

        image_pil = PIL.Image.fromarray(image)
        W, H = image_pil.size

        # Target resolution
        target_W, target_H = target_resolution

        # Calculate scale factors for width and height
        scale_x = target_W / W
        scale_y = target_H / H


        resized_image_pil = image_pil.resize(target_resolution, PIL.Image.LANCZOS)
        # ~0.3s
        resized_depth = cv2.resize(depth, target_resolution[::-1], interpolation=cv2.INTER_NEAREST)   
        
        return resized_image_pil, resized_depth

    def _downsample_to_target(self, image, target_resolution=(224, 224)):
        # 0.3 - 0.4s
        image_pil = PIL.Image.fromarray(image)
        W, H = image_pil.size

        # Target resolution
        target_W, target_H = target_resolution

        # Calculate scale factors for width and height
        scale_x = target_W / W
        scale_y = target_H / H


        resized_image_pil = image_pil.resize(target_resolution, PIL.Image.LANCZOS)
        # ~0.3s        
        return resized_image_pil

    def extract_regions_from_stitched_image(self,image_path):
        # Load the stitched image
        image = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return None, None

        # Coordinates of the left and right regions in the image (adjust according to your needs)
        # Left image region coordinates
        left_x_start, left_y_start = 81, 180
        left_x_end, left_y_end = 304, 304

        # Right image region coordinates
        right_x_start, right_y_start = 352, 180
        right_x_end, right_y_end = 574, 304

        # Extract regions
        left_image = image[left_y_start:left_y_end, left_x_start:left_x_end]
        right_image = image[right_y_start:right_y_end, right_x_start:right_x_end]

        return left_image, right_image

    def extract_green_mask(self, image):
        if image is None:
            return None

        # Convert to RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # Define the HSV range for green
        lower_green_hsv = np.array([35, 50, 50])
        upper_green_hsv = np.array([85, 255, 255])

        # Extract green mask
        green_mask = cv2.inRange(hsv_image, lower_green_hsv, upper_green_hsv)

        return green_mask
    
    def _get_views(self, idx, resolution, rng, from_tar = False, ref_view_id = None):
        random_nv_nr = random.choice(self.random_nv_nr)
        # self.num_views = random_nv_nr[0]
        # self.num_render_views = random_nv_nr[1]
        # self.num_inference_views = self.num_views - self.num_render_views
        if ref_view_id is None:
            ref_view_id = 0
        if self.ref_all:
            ref_view_id = idx % self.num_inference_views
            idx = idx // self.num_inference_views

        with open(self.json_path, 'r') as f:
            data = json.load(f)
            data_dict = data[self.dps[idx]]
            # C = data_dict['C'] if 'C' in data_dict.keys() else None
            # if C is None:
            #     C_avg = np.array(0.).astype(np.float32)
            # else:
            #     C = np.array(C)
            #     C = C[:self.num_inference_views,:self.num_inference_views]
            #     C_avg = C.mean()
            #     if C[0][0] > 0.9:
            #         C_avg -= 1 / self.num_inference_views
            #     C_avg = np.array(C_avg).astype(np.float32)
            ref_rgb = data_dict['ref_rgb_paths']
            ref_indice = data_dict['ref_indice']
            ref_covariants = data_dict['ref_covariants']
            
            pos_rgb_list = data_dict['pos_rgb_list']
            pos_indices = data_dict['pos_indices']
            pos_depth_file = data_dict['pos_depth_file']
            pos_pose_list = data_dict['pos_poses']
            pos_covariants = data_dict['pos_covariants']

            neg_rgb_list = data_dict['neg_rgb_list']
            neg_indices = data_dict['neg_indices']
            neg_depth_file = data_dict['neg_depth_file']
            neg_pose_list = data_dict['neg_poses']
            neg_covariants = data_dict['neg_covariants']

            # intrinsic_raw, intrinsic_list = None, []
            # if "intrinsic_raw" in data_dict.keys():
            #     intrinsic_raw = data_dict['intrinsic_raw']
            # else:
            #     intrinsic_list = data_dict['intrinsic_list']
        
        num_tuple = len(pos_rgb_list) + len(neg_rgb_list)
        
        # pos_render_set = [self.render_start + i for i in range(int(self.num_render_views/2))]
        # neg_render_set = [self.render_start + i for i in range(int(self.num_render_views/2))]
        
        # pos_inference_set = []
        # for i in range(int(self.num_views/2)):
        #     if i not in pos_render_set:
        #         pos_inference_set.append(i)
        #     if len(pos_inference_set) == int(self.num_inference_views/2):
        #         break
        # neg_inference_set = []
        # for i in range(int(self.num_views/2)):
        #     if i not in neg_render_set:
        #         neg_inference_set.append(i)
        #     if len(neg_inference_set) == int(self.num_inference_views/2):
        #         break
        indice_n = int((self.random_nv_nr[0][0] - 1)/2)
        pos_indices_set = random.choices(range(len(pos_indices)), k=indice_n)
        neg_indices_set = random.choices(range(len(neg_indices)), k=indice_n)
        assert(len(neg_indices) != 0)
        pos_depth_list = np.load(pos_depth_file[1:])[np.array(pos_indices)] # change later 
        neg_depth_list = np.load(neg_depth_file[1:])[np.array(neg_indices)] # change later 

        pos_rgb_list, pos_depth_list, pos_pose_list = change_to_sr([pos_rgb_list, pos_depth_list, pos_pose_list])
        neg_rgb_list, neg_depth_list, neg_pose_list = change_to_sr([neg_rgb_list, neg_depth_list, neg_pose_list])
        ref_view = []
        pos_views = []
        neg_views = []
        # ref_masks = np.zeros(resolution, dtype=np.uint8)
        ######## Positive #########
        for i in pos_indices_set:
            rgb = imageio.imread(g_pathmgr.get_local_path(pos_rgb_list[i])[1:]).astype(np.uint8) # / 256
            rgb = self._downsample_to_target(rgb, resolution)
            src_mask = np.load(os.path.join("trajectories",pos_covariants[i]))
            assert(src_mask.shape==(224,224))
            # assert(left_region is not None and right_region is not None)
            # ref_mask = self.extract_green_mask(left_region)
            # src_mask = self.extract_green_mask(right_region)
            # image_width, image_height = np.array(rgb).shape[:2]
            # ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            # src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            # ref_masks = np.logical_or(ref_masks, ref_mask)
            label=f"{str(idx).zfill(9)}"
                
            pos_views.append(dict(
                random_nv_nr=np.array(random_nv_nr),
                img=rgb,
                mask=src_mask,
                dataset=self.data_name,
                label=label,
                instance=str(idx),
                num_render_views = random_nv_nr[1],
                n_ref = self.n_ref,
                pair_label = True,
                # C_avg = C_avg,
            ))
        ######## Negative #########
        for i in neg_indices_set:
            rgb = imageio.imread(g_pathmgr.get_local_path(neg_rgb_list[i])[1:]).astype(np.uint8) # / 256
            # intrinsic_ = np.array(intrinsic_raw).astype(np.float32)
            # intrinsic = np.eye(4).astype(np.float32)
            # intrinsic[:3,:3] = intrinsic_[:3,:3]
            # camera_pose = neg_pose_list[i]
            rgb = self._downsample_to_target(rgb, resolution)
            src_mask = np.load(os.path.join("trajectories",neg_covariants[i]))
            assert(src_mask.shape==(224,224))
            # left_region, right_region = self.extract_regions_from_stitched_image(neg_covariants[i][1:])

            # assert(left_region is not None and right_region is not None)
            # ref_mask = self.extract_green_mask(left_region)
            # src_mask = self.extract_green_mask(right_region)
            # image_width, image_height = np.array(rgb).shape[:2]
            # ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            # src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            # ref_masks = np.logical_or(ref_masks, ref_mask)

            label=f"{str(idx).zfill(9)}"
                
            neg_views.append(dict(
                random_nv_nr=np.array(random_nv_nr),
                img=rgb,
                mask=src_mask,
                dataset=self.data_name,
                label=label,
                instance=str(idx),
                num_render_views = random_nv_nr[1],
                n_ref = self.n_ref,
                pair_label = False,
            ))

        ######## Reference ########
        rgb = imageio.imread(g_pathmgr.get_local_path(ref_rgb)[1:]).astype(np.uint8) # / 256
        rgb = self._downsample_to_target(rgb, resolution)
        label=f"{str(idx).zfill(9)}"
        ref_mask = np.load(os.path.join("trajectories",ref_covariants[0]))
        assert(ref_mask.shape==(224,224))
        ref_view.append(dict(
            random_nv_nr=np.array(random_nv_nr),
            img=rgb,
            mask=ref_mask,
            dataset=self.data_name,
            label=label,
            instance=str(idx),
            num_render_views = random_nv_nr[1],
            n_ref = self.n_ref,
            pair_label = True,
        ))
        # if self.random_order:
        #     random.shuffle(pos_views)  # Shuffle them
        # else:
        #     views[0], views[1] = deepcopy(views[1]), deepcopy(views[0])
        
        # if self.num_inference_views < 12:
        #     if len(views) > 3:
        #         views[1], views[3] = deepcopy(views[3]), deepcopy(views[1])
            
        #     if len(views) > 6:
        #         views[2], views[6] = deepcopy(views[6]), deepcopy(views[2])
        # else:
        # import pdb; pdb.set_trace()
        # assert(self.num_inference_views*3 >= 36)
        # change_id = self.num_inference_views // 4 + 1
        # views[1], views[change_id] = deepcopy(views[change_id]), deepcopy(views[1])
        # change_id = (self.num_inference_views * 2) // 4 + 1
        # views[2], views[change_id] = deepcopy(views[change_id]), deepcopy(views[2])
        # change_id = (self.num_inference_views * 3) // 4 + 1
        # views[3], views[change_id] = deepcopy(views[change_id]), deepcopy(views[3])

        # pos_views_inference, pos_views_render = [], []
        # for pos_view in pos_views:
        #     if pos_view['only_render']:
        #         pos_views_render.append(pos_view)
        #     else:
        #         pos_views_inference.append(pos_view)
        # pos_views = pos_views_inference + pos_views_render
        # assert len(pos_views) == int(self.num_views/2) - 1

        # neg_views_inference, neg_views_render = [], []
        # for neg_view in neg_views:
        #     if neg_view['only_render']:
        #         neg_views_render.append(neg_view)
        #     else:
        #         neg_views_inference.append(neg_view)
        # neg_views = neg_views_inference + neg_views_render
        # assert len(neg_views) == int(self.num_views/2)

        if self.random_order:
            pos_neg_views = pos_views + neg_views
            random.shuffle(pos_neg_views)  # Shuffle them
        else:
            pos_neg_views = pos_views + neg_views
        views = ref_view + pos_neg_views
        return views

if __name__ == "__main__":
    pass
