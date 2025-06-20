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
            
            pos_rgb_list = data_dict['pos_rgb_list']
            pos_indices = data_dict['pos_indices']
            pos_depth_file = data_dict['pos_depth_file']
            pos_pose_list = data_dict['pos_poses']

            neg_rgb_list = data_dict['neg_rgb_list']
            neg_indices = data_dict['neg_indices']
            neg_depth_file = data_dict['neg_depth_file']
            neg_pose_list = data_dict['neg_poses']

            intrinsic_raw, intrinsic_list = None, []
            if "intrinsic_raw" in data_dict.keys():
                intrinsic_raw = data_dict['intrinsic_raw']
            else:
                intrinsic_list = data_dict['intrinsic_list']
        
        num_tuple = len(pos_rgb_list) + len(neg_rgb_list)
        
        pos_render_set = [self.render_start + i for i in range(int(self.num_render_views/2))]
        neg_render_set = [self.render_start + i for i in range(int(self.num_render_views/2))]
        
        pos_inference_set = []
        for i in range(int(self.num_views/2)):
            if i not in pos_render_set:
                pos_inference_set.append(i)
            if len(pos_inference_set) == int(self.num_inference_views/2):
                break
        neg_inference_set = []
        for i in range(int(self.num_views/2)):
            if i not in neg_render_set:
                neg_inference_set.append(i)
            if len(neg_inference_set) == int(self.num_inference_views/2):
                break

        assert(len(neg_indices) != 0)
        pos_depth_list = np.load(pos_depth_file[1:])[np.array(pos_indices)] # change later 
        neg_depth_list = np.load(neg_depth_file[1:])[np.array(neg_indices)] # change later 

        pos_rgb_list, pos_depth_list, pos_pose_list, intrinsic_raw = change_to_sr([pos_rgb_list, pos_depth_list, pos_pose_list, intrinsic_raw])
        neg_rgb_list, neg_depth_list, neg_pose_list, _ = change_to_sr([neg_rgb_list, neg_depth_list, neg_pose_list, intrinsic_raw])
        pos_views = []
        neg_views = []
        for i in pos_inference_set + pos_render_set:
            rgb = imageio.imread(g_pathmgr.get_local_path(pos_rgb_list[i])[1:]).astype(np.uint8) # / 256
            depth = pos_depth_list[i]
            intrinsic_ = np.array(intrinsic_raw).astype(np.float32)
            intrinsic = np.eye(4).astype(np.float32)
            intrinsic[:3,:3] = intrinsic_[:3,:3]

            camera_pose = pos_pose_list[i]
            rgb, depth = self._downsample_to_target(rgb, depth, resolution)
            label=f"{str(idx).zfill(9)}"
                
            pos_views.append(dict(
                random_nv_nr=np.array(random_nv_nr),
                img=rgb,
                depthmap=depth,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsic,
                dataset=self.data_name,
                label=label,
                instance=str(idx),
                only_render = i in pos_render_set,
                num_render_views = random_nv_nr[1],
                n_ref = self.n_ref,
                pair_label = True,
                # C_avg = C_avg,
            ))

        for i in neg_inference_set + neg_render_set:
            rgb = imageio.imread(g_pathmgr.get_local_path(neg_rgb_list[i])[1:]).astype(np.uint8) # / 256
            depth = neg_depth_list[i]
            intrinsic_ = np.array(intrinsic_raw).astype(np.float32)
            intrinsic = np.eye(4).astype(np.float32)
            intrinsic[:3,:3] = intrinsic_[:3,:3]

            camera_pose = neg_pose_list[i]
            rgb, depth = self._downsample_to_target(rgb, depth, resolution)

            label=f"{str(idx).zfill(9)}"
                
            neg_views.append(dict(
                random_nv_nr=np.array(random_nv_nr),
                img=rgb,
                depthmap=depth,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsic,
                dataset=self.data_name,
                label=label,
                instance=str(idx),
                only_render = i in neg_render_set,
                num_render_views = random_nv_nr[1],
                n_ref = self.n_ref,
                pair_label = False,
                # C_avg = C_avg,
            ))
        
        if ref_view_id != 0:
            pos_views[0], pos_views[ref_view_id] = deepcopy(pos_views[ref_view_id]), deepcopy(pos_views[0])

        ref_view = [pos_views[0]]
        pos_views = pos_views[1:]  # Extract last 9 elements
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

        pos_views_inference, pos_views_render = [], []
        for pos_view in pos_views:
            if pos_view['only_render']:
                pos_views_render.append(pos_view)
            else:
                pos_views_inference.append(pos_view)
        pos_views = pos_views_inference + pos_views_render
        assert len(pos_views) == int(self.num_views/2) - 1

        neg_views_inference, neg_views_render = [], []
        for neg_view in neg_views:
            if neg_view['only_render']:
                neg_views_render.append(neg_view)
            else:
                neg_views_inference.append(neg_view)
        neg_views = neg_views_inference + neg_views_render
        assert len(neg_views) == int(self.num_views/2)

        if self.random_order:
            pos_neg_views = pos_views + neg_views
            random.shuffle(pos_neg_views)  # Shuffle them
        else:
            pos_neg_views = pos_views + neg_views
        views = ref_view + pos_neg_views
        return views

if __name__ == "__main__":
    pass
