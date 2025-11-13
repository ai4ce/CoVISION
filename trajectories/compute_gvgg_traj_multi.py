import json, os
import os.path as osp
import math
import numpy as np
import random
import pandas as pd

Train_list = ['Adrian-1', 'Adrian-2', 'Adrian-3', 'Adrian-4', 'Adrian-5', 
'Albertville-1', 'Albertville-2', 'Albertville-3', 'Albertville-4', 'Albertville-5', 
'Andover-1', 'Andover-2', 'Andover-3', 'Andover-4', 'Andover-5', 'Angiola-1', 'Angiola-2', 
'Angiola-3', 'Angiola-4', 'Angiola-5', 'Annawan-1', 'Annawan-2', 'Annawan-3', 'Annawan-4', 
'Annawan-5', 'Applewold-1', 'Applewold-2', 'Applewold-3', 'Applewold-4', 'Applewold-5', 
'Ballou-1', 'Ballou-2', 'Ballou-3', 'Ballou-4', 'Ballou-5', 'Beach-1', 'Beach-2', 'Beach-3', 
'Beach-4', 'Beach-5', 'Brevort-1', 'Brevort-2', 'Brevort-3', 'Brevort-4', 'Brevort-5', 'Cantwell-1', 
'Cantwell-2', 'Cantwell-3', 'Cantwell-4', 'Cantwell-5', 'Capistrano-1', 'Capistrano-2', 'Capistrano-3',
'Capistrano-4', 'Capistrano-5', 'Colebrook-1', 'Colebrook-2', 'Colebrook-3', 'Colebrook-4', 'Colebrook-5', 
'Crandon-1', 'Crandon-2', 'Crandon-3', 'Crandon-4', 'Crandon-5', 'Denmark-1', 'Denmark-2', 'Denmark-3', 
'Denmark-4', 'Denmark-5', 'Dryville-1', 'Dryville-2', 'Dryville-3', 'Dryville-4', 'Dryville-5', 'Dunmor-1',
'Dunmor-2', 'Dunmor-3', 'Dunmor-4', 'Dunmor-5', 'Eastville-1', 'Eastville-2', 'Eastville-3', 'Eastville-4',
'Eastville-5', 'Edgemere-1', 'Edgemere-2', 'Edgemere-3', 'Edgemere-4', 'Edgemere-5', 'Elmira-1', 'Elmira-2',
'Elmira-3', 'Elmira-4', 'Elmira-5', 'Greigsville-1', 'Greigsville-2', 'Greigsville-3', 'Greigsville-4', 
'Greigsville-5', 'Hainesburg-1', 'Hainesburg-2', 'Hainesburg-3', 'Hainesburg-4', 'Hainesburg-5', 
'Hambleton-1', 'Hambleton-2', 'Hambleton-3', 'Hambleton-4', 'Hambleton-5', 'Haxtun-1', 'Haxtun-2', 
'Haxtun-3', 'Haxtun-4', 'Haxtun-5', 'Hillsdale-1', 'Hillsdale-2', 'Hillsdale-3', 'Hillsdale-4', 
'Hillsdale-5', 'Hometown-1', 'Hometown-2', 'Hometown-3', 'Hometown-4', 'Hometown-5', 'Hominy-1', 
'Hominy-2', 'Hominy-3', 'Hominy-4', 'Hominy-5', 'Kerrtown-1', 'Kerrtown-2', 'Kerrtown-3', 'Kerrtown-4', 
'Kerrtown-5', 'Maryhill-1', 'Maryhill-2', 'Maryhill-3', 'Maryhill-4', 'Maryhill-5', 'Micanopy-1',
'Micanopy-2', 'Micanopy-3', 'Micanopy-4', 'Micanopy-5', 'Mobridge-1', 'Mobridge-2', 'Mobridge-3', 
'Mobridge-4', 'Mobridge-5', 'Monson-1', 'Monson-2', 'Monson-3', 'Monson-4', 'Monson-5', 'Mosquito-1', 
'Mosquito-2', 'Mosquito-3', 'Mosquito-4', 'Mosquito-5', 'Nemacolin-1', 'Nemacolin-2', 'Nemacolin-3', 
'Nemacolin-4', 'Nemacolin-5', 'Nicut-1', 'Nicut-2', 'Nicut-3', 'Nicut-4', 'Nicut-5', 'Nimmons-1', 
'Nimmons-2', 'Nimmons-3', 'Nimmons-4', 'Nimmons-5', 'Nuevo-1', 'Nuevo-2', 'Nuevo-3', 'Nuevo-4', 'Nuevo-5',
'Oyens-1', 'Oyens-2', 'Oyens-3', 'Oyens-4', 'Oyens-5', 'Pablo-1', 'Pablo-2', 'Pablo-3', 'Pablo-4', 
'Pablo-5', 'Parole-1', 'Parole-2', 'Parole-3', 'Parole-4', 'Parole-5', 'Pettigrew-1', 'Pettigrew-2', 
'Pettigrew-3', 'Pettigrew-4', 'Pettigrew-5', 'Placida-1', 'Placida-2', 'Placida-3', 'Placida-4',
'Placida-5', 'Pleasant-1', 'Pleasant-2', 'Pleasant-3', 'Pleasant-4', 'Pleasant-5', 'Quantico-1', 
'Quantico-2','Quantico-3', 'Quantico-4', 'Quantico-5', 'Reyno-1', 'Reyno-2', 'Reyno-3', 'Reyno-4', 'Reyno-5', 
'Roane-1', 'Roane-2', 'Roane-3', 'Roane-4', 'Roane-5', 'Roeville-1', 'Roeville-2', 'Roeville-3', 
'Roeville-4', 'Roeville-5', 'Rosser-1', 'Rosser-2', 'Rosser-3', 'Rosser-4', 'Rosser-5', 'Sands-1', 
'Sands-2', 'Sands-3', 'Sands-4', 'Sands-5', 'Sasakwa-1', 'Sasakwa-2', 'Sasakwa-3', 'Sasakwa-4', 
'Sasakwa-5', 'Sawpit-1', 'Sawpit-2', 'Sawpit-3', 'Sawpit-4', 'Sawpit-5', 'Scioto-1', 'Scioto-2', 
'Scioto-3', 'Scioto-4', 'Scioto-5', 'Seward-1', 'Seward-2', 'Seward-3', 'Seward-4', 'Seward-5', 
'Shelbiana-1', 'Shelbiana-2', 'Shelbiana-3', 'Shelbiana-4', 'Shelbiana-5', 'Sisters-1', 'Sisters-2',
'Sisters-3', 'Sisters-4', 'Sisters-5', 'Sodaville-1', 'Sodaville-2', 'Sodaville-3', 'Sodaville-4',
'Sodaville-5', 'Soldier-1', 'Soldier-2', 'Soldier-3', 'Soldier-4', 'Soldier-5', 'Spencerville-1', 
'Spencerville-2', 'Spencerville-3', 'Spencerville-4', 'Spencerville-5', 'Spotswood-1', 'Spotswood-2',
'Spotswood-3', 'Spotswood-4', 'Spotswood-5', 'Springhill-1', 'Springhill-2', 'Springhill-3', 
'Springhill-4', 'Springhill-5', 'Stanleyville-1', 'Stanleyville-2', 'Stanleyville-3', 'Stanleyville-4',
'Stanleyville-5', 'Stilwell-1', 'Stilwell-2', 'Stilwell-3', 'Stilwell-4', 'Stilwell-5', 'Stokes-1',
'Stokes-2', 'Stokes-3', 'Stokes-4', 'Stokes-5', 'Superior-1', 'Superior-2', 'Superior-3', 'Superior-4',
'Superior-5', 'Swormville-1', 'Swormville-2', 'Swormville-3', 'Swormville-4', 'Swormville-5', 'Woonsocket-1',
'Woonsocket-2', 'Woonsocket-3', 'Woonsocket-4', 'Woonsocket-5']

Test_list = ['Sumas-1', 'Sumas-2', 'Sumas-3', 'Sumas-4', 'Sumas-5', 'Roxboro-1', 'Roxboro-2', 'Roxboro-3',
'Roxboro-4', 'Roxboro-5', 'Eudora-1', 'Eudora-2', 'Eudora-3', 'Eudora-4', 'Eudora-5', 'Arkansaw-1', 
'Arkansaw-2', 'Arkansaw-3', 'Arkansaw-4', 'Arkansaw-5', 'Convoy-1', 'Convoy-2', 'Convoy-3', 'Convoy-4',
'Convoy-5', 'Ribera-1', 'Ribera-2', 'Ribera-3', 'Ribera-4', 'Ribera-5', 'Sanctuary-1', 'Sanctuary-2',
'Sanctuary-3', 'Sanctuary-4', 'Sanctuary-5', 'Silas-1', 'Silas-2', 'Silas-3', 'Silas-4', 'Silas-5', 
'Bowlus-1', 'Bowlus-2', 'Bowlus-3', 'Bowlus-4', 'Bowlus-5', 'Cooperstown-1', 'Cooperstown-2', 
'Cooperstown-3', 'Cooperstown-4', 'Cooperstown-5', 'Delton-1', 'Delton-2', 'Delton-3', 'Delton-4',
'Delton-5', 'Rancocas-1', 'Rancocas-2', 'Rancocas-3', 'Rancocas-4', 'Rancocas-5', 'Mesic-1', 
'Mesic-2', 'Mesic-3', 'Mesic-4', 'Mesic-5', 'Eagerville-1', 'Eagerville-2', 'Eagerville-3', 
'Eagerville-4', 'Eagerville-5', 'Goffs-1', 'Goffs-2', 'Goffs-3', 'Goffs-4', 'Goffs-5', 'Bolton-1',
'Bolton-2', 'Bolton-3', 'Bolton-4', 'Bolton-5', 'Mosinee-1', 'Mosinee-2', 'Mosinee-3', 'Mosinee-4',
'Mosinee-5', 'Avonia-1', 'Avonia-2', 'Avonia-3', 'Avonia-4', 'Avonia-5', 'Anaheim-1', 'Anaheim-2',
'Anaheim-3', 'Anaheim-4', 'Anaheim-5', 'Azusa-1', 'Azusa-2', 'Azusa-3', 'Azusa-4', 'Azusa-5']

def compute_list(cam_dir, rel_mat, pose_file, positives, negatives, threshold = 0.01, max_len = 30):
    rel_matrix = np.load(rel_mat)
    selected_pairs_path = osp.join(cam_dir, "GroundTruth.csv")
    selected_pairs_df = pd.read_csv(selected_pairs_path)

    poses = np.load(pose_file)
    depth_file = osp.join(cam_dir, "saved_dep.npy")
    rgb_paths = []
    ref_rgb_paths_all = []
    query_indice = []
    pos_rgb_paths_all = []
    neg_rgb_paths_all = []
    pos_pose_all = []
    neg_pose_all = []
    pos_depth_file_all = []
    neg_depth_file_all = []
    pos_indices_all = []
    neg_indices_all = []
    total_positive_len = 0

    for i in range(rel_matrix.shape[0]):
        rgb_paths.append(osp.join(cam_dir,"best_color_"+str(i)+'.png'))

    for i in range(rel_matrix.shape[0]):
        pos_indices = []
        neg_indices = []
        
        for j in range(len(selected_pairs_df)):
            image1_path = selected_pairs_df.iloc[j]["image_1"]
            image2_path = selected_pairs_df.iloc[j]["image_2"]
            idx1 = int(image1_path.split("/")[-1].split(".")[0].split("_")[-1])
            idx2 = int(image2_path.split("/")[-1].split(".")[0].split("_")[-1])
            label = int(selected_pairs_df.iloc[j]["label"])
            if label == 1 and idx1==i:
                pos_indices.append(idx2)
            elif label == 0 and idx1==i:
                neg_indices.append(idx2)
        total_positive_len += len(pos_indices)
        assert(len(pos_indices) < max_len)
        assert(len(neg_indices) < 2*max_len)
        if len(pos_indices) == 0:
            continue
        if len(neg_indices) == 0:
            continue
        
        for j in range(len(pos_indices)):   
            pos_indices_copy = pos_indices.copy()
            neg_indices_copy = neg_indices.copy()  
            pos_indices_copy[0], pos_indices_copy[j] = pos_indices_copy[j], pos_indices_copy[0]
            random.shuffle(neg_indices_copy)
            pos_indices_copy += random.choices(pos_indices, k=max_len - len(pos_indices))
            neg_indices_copy += random.choices(neg_indices, k=2*max_len - len(neg_indices))

            #### reference #####
            ref_rgb_paths_all.append(rgb_paths[i])
            query_indice.append(i)
            #### positives #####
            rgb_paths_per = np.array(rgb_paths)[pos_indices_copy].tolist()
            poses_per = np.array(poses)[pos_indices_copy].tolist()
            pos_rgb_paths_all.append(rgb_paths_per)
            pos_indices_all.append(pos_indices_copy)
            pos_depth_file_all.append(depth_file)
            pos_pose_all.append(poses_per)
            #### negatives #####
            rgb_paths_per = np.array(rgb_paths)[neg_indices_copy].tolist()
            poses_per = np.array(poses)[neg_indices_copy].tolist()
            neg_rgb_paths_all.append(rgb_paths_per)
            neg_indices_all.append(neg_indices_copy)
            neg_depth_file_all.append(depth_file)
            neg_pose_all.append(poses_per)
            

        for j in range(len(pos_indices)):   
            pos_indices_copy = pos_indices.copy()
            neg_indices_copy = neg_indices.copy()  
            try:
                neg_indices_copy[0], neg_indices_copy[j] = neg_indices_copy[j], neg_indices_copy[0]
            except:
                random.shuffle(neg_indices_copy)

            random.shuffle(pos_indices_copy)
            pos_indices_copy += random.choices(pos_indices, k=max_len - len(pos_indices))
            neg_indices_copy += random.choices(neg_indices, k=2*max_len - len(neg_indices))
            #### reference #####
            ref_rgb_paths_all.append(rgb_paths[i])
            query_indice.append(i)
            #### positives #####
            rgb_paths_per = np.array(rgb_paths)[pos_indices_copy].tolist()
            poses_per = np.array(poses)[pos_indices_copy].tolist()
            pos_rgb_paths_all.append(rgb_paths_per)
            pos_indices_all.append(pos_indices_copy)
            pos_depth_file_all.append(depth_file)
            pos_pose_all.append(poses_per)
            #### negatives #####
            rgb_paths_per = np.array(rgb_paths)[neg_indices_copy].tolist()
            poses_per = np.array(poses)[neg_indices_copy].tolist()
            neg_rgb_paths_all.append(rgb_paths_per)
            neg_indices_all.append(neg_indices_copy)
            neg_depth_file_all.append(depth_file)
            neg_pose_all.append(poses_per)
    return query_indice, ref_rgb_paths_all, pos_indices_all, pos_rgb_paths_all, pos_depth_file_all, pos_pose_all, neg_indices_all, neg_rgb_paths_all, neg_depth_file_all, neg_pose_all, total_positive_len

def create_dataset(dataset_size, scene_name, mode, data_root, Train_Test_list):
    dataset = []
    total_positive_len = 0
    for i in range(dataset_size):
        scene = osp.join(data_root, Train_Test_list[i])
        contents = os.listdir(scene)
        floor_folders = [name for name in contents if name.isdigit() and os.path.isdir(os.path.join(scene, name))]
        for floor in floor_folders:
            cam_dir = osp.join(scene, floor, "saved_obs")
            pose = osp.join(cam_dir, "saved_pose.npy")
            rel_mat = osp.join(cam_dir, "rel_mat.npy")

            selected_pairs_path = osp.join(cam_dir, "GroundTruth.csv")
            selected_pairs_df = pd.read_csv(selected_pairs_path)
            positive_indices = []
            negative_indices = []

            query_indice, rgb_paths, pos_indices, pos_rgb_paths, pos_depth_file, pos_poses, neg_indices, neg_rgb_paths, neg_depth_file, neg_poses, len_positives = compute_list(cam_dir, rel_mat, pose, positive_indices, negative_indices)
            total_positive_len += len_positives
            # import pdb; pdb.set_trace()
            # for f in os.listdir(cam_dir):
            #     if f.startswith('best_color_'):
            #         filename = f[:-4].split("_")[-1]
            #         depth_file = "best_depth_"+filename
            #         compute_list(root, floor, f[:-4], depth_file, rel_mat, pose, out_dir)
            for t, indice in enumerate(pos_indices):
                if len(neg_indices[t])==0 or len(pos_indices[t])==0:
                    continue
                data_per = {
                    'scene_name': scene_name,
                    'ref_indice': query_indice[t],
                    'pos_indices': pos_indices[t],
                    'ref_rgb_paths': rgb_paths[t],
                    'pos_rgb_list': pos_rgb_paths[t],  # List of image paths
                    'pos_depth_file': pos_depth_file[t],  # List of depth image paths
                    'pos_poses': pos_poses[t],  # List of GT paths
                    'neg_indices': neg_indices[t],
                    'neg_rgb_list': neg_rgb_paths[t],  # List of image paths
                    'neg_depth_file': neg_depth_file[t],  # List of depth image paths
                    'neg_poses': neg_poses[t],  # List of GT paths
                    'intrinsic_raw': intrinsic_matrix.tolist()  # 4x4 list of lists
                }
                dataset.append(data_per)
    print("total_positive_len:::",total_positive_len)
    return dataset

def save_dataset(dataset, filename):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)

# Example usage
db_root = "../data/gvgg"
data_root = osp.join(db_root,"temp","More_vis")
print('>> Listing all sequences')
sequences = [f for f in os.listdir(data_root)]
train_len = len(Train_list)
test_len = len(Test_list)
assert(len(sequences)==train_len+test_len)
scene_name = "CoVISION"
hfov=90 * (math.pi/180)
intrinsic_matrix = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0., 1, 0],
        [0., 0., 0, 1]])

dataset_train = create_dataset(train_len, scene_name, "train", data_root, Train_list)
print("len(dataset_train)::::", len(dataset_train))
save_dataset(dataset_train, "covision_train/dataset_train.json")

dataset_test = create_dataset(test_len, scene_name, "test", data_root, Test_list)
print("len(dataset_test)::::", len(dataset_test))
save_dataset(dataset_test, "covision_test/dataset_test.json")


