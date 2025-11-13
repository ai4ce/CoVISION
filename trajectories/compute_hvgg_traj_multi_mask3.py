import json, os
import os.path as osp
import math
import numpy as np
import random
import pandas as pd
import cv2
from tqdm import tqdm

Train_list = ["Y4L8fjz2yH7.basis", "6HMiy15cxis.basis", "7UdY7HiDnUi.basis", "nHHVbEyHX3t.basis", 
 "3UDjdrwcqMb.basis", "AfKhsVmG8L4.basis", "oz1yTAGPXkh.basis", "iLDo95ZbDJq.basis", "H1D2FZ8TAv1.basis", "iQPq34e8hJX.basis", 
 "4J8N2Ah1a6o.basis", "SBHLgvFTVMZ.basis", "PXAfUkZGMdU.basis", "WEDXu8bWRkq.basis", "yA3RqPqMrGE.basis", "oEPjPNSPmzL.basis", 
 "bDTsgcSK5Qr.basis", "aJg466zMSNt.basis", "wcJYziD5pmF.basis", "dDyovSFuViJ.basis", "NkvRYHk72vA.basis", "FgswoxWb3uN.basis", 
 "3YSDRj9kTU7.basis", "TQSiMZJawkS.basis", "kZhZfAhdnNN.basis", "zJ3fVx3BZYR.basis","kA2nG18hCAr.basis", "mHXUEKEV6gR.basis", 
 "RrfVebebfWf.basis", "YV9M9gZG3YJ.basis", "c9DeZf2fcDf.basis", "Umx6CdjZfvy.basis", "qZ4B7U6XE5Y.basis", "UQ5EhY5wve1.basis", 
 "3PiKdwyfEkX.basis", "q33GehreMrX.basis", "cVppJowrUqs.basis", "nicaPonCxvC.basis", "7PZPFHR3oJc.basis", "wwX4MFiTTrt.basis", 
 "qmvPLqLAgvC.basis", "ZxkSUELrWtQ.basis", "D5dEbkUphhr.basis", "DwDDvGo9QdA.basis", "adgwjGh4NQK.basis", "Nf3aGQTDAA1.basis", 
 "NBWrHFXBF5p.basis", "AuGMayXVFkc.basis", "HsYeeztxPG6.basis", "NRsmXFcVTbN.basis", "uhkqDVMtEnn.basis", "LLecyBe5Eq2.basis", 
 "NEVASPhcrxR.basis", "j2eqyxdYAFW.basis", "uc8QkFS11Hj.basis", "PE6kVEtrxtj.basis", "tJ7XVoEN82a.basis", "UfhK7KNBg5u.basis", 
 "Vnb6uKtzQCU.basis", "MPPDV4Gvybr.basis", "5uXtMs57HmZ.basis", "GfF9TQ34x37.basis", "k3ohRuM6bso.basis", "PaQrTquNd2v.basis", 
 "hXHUtviUKBu.basis", "SrHVAbHUpUX.basis", "gyK27yu7CP4.basis", "GTV2Y73Sn5t.basis", "yX54kr5c5g9.basis", "T7nCRmufFNR.basis", 
 "GMwtBqNLGBs.basis", "5biL7VEkByM.basis", "dNASL765WSN.basis", "v7DzfFFEpsD.basis", "wxixLWuvLjd.basis", "zR6kPe1PsyS.basis", 
 "q28T9C3q2dv.basis", "3CBBjsNkhqW.basis", "9K1WbyTZ456.basis", "UAGeBzZJgkU.basis", "kCHmLFfMDuE.basis", "dcd823nTKH9.basis", 
 "9DnDAhJ7qcj.basis", "mWqBmEyXcXN.basis", "iNpfPhK1sRz.basis", "8B43pG641ff.basis", "SrBPiU6LKxL.basis", "WZDzPCybQvS.basis", 
 "gmuS7Wgsbrx.basis", "TiWanpmC63V.basis", "CxxHb5C8ZsP.basis", "XYyR54sxe6b.basis", "giViJCyCH2C.basis", "LViDMxZp4ZN.basis", 
 "jTTGECZYKRA.basis", "XNoaAZwsWKk.basis", "BqLwEyiLbza.basis", "JY8e73x9ubE.basis", "Wo6kuutE9i7.basis", "2XVvKEDd54w.basis", 
 "6ySDHVkso9e.basis", "37c5w29pYm3.basis", "Lva3QmSMsTr.basis", "D8aaq3PH6dG.basis", "cHumXFzhHUR.basis", "4vwGX7U38Ux.basis", 
 "JWWJBQWHv64.basis", "H81QMurNRM8.basis", "aYhkzj2fEhP.basis", "sLwz8nKD3wF.basis", "YM4nG4pSAEJ.basis", "nW7z5USWzWo.basis", 
 "Y6WjWkVEUks.basis", "o94q92w5PK5.basis", "741Fdj7NLF9.basis", "JiHGQpwKUvd.basis", "nJTPfwbAj4S.basis", "oXzJVhUhmYe.basis", 
 "D2PqRE5ZvyQ.basis", "xGnehmjiCSA.basis", "QKfBMSSy7Hy.basis", "LPEMkRVudUm.basis", "F1Vhvu3osn6.basis", "TZ2jsvNG2nt.basis", 
 "SgkmkWjjmDJ.basis", "WpVxtsP4xxA.basis", "PUNuHY5M7MS.basis", "LU4A39yR8gc.basis", "qDjhFcNqFPi.basis", "mggziYKSc6S.basis", 
 "U3oQjwTuMX8.basis", "mDdyQ6azhVD.basis", "tYvWp85L81G.basis", "C5RbHBQ76DE.basis", "qnKYFQsjnHf.basis", "1S7LAXRdDqK.basis", 
 "zCMdfYaW9iF.basis", "RYzud5W7ZnC.basis", "UbsJXeCkJBA.basis", "3KZbo846fxq.basis", "TSJmdttd2GV.basis", "b3WpMbPFB6q.basis", 
 "7CXbc73tDRf.basis", "RiwBKy2YdQ7.basis", "1UnKg1rAb8A.basis", "p32JzpQyhPk.basis", "JXdzHne1mRo.basis", "JFgrz9MNz4b.basis", 
 "8EqKbkhqE4R.basis", "oKFJo8jpzRW.basis", "P6ajptD9tRP.basis", "eAUmfFLZDR3.basis", "mHJxL9jnCox.basis", "u5atqC7vRCY.basis", 
 "fKP4sxcoxpL.basis", "NwG7cpZnRZb.basis", "pUneSGJDrvY.basis", "77mMEyxhs44.basis", "rWHyWNc6ZbZ.basis", "p6RF8AUer2e.basis", 
 "DGXRxHddGAW.basis", "qgZhhx1MpTi.basis", "nd6Vw5SHCoy.basis", "MLVm7dZk7dp.basis", "by8SK9u18S8.basis", "kJxT5qssH4H.basis", 
 "kyoZhaD9HuW.basis", "1EiJpeRNEs1.basis", "o1F5JVHc6mb.basis", "ij6Fizhrr6c.basis", "8oSQng53cGV.basis", "xcTV5UHYHFV.basis", 
 "5Kw4nGdqYtS.basis", "h5VYFcePkbn.basis", "bCFcvb4zc3N.basis", "QDvRVeWFCjM.basis", "b2e31HFFizw.basis", "1xGrZPxG1Hz.basis",
 "k7vRbGpz44m.basis", "C3ifY177Ldq.basis", "dioA6agn1cP.basis", "PyZonHqd5gy.basis"]

Test_list = ["BfzKZxFShtq.basis", "saBtfCeVoJ4.basis", "t3t9ofFLcFU.basis", "WypGcNbCdsH.basis", "fc7RfUCN5mY.basis",
 "8mXffaQTtmP.basis", "Bnq6SeZGL5b.basis", "QVAA6zecMHu.basis", "RfNGMBdVbAZ.basis", "TziyvKgzdAs.basis", 
 "41FNXLAZZgC.basis", "bdp1XNEdvmW.basis", "FRQ75PjD278.basis", "qpcpnP8TosR.basis", "YHmAkqgwe2p.basis", 
 "YJDUB7hWg9h.basis", "z9VLaZqCsW5.basis", "9SpHCfHaNiG.basis", "AMEM2eWycTq.basis", "aosjAwX5Lnq.basis", 
 "aRKASs4e8j1.basis", "DNWbUAJYsPy.basis", "EU6QPFpqdoU.basis", "RaYrxWt5pR1.basis", "SQqGpSHzfSr.basis", 
 "uzH9yHazm9t.basis", "YmWinf3mhb5.basis", "zmZvNTCxMZE.basis", "Coer9RdivP7.basis", "33ypawbKCQf.basis", 
 "4MRLu1yET6a.basis", "8uSpPmctPXC.basis", "C6JvMamYTRg.basis", "F5j7ZLfMm1n.basis", "RcuYAHzrjK7.basis", 
 "y4YiUQwvWGH.basis", "bxwHR9ipFG8.basis", "ceJTwFNjqCt.basis", "cjLuWviyDEo.basis", "rBmEe6ab5VP.basis", 
 "S3r45BMWy6H.basis", "tL6i2PtktSh.basis", "WRphMcFxfhe.basis", "x4LVLSsYWcV.basis", "YGc1h9nNrJP.basis", 
 "4dbCzNN5L5t.basis", "5Poh4Qz68hd.basis", "6TPCFES8fhh.basis", "C7xtw9uhYFn.basis", "F8PSGjTiv61.basis", 
 "sfbj7jspYWj.basis", "VaEwVD182FS.basis", "X6Pct1msZv5.basis", "X9fRPGxw1jS.basis", "Y8Y6ukxGMvn.basis", 
 "ZVScmfktNQ1.basis", "ASKXmHbw68X.basis", "W9YAR9qcuvN.basis", "zWydhyFhvcj.basis", "1mCzDx3EMom.basis", 
 "6EMViBCA2N7.basis", "8qbZhbTc1wX.basis", "99ML7CGPqsQ.basis", "aCtdWA5n56Z.basis", "BUFVGDCQNGb.basis", 
 "CETmJJqkhcK.basis", "qxwfVS8MQ67.basis", "tAQTHnJ7n72.basis", "tpxKD3awofe.basis", "UAByLdpaokx.basis", 
 "w5YEujJKsiy.basis", "XYQdAu1qsK9.basis", "yPKGKBCyYx8.basis", "z8SrvZ4eyqV.basis", "ZwnLFNzxASM.basis", 
 "6vJMULqvYe8.basis", "r38SGhq8aJr.basis", "VKmpsujnc5t.basis", "xccdSFAEPau.basis", "Xuky7E5df6A.basis"]

def extract_regions_from_stitched_image(image_path):
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

def extract_green_mask(image):
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

def compute_list(cam_dir, rel_mat, pose_file, positives, negatives, floor, threshold = 0.01, max_len = 30, resolution=(224,224)):
    rel_matrix = np.load(rel_mat)
    selected_pairs_path = osp.join(cam_dir, "GroundTruth.csv")
    covariant_dir = "/".join(cam_dir.split("/")[:-1])
    covariant_dir = osp.join(covariant_dir,"covariant_images")
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
    ref_covariants = []
    pos_covariants = []
    neg_covariants = []
    total_positive_len = 0
    os.makedirs("HM3D", exist_ok=True) 

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
        if len(pos_indices) >=10:
            pos_indices = pos_indices[:10]
        if len(neg_indices) >=10:
            neg_indices = neg_indices[:10]
        
        for j in range(len(pos_indices)):   
            pos_indices_copy = pos_indices.copy()
            neg_indices_copy = neg_indices.copy()  
            pos_indices_copy[0], pos_indices_copy[j] = pos_indices_copy[j], pos_indices_copy[0]
            random.shuffle(neg_indices_copy)
            # pos_indices_copy += random.choices(pos_indices, k=max_len - len(pos_indices))
            # neg_indices_copy += random.choices(neg_indices, k=2*max_len - len(neg_indices))

            ref_masks = np.zeros(resolution, dtype=np.uint8)
            #### positives #####
            rgb_paths_per = np.array(rgb_paths)[pos_indices_copy].tolist()
            poses_per = np.array(poses)[pos_indices_copy].tolist()
            pos_rgb_paths_all.append(rgb_paths_per)
            pos_indices_all.append(pos_indices_copy)
            pos_covariant_paths = []
            for pos_index in pos_indices_copy:
                scene_seed = rgb_paths_per[0].split("/")[6]
                scene = scene_seed.split("-")[0]
                covariant_path = osp.join(covariant_dir, scene+"_depth_"+str(pos_index)+"_color_"+str(i)+".png")
                assert os.path.exists(covariant_path), f"File does not exist: {covariant_path}"
                if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy"):
                    left_region, right_region = extract_regions_from_stitched_image(covariant_path)
                    assert(left_region is not None and right_region is not None)
                    ref_mask = extract_green_mask(left_region)
                    src_mask = extract_green_mask(right_region)
                    image_width, image_height = resolution
                    ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    ref_masks = np.logical_or(ref_masks, ref_mask)           
                    os.makedirs(f"HM3D/{scene_seed}/{floor}", exist_ok=True) 
                    bool_arr = src_mask == 255 
                    np.save(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy", bool_arr) 
                pos_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy")
            pos_depth_file_all.append(depth_file)
            pos_pose_all.append(poses_per)
            pos_covariants.append(pos_covariant_paths)
            #### negatives #####
            neg_covariant_paths = []
            for neg_index in neg_indices_copy:
                scene = rgb_paths_per[0].split("/")[6].split("-")[0]
                covariant_path = osp.join(covariant_dir, scene+"_depth_"+str(neg_index)+"_color_"+str(i)+".png")
                assert os.path.exists(covariant_path), f"File does not exist: {covariant_path}"
                if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy"):
                    left_region, right_region = extract_regions_from_stitched_image(covariant_path)
                    assert(left_region is not None and right_region is not None)
                    ref_mask = extract_green_mask(left_region)
                    src_mask = extract_green_mask(right_region)
                    image_width, image_height = resolution
                    ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    ref_masks = np.logical_or(ref_masks, ref_mask)
                    os.makedirs(f"HM3D/{scene_seed}/{floor}", exist_ok=True) 
                    bool_arr = src_mask == 255 
                    np.save(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy", bool_arr) 
                neg_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy")
            rgb_paths_per = np.array(rgb_paths)[neg_indices_copy].tolist()
            poses_per = np.array(poses)[neg_indices_copy].tolist()
            neg_rgb_paths_all.append(rgb_paths_per)
            neg_indices_all.append(neg_indices_copy)
            neg_depth_file_all.append(depth_file)
            neg_pose_all.append(poses_per)
            neg_covariants.append(neg_covariant_paths)  
            #### reference #####
            ref_covariant_paths = []
            ref_rgb_paths_all.append(rgb_paths[i])
            query_indice.append(i)
            # if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy"):
            #     np.save(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy", ref_masks) 
            ref_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy")
            ref_covariants.append(ref_covariant_paths)

        for j in range(len(neg_indices)):   
            pos_indices_copy = pos_indices.copy()
            neg_indices_copy = neg_indices.copy()  
            try:
                neg_indices_copy[0], neg_indices_copy[j] = neg_indices_copy[j], neg_indices_copy[0]
            except:
                random.shuffle(neg_indices_copy)

            random.shuffle(pos_indices_copy)
            # pos_indices_copy += random.choices(pos_indices, k=max_len - len(pos_indices))
            # neg_indices_copy += random.choices(neg_indices, k=2*max_len - len(neg_indices))
            
            #### positives #####
            pos_covariant_paths = []
            for pos_index in pos_indices_copy:
                scene = rgb_paths_per[0].split("/")[6].split("-")[0]
                covariant_path = osp.join(covariant_dir, scene+"_depth_"+str(pos_index)+"_color_"+str(i)+".png")
                assert os.path.exists(covariant_path), f"File does not exist: {covariant_path}"
                if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy"):
                    left_region, right_region = extract_regions_from_stitched_image(covariant_path)
                    assert(left_region is not None and right_region is not None)
                    ref_mask = extract_green_mask(left_region)
                    src_mask = extract_green_mask(right_region)
                    image_width, image_height = resolution
                    ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    ref_masks = np.logical_or(ref_masks, ref_mask)
                    os.makedirs(f"HM3D/{scene_seed}/{floor}", exist_ok=True) 
                    bool_arr = src_mask == 255 
                    np.save(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy", bool_arr) 
                pos_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy")

            rgb_paths_per = np.array(rgb_paths)[pos_indices_copy].tolist()
            poses_per = np.array(poses)[pos_indices_copy].tolist()
            pos_rgb_paths_all.append(rgb_paths_per)
            pos_indices_all.append(pos_indices_copy)
            pos_depth_file_all.append(depth_file)
            pos_pose_all.append(poses_per)
            pos_covariants.append(pos_covariant_paths)
            #### negatives #####
            neg_covariant_paths = []
            for neg_index in neg_indices_copy:
                scene = rgb_paths_per[0].split("/")[6].split("-")[0]
                covariant_path = osp.join(covariant_dir, scene+"_depth_"+str(neg_index)+"_color_"+str(i)+".png")
                assert os.path.exists(covariant_path), f"File does not exist: {covariant_path}"
                if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy"):
                    left_region, right_region = extract_regions_from_stitched_image(covariant_path)
                    assert(left_region is not None and right_region is not None)
                    ref_mask = extract_green_mask(left_region)
                    src_mask = extract_green_mask(right_region)
                    image_width, image_height = resolution
                    ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    ref_masks = np.logical_or(ref_masks, ref_mask)
                    bool_arr = src_mask == 255 
                    np.save(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy", bool_arr) 
                neg_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy")
            rgb_paths_per = np.array(rgb_paths)[neg_indices_copy].tolist()
            poses_per = np.array(poses)[neg_indices_copy].tolist()
            neg_rgb_paths_all.append(rgb_paths_per)
            neg_indices_all.append(neg_indices_copy)
            neg_depth_file_all.append(depth_file)
            neg_pose_all.append(poses_per)
            neg_covariants.append(neg_covariant_paths)  
            #### reference #####
            ref_rgb_paths_all.append(rgb_paths[i])
            query_indice.append(i)
            ref_covariant_paths = []
            ref_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy")
            ref_covariants.append(ref_covariant_paths)
        if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy"):
            np.save(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy", ref_masks) 
    assert(len(query_indice)==len(ref_covariants)==len(pos_indices_all))
    return query_indice, ref_rgb_paths_all, ref_covariants, pos_indices_all, pos_rgb_paths_all, pos_depth_file_all, pos_pose_all, pos_covariants, neg_indices_all, neg_rgb_paths_all, neg_depth_file_all, neg_pose_all, neg_covariants, total_positive_len

def create_dataset(dataset_size, scene_name, mode, data_root, Train_Test_list):
    dataset = []
    total_positive_len = 0
    for i in tqdm(range(dataset_size), desc="Processing"):
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

            query_indice, rgb_paths, ref_covariants, pos_indices, pos_rgb_paths, pos_depth_file, pos_poses, pos_covariants, neg_indices, neg_rgb_paths, neg_depth_file, neg_poses, neg_covariants, len_positives = compute_list(cam_dir, rel_mat, pose, positive_indices, negative_indices, floor)
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
                    'ref_covariants': ref_covariants[t],
                    'pos_indices': pos_indices[t],
                    'ref_rgb_paths': rgb_paths[t],
                    'pos_rgb_list': pos_rgb_paths[t],  # List of image paths
                    'pos_depth_file': pos_depth_file[t],  # List of depth image paths
                    'pos_poses': pos_poses[t],  # List of GT paths
                    'pos_covariants': pos_covariants[t],
                    'neg_indices': neg_indices[t],
                    'neg_rgb_list': neg_rgb_paths[t],  # List of image paths
                    'neg_depth_file': neg_depth_file[t],  # List of depth image paths
                    'neg_poses': neg_poses[t],  # List of GT paths
                    'neg_covariants': neg_covariants[t],
                    'intrinsic_raw': intrinsic_matrix.tolist()  # 4x4 list of lists
                }
                dataset.append(data_per)
    print("total_positive_len:::",total_positive_len)
    return dataset

def save_dataset(dataset, filename):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)

# Example usage
db_root = "../data/hvgg/parta"
data_root = osp.join(db_root,"temp","More_vis")
print('>> Listing all sequences')
sequences = [f for f in os.listdir(data_root)]
train_len = len(Train_list)
test_len = len(Test_list)
# assert(len(sequences)==train_len+test_len)
scene_name = "HM3D"
hfov=90 * (math.pi/180)
intrinsic_matrix = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0., 1, 0],
        [0., 0., 0, 1]])

dataset_train = create_dataset(train_len, scene_name, "train", data_root, Train_list)
print("len(dataset_train)::::", len(dataset_train))
save_dataset(dataset_train, "HM3D_train/dataset_train.json")

dataset_test = create_dataset(test_len, scene_name, "test", data_root, Test_list)
print("len(dataset_test)::::", len(dataset_test))
save_dataset(dataset_test, "HM3D_test/dataset_test.json")


