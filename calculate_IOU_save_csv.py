import json
import numpy as np
import pandas as pd
import glob
import os
from PIL import Image
import csv
from dust3r.model import AsymmetricCroCo3DStereoMultiView
from dust3r.utils.image import load_images
from dust3r.inference import inference, inference_mv
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa
from tqdm import tqdm

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

def calculate_iou_new(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape, "The matrices must have the same shape."
    
    TP = np.logical_and(matrix1, matrix2).sum()

    FP = np.logical_and(matrix1, np.logical_not(matrix2)).sum()
    FN = np.logical_and(np.logical_not(matrix1), matrix2).sum()

    return TP/(TP + FP + FN) if TP + FP + FN != 0 else 0

def get_result(model, scene, level, save_dir = "pairwise_results", device='cuda', batch_size = 1, size=224, silent=False, n_frame=10):
    scene_path = os.path.expandvars(f"data/gvgg/temp/More_vis/{scene}/{level}/saved_obs/")
    gt = pd.read_csv(scene_path + "GroundTruth.csv")
    num_imgs = len(list(glob.glob(scene_path + "*.png")))
    assert num_imgs > 0, f"{scene} at level {level} found 0 images"
    image_paths = [scene_path + f"best_color_{i}.png" for i in range(num_imgs)]
    images = load_images(image_paths, size=size, verbose=not silent, n_frame = n_frame)
    total_res = {}

    for i in range(len(images)-1):
        for j in range(i+1, len(images)):
            criterion = eval("BCELoss()").to(device)
            pred_ret = loss_of_one_batch([images[i], images[j]], model, None, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(0), ret=None, log = False)
            pred_prob1 = pred_ret['pred1']
            pred_prob1 = torch.sigmoid(pred_prob1) 
            pred_prob1 = (pred_prob1 >= 0.5).float()

            pred_prob2 = pred_ret['pred2s'][0]
            pred_prob2 = torch.sigmoid(pred_prob2) 
            pred_prob2 = (pred_prob2 >= 0.5).float()

            total_res[f"img {i} and img {j}"] = {
                "prob_1": bool(pred_prob1.item()),
                "prob_2": bool(pred_prob2.item()),

                f"path {i}": image_paths[i],    
                f"path {j}": image_paths[j],
                "ground truth": int(gt[(gt['image_1'] == gt_path(i, scene, level)) & (gt['image_2'] == gt_path(j, scene, level))]["label"])
            }
    
    with open(save_dir + f"/{scene}_{level}.json", "w") as file:
        json.dump(total_res, file, indent=4)


def save_two_images_as_one(image_path1, image_path2, output_path, arrangement="horizontal"):
    """
    Save two images to a single file.
    
    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
        output_path (str): Path to save the combined image.
        arrangement (str): "horizontal" or "vertical". How to combine the images.
    """
    # Open the images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Determine the new image size based on the arrangement
    if arrangement == "horizontal":
        new_width = img1.width + img2.width
        new_height = max(img1.height, img2.height)
    elif arrangement == "vertical":
        new_width = max(img1.width, img2.width)
        new_height = img1.height + img2.height
    else:
        raise ValueError("Arrangement must be 'horizontal' or 'vertical'.")

    # Create a new blank image
    combined_image = Image.new("RGB", (new_width, new_height))

    # Paste the two images onto the blank image
    combined_image.paste(img1, (0, 0))  # Paste img1 at the top-left corner
    if arrangement == "horizontal":
        combined_image.paste(img2, (img1.width, 0))  # Paste img2 to the right of img1
    else:  # Vertical
        combined_image.paste(img2, (0, img1.height))  # Paste img2 below img1

    # Save the combined image
    combined_image.save(output_path)
    print(f"Combined image saved to {output_path}")

def calculate_iou(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape, "The matrices must have the same shape."
    
    intersection = np.logical_and(matrix1, matrix2).sum()
    
    union = np.logical_or(matrix1, matrix2).sum()

    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou

def process_matrix_in_csv(gt_matrix, pred_overlap, scene_path):
    overlap_diff = []
    gt_diff = []
    diff_matrix = np.logical_xor(pred_overlap, gt_matrix).astype(float)
    coordinates = np.argwhere(diff_matrix == 1)
    for coordinate in coordinates:
        overlap_diff.append(pred_overlap[coordinate[0]][coordinate[1]])
        gt_diff.append(gt_matrix[coordinate[0]][coordinate[1]])

    return coordinates, overlap_diff, gt_diff

def gt_path(i, scene, level):
    return f"./temp/More_vis/{scene}/{level}/saved_obs/best_color_{i}.png"

def get_edge(overlap1, overlap2):
    true_shape = 288*512
    assert overlap1 <= true_shape and overlap2 <= true_shape, "overflow of overlap"
    return int(((overlap1/true_shape) * (overlap2/true_shape)) > 0.01), int((overlap1/true_shape) * (overlap2/true_shape) * 100)

def get_label(scene, level):
    scene_path = f"data/gvgg/temp/More_vis/{scene}/{level}/saved_obs/"

    gt = pd.read_csv(scene_path + "GroundTruth.csv")
    num_imgs = len(list(glob.glob(scene_path + "*.png")))
    pair_wise_matrix = np.zeros((num_imgs, num_imgs))
    pair_wise_overlap = np.zeros((num_imgs, num_imgs))

    gt_matrix = np.zeros((num_imgs, num_imgs))
    pair_wise_results = "pairwise_results"

    with open(os.path.join(pair_wise_results, f"{scene}_{level}.json"), "r", encoding="utf-8") as file:
        result = json.load(file)  

    for i in range(num_imgs - 1):
        for j in range(i+1, num_imgs):
            # print(f"img {i} and img {j}")
            # assert f"img {i} and img {j}" in list(result.keys()), f"key: img {i} and img {j}"
            overlap_result = int(result[f"img {i} and img {j}"]["prob_2"])
            gt_matrix[i][j] = int(result[f"img {i} and img {j}"]["ground truth"])
            # overlap1, overlap2 = overlap_result[f"base of image{i}, overlap"], overlap_result[f"base of image{j}, overlap"] 
            # edge, score = get_edge(overlap1, overlap2)
            # pair_wise_matrix[i][j] = edge
            pair_wise_overlap[i][j] = overlap_result
    
    coordinates, over_diff, gt_diff = process_matrix_in_csv(gt_matrix, pair_wise_overlap, scene_path)
    IOU = calculate_iou_new(pair_wise_overlap, gt_matrix)
    return coordinates, over_diff, gt_diff, IOU


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    inf = np.inf

    model_name = "outputs/MVD/checkpoint-last.pth"
    model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linearvpr', output_mode='feat', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True})
    model.to(device)
    model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(model_name).to(device)
    state_dict_loaded = model_loaded.state_dict()
    model.load_state_dict(state_dict_loaded, strict=True)
    # you can put the path to a local checkpoint in model_name if needed
    # load_images can take a list of images or a directory

    more_vis = os.path.expandvars("data/gvgg/temp/More_vis")
    scenes = list(glob.glob(more_vis+"/*"))
    scenes = [scene for scene in scenes if os.path.basename(scene) in Test_list]
    if not os.path.exists("pairwise_results/"):
        os.makedirs("pairwise_results/")
    scene_levels = {}
    for scene in scenes:
        scene_name = os.path.basename(scene)
        subfolder_paths = glob.glob(f"{scene}/*/")
        levels = [os.path.basename(os.path.dirname(folder)) for folder in subfolder_paths]
        levels = [foldername for foldername in levels if foldername != "visualization"]
    # #     scene_levels[scene_name] = levels
    # # scene_names = list(scene_levels.keys())
    # # scene_names = sorted(scene_names)
    # # for scene_name in scene_names:
    # #     print(scene_name, scene_levels[scene_name])
        for i in range(len(levels)):
            # print(os.path.join("pairwise_results", f"{scene}_{i}.json"))
            # print(os.path.exists(os.path.join("pairwise_results", f"{scene_name}_{i}.json")))
            if not os.path.exists(os.path.join("pairwise_results", f"{scene_name}_{i}.json")):
                get_result(model, scene_name, i)

    ################################################################
    results = {}
    more_vis = "data/gvgg/temp/More_vis/"
    scenes = list(glob.glob(more_vis+"/*"))
    scenes = [scene for scene in scenes if os.path.basename(scene) in Test_list]

    # if not os.path.exists("saved_pairs/"):
    #     os.makedirs("saved_pairs/")

    data = []
    data.append(['Folder', 'Subdir', 'First_image', 'Second_image', "predict", "GT"])

    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_name = os.path.basename(scene)
        subfolder_paths = glob.glob(f"{scene}/*/")
        levels = [os.path.basename(os.path.dirname(folder)) for folder in subfolder_paths]
        levels = [foldername for foldername in levels if foldername != "visualization"]

        for i in range(len(levels)):
            coordinates, over_diff, gt_diff, IOU = get_label(scene_name, i)
            results[scene_name+"-"+str(i)] = IOU
            for j in range(len(coordinates)):
                data_entry = [scene.split('/')[-1], i, coordinates[j][0], coordinates[j][1], int(over_diff[j]), int(gt_diff[j])]
                data.append(data_entry)

    with open('stat.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    iou_sum = 0
    for key in results.keys():
        iou_sum += results[key]

    iou_avg = iou_sum / len(results)

    results["iou avg overall"] = iou_avg

    with open("pair_wise_IOU_results.json", "w") as file:
        json.dump(results, file, indent=2)
    
            






