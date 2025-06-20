import matplotlib.pyplot as plt
import glob
#import path
import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import *
import torch


for p in glob.glob("./temp/More_vis/*/*/saved_obs"):
    
    splitPath = p.split("/")
    
    base = splitPath[0] + "/" + splitPath[1] + "/" + splitPath[2]  
    sceneName = splitPath[-3]
    floor = splitPath[-2]
    
    pathName = base + "/" + sceneName + "/" + str(floor) + "/saved_obs/"

    file_prefix = "best_color"
    images = []
    paths = []
    
    paths=[(pathName+file) for file in os.listdir(pathName) if file.startswith(file_prefix)]
    paths = sorted(paths, key = lambda x: [len(x), x])

    adj = np.load(pathName+"rel_mat.npy", allow_pickle=True)
    poses = np.load(pathName+"saved_pose.npy", allow_pickle=True)
    deps = np.load(pathName+"saved_dep.npy", allow_pickle=True) 

    images1 = []
    images2 = []

    for i in range(0, adj.shape[0] - 1):
        for j in range(i+1, adj.shape[0]):

            print("Scene: {}, floor: {}, Node1: {}, Node2: {}".format(sceneName, floor, i, j))
            images1.append(paths[i])
            images2.append(paths[j])
            
            if adj[i][j]!=0.0:
                obs1 = {}
                obs1["color_sensor"] = plt.imread(paths[i])
                obs1["depth_sensor"] = deps[i]
                pose1 = poses[i].copy()

                obs2 = {}
                obs2["color_sensor"] = plt.imread(paths[j])
                obs2["depth_sensor"] = deps[j]
                pose2 = poses[j].copy()

                mask = get_mask(obs1, pose1, obs2, pose2)

                # visualize the mask
                # print(mask.shape)
                # plt.imshow(mask)
                # plt.savefig("./mask.png")
                # plt.close()
                
