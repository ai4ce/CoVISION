import matplotlib.pyplot as plt
import glob
#import path
import os
import json
import pandas as pd
import numpy as np

masterFileGroundTruth = "./MasterGroundTruth.csv"

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

    adj = np.load(pathName+"rel_mat.npy")

    images1 = []
    images2 = []
    labels = []
    for i in range(adj.shape[0] - 1):
        for j in range(i+1, adj.shape[0]):
            images1.append(paths[i])
            images2.append(paths[j])
            
            if adj[i][j]!=0.0:
                labels.append(1)
            else:
                labels.append(0)
    gt = {'image_1': images1, 'image_2': images2, 'label':labels}
    df = pd.DataFrame(gt)
    df.to_csv(pathName + 'GroundTruth.csv', index = False)
    df.to_csv(masterFileGroundTruth, mode = 'a', index = False, header = False)