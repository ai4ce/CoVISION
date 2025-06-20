import matplotlib.pyplot as plt
import glob
#import path
import os
import json
import pandas as pd
import numpy as np

for p in glob.glob("./temp/More_vis/Adrian/0/saved_obs"):
    idx = 0
    splitPath = p.split("/")
    
    base = splitPath[0] + "/" + splitPath[1] + "/" + splitPath[2]  
    sceneName = splitPath[-3]
    floor = splitPath[-2]
    
    pathName = base + "/" + sceneName + "/" + str(floor) + "/saved_obs/"

    file_prefix = "best_color"
    images = []
    paths = []
    
    paths=[(pathName+file) for file in os.listdir(pathName) if file.startswith(file_prefix)]
    paths = sorted(paths, key = lambda x: len(x))

    adj = np.load(pathName+"rel_mat.npy")
    print("idx-{} {}: ".format(idx, adj[idx]))
