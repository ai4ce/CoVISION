import glob
import numpy as np
import os

total_images = 0
total_floors = 0
scenes = set()
for p in glob.glob("./temp/More_vis/*/*/saved_obs"):
    
    splitPath = p.split("/")
    
    base = splitPath[0] + "/" + splitPath[1] + "/" + splitPath[2]  
    sceneName = splitPath[-3]
    floor = splitPath[-2]
    
    pathName = base + "/" + sceneName + "/" + str(floor) + "/saved_obs/"

    file_prefix = "best_color"
    images = []
    paths = []
    # total number of scenes
    scenes.add(sceneName)

    paths=[(pathName+file) for file in os.listdir(pathName) if file.startswith(file_prefix)]
    paths = sorted(paths, key = lambda x: [len(x), x])

    # total number of images
    adj = np.load(pathName+"rel_mat.npy")
    total_images += adj.shape[0]

    # total number of floors
    total_floors += 1



print("Total scenes: ", len(scenes))
print("Total images: ", total_images)
print("Total floors: ", total_floors)
