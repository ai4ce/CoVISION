import numpy as np
# import torch
import cv2
import matplotlib.pyplot as plt
# import glob
# import path
import networkx as nx
import scipy.io as sio
import os
# import json
# import imageio
# import sys
# import plotly.graph_objects as go
import networkx as nx
# from skimage.feature import plot_matches
import matplotlib.ticker as ticker
import collections
import math

sceneName = "./Stilwell/"
floor = 0
pathName = sceneName + str(floor) +"/saved_obs/"
top_down_path = sceneName + "visualization/topdown_" + str(floor) +".png"
top_down = cv2.imread(top_down_path)
file_prefix = "best_color"
images = []
paths = []
paths=[(pathName+file) for file in os.listdir(pathName) if file.startswith(file_prefix)]
paths = sorted(paths, key = lambda x: len(x))

for file in paths:
    img = cv2.imread(file)
    images.append(img)

indexes = list(range(len(paths)))
coords = np.load(pathName+"saved_grid_pose.npy")[:, :3]
coords = list(np.expand_dims(coords, -1))
adj = np.load(pathName+"rel_mat.npy")
gt_matches = collections.defaultdict(lambda: set())

for i in range(adj.shape[0]):
    for j in range(i, adj.shape[0]):
        if i == j:
            pass
            # gt_matches[i].add(j)
        
        elif adj[i][j]!=0:
            gt_matches[i].add(j)

numberOfAdjacencies = {}

sources = [(x[0][0], x[1][0]) for x in coords]
targets = [(x[0][0], x[1][0]) for x in coords]
pos = {ii : (x, y) for ii, (x, y) in enumerate(sources)}
positions = np.array([np.array(s) for s in sources], dtype = "float32")
inverse = {xy : ii for ii, xy in enumerate(sources)}
images_dic = {}
for i in range(len(images)):
    images_dic[i] = images[i]  #### can resize here
# create a more conventional edge list
edges = []
for i, s in enumerate(sources):
    for j, t in enumerate(targets):
        if j<i:
            continue
        if i in gt_matches.keys() and j in gt_matches[i]:
            edge = (i, j, gt_matches[i])
            edges.append(edge)

# create graph and plot
G = nx.Graph()
G.add_weighted_edges_from(edges)
nx.set_node_attributes(G, pos, "pos")
for node in G.adjacency():
    numberOfAdjacencies[node[0]] = len(node[1]) - 1
#     numberOfAdjacencies[node] = len(adjacency[1]) - 1
nx.set_node_attributes(G, numberOfAdjacencies, "Number of Scenes correspondences")

def update_annot(ind, index):
    node = index
    xy = pos[node]
#     annot.xy = xy
    annot.xy = (1.5,4.25)
    node_attr = {'node': node}
    node_attr.update(G.nodes[node])
    text = '\n'.join(f'{k}: {v}' for k, v in node_attr.items())
    annot.set_text(text)
nodes_selected = []
secondAx = False

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax[0]:
        cont, ind = nodes.contains(event)
        pos_clicked = np.array([event.xdata, event.ydata])
        
        index = np.argmin([math.dist(position, pos_clicked) for position in positions])
        
        global secondAx
        if secondAx == False:
            image = np.ones(images[index].shape, dtype = "uint32")
            image[:, :, 0] = images[index][:, :, 2]
            image[:, :, 1] = images[index][:, :, 1]
            image[:, :, 2] = images[index][:, :, 0]
            ax[1].imshow(image)
            secondAx = True
        else:
            image = np.ones(images[index].shape, dtype = "uint32")
            image[:, :, 0] = images[index][:, :, 2]
            image[:, :, 1] = images[index][:, :, 1]
            image[:, :, 2] = images[index][:, :, 0]
            ax[2].imshow(image)
            secondAx = False
        if cont:
            update_annot(ind, index)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()
fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [4.5, 3, 3]})
ax[0].imshow(top_down)
ax[1].xaxis.set_major_locator(ticker.NullLocator())
ax[1].yaxis.set_major_locator(ticker.NullLocator())
ax[2].xaxis.set_major_locator(ticker.NullLocator())
ax[2].yaxis.set_major_locator(ticker.NullLocator())

# nodes = nx.draw_networkx_nodes(G, pos=pos, ax=ax[0], node_size = 20, alpha = 0.75, cmap = 'Accent_r', node_color = list(numberOfAdjacencies.values()))
nodes = nx.draw_networkx_nodes(G, pos=pos, ax=ax[0], node_size = 40, alpha = 0.75, cmap = 'Blues', node_color = 'blue')
nx.draw_networkx_edges(G, pos=pos, ax=ax[0], alpha = 0.7, edge_color = 'red')

annot = ax[0].annotate("", xy=(0,0), xytext=(5, 5),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"))
annot.set_visible(False)


fig.canvas.mpl_connect("button_press_event", lambda event: hover(event))
plt.xticks([])
plt.yticks([])
plt.show()
plt.close()