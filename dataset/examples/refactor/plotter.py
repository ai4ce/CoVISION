import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from time import time
import os
from floor import Candidate


def arrow(x0, y0, length, angle, color):
    x1 = np.cos(np.pi / 2 + angle) * length
    y1 = np.sin(np.pi / 2 + angle) * length

    return Arrow(x0, y0, x1, -y1, width=100.0, color=color)


def plot_pivot_candidates(original_map: np.ndarray, pivot_array: np.ndarray, base_dir: str, pivot_number:int) -> None:
      """
      Plots the pivot candidates ([[x,y, theta],...]) from the original_map.
      """
      fig, ax = plt.subplots()
      ax.imshow(original_map, cmap="gray")
      colors_ = {
      0: "red",
      1: "blue",
      2: "green",
      3: "yellow",
      4: "m",
      5: "k",
      6: "purple",
      7: "c",
      }
      # @TODO: Need optimization here
      for pivot_ in pivot_array:
            pivot_color = colors_.get(pivot_[2], "black")
            patch = Circle((int(pivot_[1]), int(pivot_[0])), radius=1, color=pivot_color)
            ax.add_patch(patch)

      fig.savefig(f"{base_dir}/potential_pivot_{pivot_number}.png")
      plt.close()


def plot_pivot_poins(original_map: np.ndarray, obs_array: np.ndarray, base_dir: str) -> None:
      """
      Plots the pivot candidates [x,y] locations from the original_map.
      """
      _, ax = plt.subplots(figsize=(8, 8))
      ax.imshow(original_map, cmap="gray")
      ax.scatter(obs_array[:, 0], obs_array[:, 1], c="r", s=1)

      # Save the plot to a file
      plt.savefig(f"{base_dir}/pivot_points.png")
      plt.close()


def plot_sight_rays(topdown: np.ndarray, patch: Circle, base_dir: str, iter_number: int) -> None:
      """
      Plots the sight rays from the topdown map.
      """
      _, ax = plt.subplots(figsize=(8, 8))
      ax.imshow(topdown, cmap="gray")
      ax.add_patch(patch)

      # Save the plot to a file
      plt.savefig(f"{base_dir}/sight_rays_{iter_number}.png")
      plt.close()


def plot_sight_rays_all(topdown: np.ndarray, patch: Arrow, grid_lines: np.ndarray, base_dir: str, pivot_number: int) -> None:
      """
      Plots the sight rays from the topdown map.
      """
      _, ax = plt.subplots(figsize=(8, 8))
      ax.imshow(topdown, cmap="gray")
      ax.add_patch(patch)
      for grid_line in grid_lines:
            x0, y0, px, py = grid_line
            ax.plot((x0, px), (y0, py), color="blue")

      # Save the plot to a file
      plt.savefig(f"{base_dir}/sight_rays_{pivot_number}.png")
      plt.close()


def plot_update_map_check(scene_name: str, pivot_1, explored_map: np.ndarray, base_dir: str, iter_number: int) -> None:
      """
      Plots the sight rays from the topdown map.
      """
      _, ax = plt.subplots(figsize=(8, 8))
      ax.set_title(scene_name + " " + str(pivot_1))
      ax.imshow(explored_map, cmap="gray")

      # Save the plot to a file
      plt.savefig(f"{base_dir}/update_map_check_{iter_number}.png")
      plt.close()


def plot_update_map(explored_map: np.ndarray, pivot, grid_pivot, scene_name: str, pivot_number: int, pivot_time: float, scene_time: float, floor_time: float, coverage: float, base_dir: str, iter_number: int) -> None:
      """
      Plots the sight rays from the topdown map.
      """
      fig, ax = plt.subplots()
      ax.set_title(scene_name + " " + str(pivot[1]))
      ax.imshow(explored_map, cmap="gray")
      ax.add_patch(
      Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red")
      )

      this_pivot_time = time() - pivot_time
      total_scene_time = time() - scene_time
      total_floor_time = time() - floor_time

      coverage = round(
      abs(coverage) * 100, 3
      )

      # convert total_floor_time and total_scene_time to minutes
      total_floor_time, total_scene_time = (
      total_floor_time / 60,
      total_scene_time / 60,
      )

      metrics = f"pivot_time: {this_pivot_time:>3.2f}s\nscene_time: {total_scene_time:>3.2f}min\nfloor_time: {total_floor_time:>3.2f}min"

      # plt.text(1, -0.1, metrics, ha='right', va='bottom', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
      plt.text(
      0.99,
      0.005,
      metrics,
      ha="right",
      va="bottom",
      transform=plt.gcf().transFigure,
      fontsize="small",
      bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
      )

      # set title to be scene, pivot number, then coverage
      ax.set_title(f"{scene_name}  iter{pivot_number} | {coverage}%")


      fig.savefig(os.path.join(
            f"{base_dir}",f"update_map_{pivot_number}_fc.png"))

      plt.close()


def plot_test_slice_pick_best_candidate(depth_vis:np.ndarray, iter_number: int, candidate_number: int, base_dir: str) -> None:
      """
      Plots the sight rays from the topdown map.
      """
      # Compare slice and original depth
      plt.imshow(depth_vis)
      plt.savefig(
            os.path.join(
            f"{base_dir}",f"test_slice_{iter_number}_candidate_{candidate_number}.png"
            )
      )
      plt.close()


def plot_update_grid_lines_test(grid_pivot:list,topdown: np.ndarray, grid_lines: np.ndarray,candidate: Candidate, scene_name:str, pivot_height,base_dir: str, iter_number: int, pivot_number: int) -> None:
      """
      Plots the sight rays from the topdown map.
      Called from intersect_score_simple
      """
      fig, ax = plt.subplots()
      ax.set_title(f"Scene {scene_name} {pivot_height}")
      ax.imshow(topdown)
      pivot=candidate.pivot
      cand_degree=candidate.angle

      cand_radians = np.radians(cand_degree)
      x0,y0=int(grid_pivot[0]),int(grid_pivot[1])

      ax.add_patch(
            Circle(
            (int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red"
            )
      )
      ax.add_patch(arrow(x0, y0, 100, cand_radians, "green"))

      fig.savefig(
            os.path.join(
            f"{base_dir}",f"update_grid_lines_test_{iter_number}_pivot_{pivot_number}.png"))
      
      plt.close()



def plot_update_map_check_test(explored_map:np.ndarray,height: float, scene_name:str, base_dir: str, pivot_number: int) -> None:
      """
      Plots the updated_map from the topdown map of the first raycast.
      Called from intersect_score_simple
      """
#     if plot_flag:
#         fig, ax = plt.subplots()
#         ax.set_title(scene_name + " " + str(pivot_c1[1]))
#         ax.imshow(temp_map_copy, cmap="gray")

#         fig.savefig(
#             os.path.join("temp", "More_vis", str(scene_name), str(iter_number), f"update_map_{iter}_1.png")
#         )
#         plt.close()
      fig,ax=plt.subplots()
      ax.set_title(f"Scene {scene_name} {height}")
      ax.imshow(explored_map,cmap="gray")

      fig.savefig(
            os.path.join(
            f"{base_dir}",f"update_map_{pivot_number}_1.png")
      )
      plt.close()



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# def visualize_graph(adj_matrix, output_file=None):
#       # Create a graph from the adjacency matrix
#       # directed

#       # don't draw the graph if it is empty
#       if len(adj_matrix) <= 1:
#             return
#       graph = nx.DiGraph(adj_matrix)
#       #graph = nx.Graph(adj_matrix)


#       # Get the positions of the nodes using a spring layout
#       pos = nx.spring_layout(graph)

#       # Normalize the adjacency matrix to use as edge weights
#       weights = np.array(adj_matrix) / np.max(adj_matrix)

#       color = (0.2, 0.4, 0.6)

#       # Draw the graph
#       fig, ax = plt.subplots()

#       # Crate labels (numbers) for the nodes
#       labels = {i: i for i in range(len(adj_matrix))}


#       nx.draw_networkx(graph, pos, node_size=100, node_color=color, ax=ax, labels=labels)

#       ax.axis('off')
#       ax.set_title('Graph')
#       ax.set_aspect('equal')
#       fig.show()

#       # save the graph
#       if output_file is not None:
#             fig.savefig(output_file)
#       #close
#       plt.close(fig)

#       # Show the plot
#       #plt.show()

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow, Circle
import matplotlib.cm as cmx
import math
import matplotlib.colors as color_


def arrow(x0, y0, length, angle, color):
    x1 = math.cos(np.pi / 2 + angle) * length
    y1 = math.sin(np.pi / 2 + angle) * length

    return Arrow(x0, y0, x1, -y1, width=100.0, color=color)


def visualize_graph(runner, adj_matrix, top_down_img, saved_pose, output_file=None):
    # Create a graph from the adjacency matrix
    # directed
    # normalize adj_matrix across rows
    adj_matrix = adj_matrix / np.sum(adj_matrix, axis=1, keepdims=True)

    # don't draw the graph if it is empty
    if len(adj_matrix) <= 1:
        return
    for i in range(len(adj_matrix)):
        adj_matrix[i][i] = 0
    graph = nx.DiGraph(adj_matrix)
    for i in range(len(adj_matrix)):
        adj_matrix[i][i] = 1
    # graph = nx.Graph(adj_matrix)

    # Get the positions of the nodes using a spring layout
    # pos = nx.spring_layout(graph)

    # Normalize the adjacency matrix to use as edge weights
    weights = np.array(adj_matrix) / np.max(adj_matrix)

    color = (0.2, 0.4, 0.6)

    # Draw the graph
    fig, ax = plt.subplots()
    # plot image
    ax.imshow(top_down_img)
    # Crate labels (numbers) for the nodes
    labels = {i: i for i in range(len(adj_matrix))}

    # nx.draw_networkx(graph, pos, node_size=100, node_color=color, ax=ax, labels=labels)

    ax.axis("off")
    ax.set_title("Graph")
    ax.set_aspect("equal")
    turbo = plt.get_cmap("turbo")
    cNorm = color_.Normalize(vmin=0, vmax=len(saved_pose))
    # plot the robot pose
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=turbo)
    scalarMap.set_array([])
    x_y_ = []
    for idx, svd_pose in enumerate(saved_pose):
        pivot_ = svd_pose[:3]
        pivot_[2], pivot_[1] = pivot_[1], pivot_[2]
        grid_pivot = runner.get_grid_pos(pivot_)
        x_y_.append([grid_pivot[0], grid_pivot[1]])
        # add number
        ax.scatter(grid_pivot[0], grid_pivot[1], s=100, c="red", label=idx)
        ax.text(
            grid_pivot[0],
            grid_pivot[1],
            idx,
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Add edges to the plot
    pos = {i: x_y_[i] for i in range(len(x_y_))}
    for i, j in graph.edges():
        ax.plot(
            [pos[i][0], pos[j][0]],
            [pos[i][1], pos[j][1]],
            "k-",
            lw=weights[i][j],
            color="black",
        )

    fig.show()

    # save the graph
    if output_file is not None:
        fig.savefig(output_file)
    # close
    plt.close(fig)

    # Show the plot
    # plt.show() 