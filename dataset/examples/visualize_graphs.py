import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow, Circle
import matplotlib.cm as cmx
import math
import matplotlib.colors as color_

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
    # weights = np.array(adj_matrix) / np.max(adj_matrix)
    weights = np.ones_like(adj_matrix)/3.0

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
    turbo = plt.get_cmap("gist_rainbow")
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