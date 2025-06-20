from folder_structure import FolderStructure
import os
import numpy as np

class Observations(FolderStructure):
      """Observations(scene: str, floor_index: int)
      Class to store observations (depth, color, pose, and rel_mat) for a given 
      scene and floor index.

      Stores the observations of a scene and floor index, these include the 
      depth, color, pose, and rel_mat (adjacency matrix) for all pivots of a 
      scene and floor index. 
      Inherits from the FolderStructure class.
      """
      def __init__(self, scene: str, floor_index: int) -> None:
            super().__init__(scene, floor_index)
            self.clear_observations()

      
      def save_dep_color_pose_gridpose(self) -> None:
            """
            Saves the depth, color, and pose for all pivots of a given scene and floor index.
            """
            np.save(os.path.join(self.saved_obs_dir, "saved_dep.npy"), self.saved_depth)
            np.save(os.path.join(self.saved_obs_dir, "saved_color.npy"), self.saved_color)
            np.save(os.path.join(self.saved_obs_dir, "saved_pose.npy"), self.saved_pose)

      def save_rel_mat(self) -> None:
            """
            Saves the rel_mat (adjacency matrix) for all pivots of a given scene and floor index.
            """
            np.save(os.path.join(self.saved_obs_dir, "rel_mat.npy"), self.rel_mat)

      def load_observation(self) -> None:
            """
            Loads the depth, color, pose, and rel_mat (adjacency matrix) for all pivots of a given scene and floor index.

            If no saved observations are found, then the function will pass.
            """
            try:
                  self.saved_depth = np.load(os.path.join(self.saved_obs_dir, "saved_dep.npy"))
                  self.saved_color = np.load(os.path.join(self.saved_obs_dir, "saved_color.npy"))
                  self.saved_pose = np.load(os.path.join(self.saved_obs_dir, "saved_pose.npy"))
                  self.rel_mat = np.load(os.path.join(self.saved_obs_dir, "rel_mat.npy"))
            except:
                  print("No saved observation found")
                  pass

      def clear_observations(self) -> None:
            """
            Clears the depth, color, pose, and rel_mat (adjacency matrix) for all pivots of a given scene and floor index, initializing them to empty NumPy arrays.
            """
            self.saved_depth = np.array([])
            self.saved_color = np.array([])
            self.saved_pose= np.array([])
            self.rel_mat= np.array([])
