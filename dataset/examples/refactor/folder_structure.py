import os

class FolderStructure():
      """
      Class for creating the folder structure for the visualizations and batch 
      indices. Automatically creates the directories if they do not exist. Used 
      to access the directories for the visualizations and batch indices.
      Inherited by the Observations and Maps classes.
      """
      def __init__(self, scene: str, floor_index: int, vis_dir="temp/More_vis",batch_dir="temp/batch_index") -> None: 
            """FolderStructure(scene_name: str, floor_index: int, vis_dir="temp/More_vis",batch_dir="temp/batch_index")
            """

            self.scene_name = scene
            self.floor_index = floor_index

            self.vis_dir = os.path.join(vis_dir, str(scene))
            self.batch_dir = os.path.join(batch_dir, str(scene))
            self.vis_floor_dir = os.path.join(self.vis_dir, str(floor_index))
            self.saved_obs_dir = os.path.join(self.vis_floor_dir, "saved_obs")
            self.saved_rel_mats_dir = os.path.join(self.vis_floor_dir, "rel_mats")

            self.create_vis_batch_dirs()
            self.create_floor_vis_batch_obs_dirs()
 

      def create_vis_batch_dirs(self) -> None:
            """
            Creates the directories for the visualizations and batch indices.
            """
            os.makedirs(self.vis_dir, exist_ok=True)
            os.makedirs(self.batch_dir, exist_ok=True)

      def create_floor_vis_batch_obs_dirs(self) -> None:
            """
            Creates the directories for the visualizations, batch indices, and observations of a floor index.
            """
            os.makedirs(os.path.join(self.vis_dir, str(self.floor_index)), exist_ok=True)
            os.makedirs(os.path.join(self.batch_dir, str(self.floor_index)), exist_ok=True)
            os.makedirs(self.saved_obs_dir, exist_ok=True)
            os.makedirs(self.saved_rel_mats_dir, exist_ok=True)
