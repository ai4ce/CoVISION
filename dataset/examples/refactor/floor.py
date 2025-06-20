import numpy as np

from folder_structure import FolderStructure
from maps import Maps
from typing import List, Tuple, Dict, Union
from typing import Any, TypeVar, Generic
from settings import default_sim_settings
import habitat_sim
from test_runner import TestRunner

from observations import Observations
from scipy.spatial.transform import Rotation as R
from plotter import *
from maps import *
from time import time
from PIL import Image

class Candidate():
      def __init__(self, obs, pivot, rotation, angle):
            self.obs = obs
            self.pivot = pivot
            self.rotation = rotation
            self.angle = angle


class CandidateScores():
      def __init__(self, overlap_score, diff_score, cloud_obs, cloud_pose, rel_ele):
            self.overlap_score = overlap_score
            self.diff_score = diff_score
            self.cloud_obs = cloud_obs
            self.cloud_pose = cloud_pose
            self.rel_ele = rel_ele


class TimeWrapper():
      def __init__(self):
            self._init_time=time()
            self._end_time=-1

      def get_time(self):
            if self._end_time==-1:
                  return time()-self._init_time
            return self._end_time-self._init_time
      
      def set_end_time(self):
            self._end_time=time()

      def clear(self):
            self._init_time=time()
            self._end_time=-1
            

class Floor():
      def __init__(self, scene_name: str,  floor_maps: List[np.ndarray[Any, Any]], floor_heights: List[float]):
            self.scene_name = scene_name
            self.floor_maps = floor_maps
            self.floor_heights = floor_heights
            

            # Start habitatsim navmensh
            self._init_test_runner()

            # Keeping tabs on (@TODO: check logic here)
            self.bounds = self.runner._sim.pathfinder.get_bounds()

            # algorithm data structures
            self.maps: List[Maps]= []
            self.observations: List[Observations] = []

            # extra
            self.focal_length = 364.86  ## in mm

            self.floor_timer=TimeWrapper()
            self.pivot_timer=TimeWrapper()
            self.stuck=False


      def get_scene_time(self) -> float:
            return self.floor_timer.get_time()
      
      def set_end_time(self)-> None:
            return self.floor_timer.set_end_time()
      
      def get_pivot_time(self) -> float:
            return self.pivot_timer.get_time()
      
      def clear_pivot_time(self)->None:
            return self.pivot_timer.clear()
            
            
            
      def _init_test_runner(self) -> None:
            """
            Initializes the test runner for a floor of a given scene.
            """
            # initialize the test runner
            default_sim_settings["scene"] = f"gibson/gibson/{self.scene_name}.glb"
            self.runner = TestRunner(default_sim_settings)

            # Navmesh settings
            navmesh_settings=habitat_sim.NavMeshSettings()

            use_custom_settings=False
            navmesh_settings.set_defaults()
            self.cam2base_h=1.5
            if use_custom_settings:
                  navmesh_settings.agent_height = self.cam2base_h
                  navmesh_settings.agent_max_climb = 0.2
                  navmesh_settings.filter_walkable_low_height_spans = False
                  navmesh_settings.region_merge_size = 10
                  navmesh_settings.verts_per_poly = 8.0

            navmesh_success = self.runner._sim.recompute_navmesh(self.runner._sim.pathfinder, navmesh_settings)

            assert navmesh_success, "Failed to build navmesh"


      def algorithm(self, resume_flag=False) -> None:
            """
            Algorithm to be implemented.
            """
            trial_limit=3
            for i, floor in enumerate(self.floor_maps):
                  self.maps.append(Maps(self.scene_name, i, np.array(floor,copy=True,dtype=float), self.floor_heights[i]))

                  self.observations.append(Observations(self.scene_name, i))

                  if not self.maps[i].is_valid:
                        # @TODO: different behavior for invalid vs already computed
                        print(f"Floor {i} is not valid or already computed.")
                        continue

                  # MAIN STEPS

                  # WHILE not full coverage
                  self.pivot_number=0
                  while not self.maps[i].coverage_sanity_check():

                        candidates = self.generate_pivot_candidates(i)

                        # @TODO: add plot logic from original code



                        # Pick the best candidate
                        best_index,max_score,selected_flag = self.pick_best_candidate(i, candidates,plot_flag=True)


                        if self.stuck:
                              self.stuck=False

                        if best_index is not None:
                              self.maps[i].seen_array = np.append(self.maps[i].seen_array, self.obs_inds[best_index])
                        else:
                              trial_limit -=1
                              stuck=True if trial_limit==0 else False
                              continue


                        # @TODO: add plot logic from original code

                        world_lines,grid_lines = self.maps[i].draw_sight_rays(
                              runner_grid_pos=self.runner.get_grid_pos(candidates[best_index].pivot),
                              runner_bounds=self.bounds,
                              runner_top_down=self.runner.get_top_down(height=candidates[best_index].pivot[1]),
                              candidate_at_best_index=candidates[best_index],
                              pivot_number=self.pivot_number
                        )

                        obs_,pivot_,rotation_,cand_rad=candidates[best_index].obs,candidates[best_index].pivot,candidates[best_index].rotation,candidates[best_index].angle

                        self.maps[i].explored_map,self.maps[i].stopping_map, _ = self.maps[i].update_map(
                              explored_map=self.maps[i].explored_map,
                              stopping_map=self.maps[i].stopping_map,
                              world_lines=world_lines,
                              grid_lines=grid_lines,
                              runner_grid_pos=self.runner.get_grid_pos(pivot_),
                              pivot=pivot_,
                              scene_time=self.get_scene_time(),
                              floor_time=self.maps[i].get_floor_generation_time(),
                              pivot_time=self.maps[i].pivot_time,
                              iter_number=i)
                        

                        
                        if self.maps[i].get_current_floor_coverage() > 0.7:
                              # save explored and stopping map
                              self.maps[i].save_explored_stopping_maps(
                                    self.maps[i].explored_map,
                                    self.maps[i].stopping_map,
                                    self.get_scene_time(),
                                    self.maps[i].pivot_time)
                              

                        # exclude pivots that are too close to the current pivot
                        exclude_array=np.arange(
                              self.obs_inds[best_index]-50,
                              self.obs_inds[best_index]+50,
                              0.5
                        )
                        self.maps[i].obs_array_index=np.setdiff1d(self.maps[i].obs_array_index,exclude_array)
                        


                        self.pivot_number+=1


      def generate_pivot_candidates(self, floor: int) -> List[Candidate]:
            """
            Processes a floor of a scene. Return a list of candidates from the Candidates class.
            """
            map =self.maps[floor]

            # Sample from pivot array candidates

            # np.array of length 6, each element is an int, representing the index int the map.pivot_array
            self.obs_inds = map.sample_pivots()

            

            height = self.floor_heights[floor]

            g1_array = map.obs_array[self.obs_inds] # np.array of length 6, each element is an Observation object

            # convert to list (@TODO: check if this is necessary)


            #[[posX, height, posY], ...]
            p1_array = [self.runner.get_pos_grid(g1[:2], height,meters_per_pixel=0.01) for g1 in g1_array] # np.array of length 6, each element is a tuple of ints

            rotation_array = [
                  R.from_euler("xyz", [0, g1[2], 0], degrees=True).as_quat() for g1 in g1_array
            ]

            candidates = []

            for angle, pivot, rotation in zip(g1_array[:,2], p1_array,rotation_array):
                  # get the observations from the pivot

                  new_pivot, _ = self.runner.get_random_point_near(
                                                height,
                        grid=map.original_map,
                        circle_center=pivot,
                        radius_grid=50,
                        pivot_list=map.pivot_array,
                        pivot_indicies=map.obs_array_index,
                        scalar_factor=50,
                  )

                  obs = self.runner.get_rgbd_from_pose(new_pivot,rotation)

                  candidates.append(Candidate(obs, new_pivot, rotation, angle))

            return candidates
      
      def pick_best_candidate(self, floor:int, candidates: List[Candidate],overlap_score_high_bound=0.4,plot_flag=False) ->Tuple[int, float, bool]:
            """
            Picks the best candidate from a list of candidates.
            """
            map = self.maps[floor]
            ALPHA=0.2
            max_score = -np.inf
            max_index = -1
            best_obs = None
            best_pose = None
            best_rel = None
            diff_score = np.inf

            if np.random.rand() < 0.01:
                  candidates = [c for c in candidates if c.obs["depth_sensor"].shape[0] > 0]



            
            for i, candidate in enumerate(candidates):

                  
                  obs, pivot, rotation, yaw = candidate.obs, candidate.pivot, candidate.rotation, candidate.angle

                  depth_image = obs["depth_sensor"]

                  width,height = depth_image.shape

                  # pcd=self.runner.convert_obs_to_pcd(obs,pivot,rotation,self.scene_name,floor,width=width,height=height,plot_flag=i)

                  # point_cloud_array=np.asarray(pcd.points)

                  

                  # Take slice of depth image at sensor height (center of image)
                  slice_ind = height//2
                  depth_vis=np.zeros(depth_image.shape)
                  depth_vis[slice_ind]=depth_image[slice_ind]
                  array_len=len(depth_vis[slice_ind])

                  offset=4
                  #@TODO: why not np.sum?
                  if (
                        sum(depth_vis[slice_ind]) <= 1000 * offset
                        or sum(depth_vis[slice_ind][int(array_len / 2) :]) <= 400 * offset
                        or sum(depth_vis[slice_ind][: int(array_len / 2)]) <= 400 * offset
                        or sum(depth_vis[slice_ind][int(array_len / 4) : int(array_len * 3 / 4)]) <= 400 * offset):
                        continue

                  if plot_flag:
                        plot_test_slice_pick_best_candidate(depth_vis,floor, i,map.vis_floor_dir)

                  rotation_ = [0,candidate.rotation,0]
                  pose=list(pivot.copy().squeeze())
                  pose.extend(rotation_)


                  # get intersect_score
                  candidate_scores=self.intersect_score_simple(floor,pose,candidate,i)
                  (overlap_score, diff_score, cloud_obs,cloud_pose,rel_ele) = candidate_scores.overlap_score, candidate_scores.diff_score, candidate_scores.cloud_obs, candidate_scores.cloud_pose, candidate_scores.rel_ele

                  #### Overlap not too large and not too small
                  if len(self.observations[floor].saved_depth) != 0 and overlap_score > overlap_score_high_bound and not self.stuck:
                        continue

                  weighted_score=(ALPHA*overlap_score)+(1-ALPHA)*diff_score

                  if (max_score < weighted_score) and overlap_score < overlap_score_high_bound:
                        max_score = weighted_score
                        max_index = i
                        best_obs=cloud_obs
                        best_pose=cloud_pose
                        best_rel_ele=rel_ele

            
            if max_index == -1 or diff_score ==0:
                  return None, None, saved_depth,saved_color,saved_pose,rel_mat,False
            
            #### Modification before saved
            best_pose[1] += self.cam2base_h
            # swap y and z
            temp=best_pose[1].copy()
            best_pose[1]=best_pose[2]
            best_pose[2]=temp
            best_obs["color_sensor"]=best_obs["color_sensor"][:,:,:3]


            # save depth,color,pose
            self.observations[floor].saved_depth=np.append(self.observations[floor].saved_depth,best_obs["depth_sensor"])
            self.observations[floor].saved_color=np.append(self.observations[floor].saved_color,best_obs["color_sensor"])
            self.observations[floor].saved_pose=np.append(self.observations[floor].saved_pose,best_pose)


            ### saving for visualization
            saved_grid_pose=self.observations[floor].saved_pose.copy()
            for idx, svd_pose in enumerate(saved_grid_pose):
                  pivot_ = svd_pose[:3]
                  pivot_[2], pivot_[1] = pivot_[1], pivot_[2]
                  grid_pivot = self.runner.get_grid_pos(pivot_)
                  saved_grid_pose[idx] = grid_pivot

            if map.get_current_floor_coverage() > 0.7:
                  self.observations[floor].save_dep_color_pose_gridpose()


            # Update rel_mat
            if self.observations[floor].rel_mat.shape[0] == 0:
                  self.observations[floor].rel_mat = np.eye(1,dtype=np.double)
            else:
                  len_rel_mat=len(self.observations[floor].rel_mat)
                  rel_mat_copy=np.eye(len_rel_mat,dtype=np.double)
                  rel_mat_copy[:,len_rel_mat,:len_rel_mat]=self.observations[floor].rel_mat

                  rel_mat_copy[-1,:]=best_rel_ele
                  rel_mat_copy[:,-1]=best_rel_ele.transpose()
                  self.observations[floor].rel_mat=rel_mat_copy


            # visualize
            visualize_graph(self.runner,self.observations[floor].rel_mat,self.runner.get_top_down(height=self.observations[floor].saved_pose[0][2]-1.5),self.observations[floor].saved_pose,output_file=os.path.join(
                  map.saved_rel_mats_dir, f"rel_mat_{len(self.observations[floor].rel_mat[0])}.png"))
            
            if map.get_current_floor_coverage() > 0.7:
                  self.observations[floor].save_rel_mat()

            image_pil = Image.fromarray(np.array(best_obs["color_sensor"]).astype(np.uint8))
            image_pil.save(
                  os.path.join(
                  map.saved_obs_dir,
                  map.saved_obs_dir, f"best_color_{len(self.observations[floor].saved_depth) - 1}.png"
                  ), "PNG"
            )
            depth_array = np.array(best_obs["depth_sensor"])
            depth_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
            depth_scaled = (depth_normalized * 255).astype(np.uint8)
            depth_image = Image.fromarray(depth_scaled)
            depth_image.save(
                  os.path.join(
                  map.saved_obs_dir, "depth", f"best_depth_{len(self.observations[floor].saved_depth) - 1}.png"
                  ), "PNG"
            )
            fig, ax = plt.subplots()
            ax.imshow(np.array(best_obs["depth_sensor"]))

            fig.savefig(
                  os.path.join(
                  map.saved_obs_dir, "depth", f"plt_best_depth_{len(self.observations[floor].saved_depth) - 1}.png"
                  )
            )
            plt.close()

            fig, ax = plt.subplots()
            turbo = plt.get_cmap("turbo")
            cNorm = color_.Normalize(vmin=0, vmax=len(self.observations[floor].saved_pose))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=turbo)
            scalarMap.set_array([])

            for idx, svd_pose in enumerate(self.observations[floor].saved_pose):
                  # print("svd_pose::"+str(svd_pose))
                  if idx == 0:
                        ax.set_title(f"{self.scene_name} {svd_pose[2] - 1.5}")
                        ax.imshow(self.runner.get_top_down(height=svd_pose[2] - 1.5))
                  length = 100
                  cand_radians = svd_pose[-2]
                  # print("cand_radians:::"+str(cand_radians))
                  # assert()
                  pivot_ = svd_pose[:3]
                  pivot_[2], pivot_[1] = pivot_[1], pivot_[2]
                  grid_pivot = self.runner.get_grid_pos(pivot_)
                  x0, y0 = int(grid_pivot[0]), int(grid_pivot[1])
                  colorVal = scalarMap.to_rgba(idx)
                  ax.add_patch(
                  Circle((int(grid_pivot[0]), int(grid_pivot[1])), radius=25, color="red")
                  )
                  ax.add_patch(arrow(x0, y0, length, math.radians(cand_radians), colorVal))

            fig.colorbar(cmx.ScalarMappable(norm=cNorm, cmap=turbo), ax=ax)
            fig.savefig(
                  os.path.join(
                  map.saved_obs_dir, "all_best_locs.png"
                  )
            )
            plt.close()
            return max_index, max_score, True


      
      def intersect_score_simple(self,floor:int,pose: List,candidate: Candidate,candidate_number:int,upper_threshold=0.5,plot_flag=False) -> CandidateScores:
            ALPHA=0.1

            map: Maps =self.maps[floor]
            max_overlap=-np.inf
            max_weight=-np.inf

            p1 = pose #xzyrpy
            c1 = candidate
            (obs_c1,pivot_c1,rotation_c1,candidates_rotation_rad_c1) = (c1.obs,c1.pivot,c1.rotation,c1.angle)

            g1 = self.runner.get_grid_pos(pivot_c1)
            rotation_1=[0,candidates_rotation_rad_c1,0]

            c1_pose=list(pivot_c1.copy())
            c1_pose.extend(rotation_1)
            c1_pose[1]+=self.cam2base_h

            #### xzy -> xyz
            temp = c1_pose[1]
            c1_pose[1] = c1_pose[2]
            c1_pose[2] = temp

            grid_pos=self.runner.get_grid_pos(pivot_c1)
            topdown=self.runner.get_top_down(height=pivot_c1[1])


            world_lines1,grid_lines1=map.draw_sight_rays(
                  runner_grid_pos=grid_pos,
                  runner_bounds=self.bounds,
                  runner_top_down=topdown,
                  candidate_at_best_index=c1,pivot_number=self.pivot_number)
            

            temp_map_copy=map.original_map.copy()
            temp_map2_copy=map.original_map.copy()

            if plot_flag:
                  plot_update_grid_lines_test(grid_pos,topdown, grid_lines1, candidate,self.scene_name,pivot_c1[1],map.vis_floor_dir, floor,self.pivot_number)


            # 1st raycast
            temp_map_copy,temp_map2_copy, count_1 = map.update_map(
                  explored_map=temp_map_copy,
                  stopping_map=temp_map2_copy,
                  world_lines=world_lines1,
                  grid_lines=grid_lines1,
                  runner_grid_pos=grid_pos,
                  pivot=pivot_c1,
                  scene_time=self.get_scene_time(),
                  floor_time=map.get_floor_generation_time(),
                  pivot_time=map.pivot_time,
                  iter_number=floor
            )

            explored_map_copy=map.explored_map.copy()
            stopping_map_copy=map.stopping_map.copy()


            _,_,count_2,o_count=map.update_map_check(
                  explored_map=explored_map_copy,
                  stopping_map=stopping_map_copy,
                  runner_grid_pos=grid_pos,
                  pivot=pivot_c1,
                  grid_lines=grid_lines1,
                  world_lines=world_lines1,
            )

            #@TODO: why not using count_1?
            diff_score=100*(count_2-o_count)/map.sum_original_map


            if plot_flag:
                  plot_update_map_check_test(temp_map_copy,pivot_c1[1],self.scene_name,map.vis_floor_dir,self.pivot_number)


            rel_ele=np.ones(1,dtype=float)
            max_rel_ele=rel_ele.copy()
            
            if len(self.observations[floor].saved_depth)==0:
                  return CandidateScores(0,count_1/map.sum_original_map,obs_c1,pose,rel_ele)
            

            len_saved_pose=len(self.observations[floor].saved_pose)
            rel_ele = np.zeros(len_saved_pose+1,dtype=float)
            rel_ele[-1]=1

            
            # calculate overlap score with each observation
            for ind_2, svd_pose in enumerate(self.observations[floor].saved_pose):
                  p2=copy(svd_pose)

                  # revert to base height
                  p2[2] -= self.cam2base_h

                  # swap y and z
                  temp = p2[2]
                  p2[2] = p2[1]
                  p2[1] = temp
                  
                  g2=self.runner.get_grid_pos(p2)

                  c2=self.get_candidate_GA(
                        pivot=p2,angle=p2[4],original_map=map.original_map,floor_no=floor,iter_number=candidate_number,
                        pivot_grid=self.runner.get_grid_pos(np.array(p2)),
                        plot_flag=plot_flag
                  )



                  (obs_c2, pivot_c2, rotation_c2, candidates_rotation_rad_c2) = c2.obs, c2.pivot, c2.rotation, c2.angle

                  rotation_2=[0,candidates_rotation_rad_c2,0]
                  c2_pose=list(pivot_c2.copy())
                  c2_pose.extend(rotation_2)

                  world_lines2,grid_lines2=map.draw_sight_rays(
                        runner_grid_pos=g2,
                        runner_bounds=self.bounds,
                        runner_top_down=topdown,
                        candidate_at_best_index=c2,
                        pivot_number=self.pivot_number+0.5
                  )

                  # @TODO: add plot from original code


                  temp_map=temp_map_copy.copy()
                  temp_map2=temp_map2_copy.copy()

                  temp_map,temp_map2,count_2,o_count=map.update_map_check(
                        explored_map=temp_map,
                        stopping_map=temp_map2,
                        runner_grid_pos=g2,
                        pivot=pivot_c2,
                        grid_lines=grid_lines2,
                        world_lines=world_lines2,
                  )

                  # @TODO: add plot from original code

                  overlap_score=o_count/count_2

                  p1_copy=p1.copy()
                  p2_copy=p2.copy()
                  if p1_copy[-2] < 0:
                        p1_copy[-2] += 360
                  if p2_copy[-2] < 0:
                        p2_copy[-2] += 360


                  if abs(p1_copy[-2] - p2_copy[-2]) > 150 and abs(p1_copy[-2] - p2_copy[-2]) < 210:#> 90:
                        overlap_score = 0
                        print("pivot: ", len_saved_pose + 1)
                        print("overlap_score: ", overlap_score)
                        print("ind_2: ", ind_2)

                  rel_ele[ind_2]=overlap_score

                  max_overlap=max(max_overlap,overlap_score)

                  weighted_score=(ALPHA*overlap_score)+(1-ALPHA)*diff_score

                  if max_weight < weighted_score:
                        max_weight=weighted_score
                        max_rel_ele=rel_ele

            return CandidateScores(max_overlap,diff_score,obs_c1,pose,rel_ele)







      def get_candidate_GA(self, pivot, angle, original_map,floor_no: int, iter_number: int, pivot_grid, plot_flag=False) -> Candidate:

            rotation=R.from_euler("xyz", [0, angle, 0], degrees=True).as_quat()

            # get pivot_grid
            pivot=np.array(pivot)
            

            # obtain new_pivot
            new_pivot, new_pivot_grid = self.runner.get_random_point_near(
                  height=self.floor_heights[floor_no],
                  grid=original_map,
                  circle_center=pivot,
                  radius_grid=50,
                  # pivot_list=pivot_grid,
                  # pivot_indicies=self.maps[floor_no].obs_array_index,
                  scalar_factor=100,
            )
            new_pivot=np.array(new_pivot)

            obs=self.runner.get_rgbd_from_pose(new_pivot,rotation)

            # @TODO: add plot_flag

            return Candidate(obs,new_pivot,rotation,angle)