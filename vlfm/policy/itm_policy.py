# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union
from collections import defaultdict
import cv2
import numpy as np
import glob
from torch import Tensor
import torch
import ast
import itertools
import time
import yaml
import math
from pathlib import Path
from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.mapping.traj_visualizer import TrajectoryVisualizer
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.utils.geometry_utils import xyz_yaw_to_extrinsic
from vlfm.utils.room_processor import PointLabelRegionProcessor
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
from vlfm.planner.SA_inte import SA
from openai import OpenAI
from scipy.spatial import cKDTree
try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"
class TorchActionIDs_plook:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
    LOOK_UP = torch.tensor([[4]], dtype=torch.long)
    LOOK_DOWN = torch.tensor([[5]], dtype=torch.long)

class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 3
    _circle_marker_radius: int = 5
    _circle_marker_radius_edge: int = 1
    _edge_color: Tuple[int, int, int] = (255, 0, 255)
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self.gpt_client = OpenAI(api_key="your key",base_url = "your url")
        self.RoomProcessor = PointLabelRegionProcessor()
        self._acyclic_enforcer = AcyclicEnforcer()
        self.result_path = os.environ.get("result_path")
        project_root = Path.cwd()
        outputs_dir = project_root / 'outputs'
        self.result_path = outputs_dir / self.result_path
        
        target_dir = os.path.dirname(f"{self.result_path}/merge_images")
        os.makedirs(target_dir, exist_ok=True)
        self.task_num = self.get_max_numbered_folder(f"{self.result_path}/merge_images")
        self.start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self._traj_vis = TrajectoryVisualizer(self._value_map._episode_pixel_origin, self._value_map.pixels_per_meter)
        self.looking_for_cross_stairs = False
        self._objects_attributes_dict = {"couch":{"attributes": [
    ["Living Room", "Media Room", "bedroom"],
    ["center of the Room", "against wall in the Room", "near window in the Room"],
    ["chair", "television", "floor lamp"],["couch"]
  ],"probability":[
    [0.70, 0.18, 0.12],
    [0.45, 0.35, 0.20],
    [0.48, 0.32, 0.20],[1]
  ]},"tv":{"attributes":  [
    ["Living Room", "Media Room", "Bedroom"],
    ["on tv stand in the Room", "against wall in the Room", "across bed in Bedroom"],
    ["remote control", "soundbar", "lamp"],["tv"]
  ],"probability":[  
    [0.62, 0.26, 0.12],
    [0.50, 0.30, 0.20],
    [0.44, 0.36, 0.20],[1]
  ]},"bed":{"attributes": [
    ["Bedroom", "Dressing Room", "child game room"],
    ["center of the room", "against wall in the room", "near window in room"],
    ["pillow", "blanket", "nightstand"],["bed"]
  ],"probability":
  [
    [0.75, 0.15, 0.10],
    [0.50, 0.30, 0.20],
    [0.42, 0.36, 0.22],[1]
  ]},"toilet":{"attributes":  [
    ["Bathroom", "hallway", "Living areas"],
    ["corner of the bathroom", "door by the long hallway", "door in Living areas"],
    ["sink", "toilet paper holder", "trash bin"],["toilet"]
  ],"probability": [
    [0.7, 0.2, 0.1],
    [0.50, 0.30, 0.20],
    [0.7, 0.2, 0.1],[1]
  ] },"potted plant":{"attributes":  [
    ["Living Room", "dinning room", "Bedroom"],
    ["near window in the Room", "corner of room", "at the wall"],
    ["window", "floor lamp", "bookshelf"],["potted plant"]
  ],"probability": [
    [0.7, 0.20, 0.10],
    [0.45, 0.35, 0.20],
    [0.44, 0.33, 0.23],[1]
  ]},"chair":{"attributes":   [
    ["Dining Room", "Living Room", "Home Office"],
    ["near the window in the room", "in front of desk in the room", "at the wall of the room"],
    ["table", "desk", "sofa"],["chair"]
  ],"probability":   [
    [0.50, 0.30, 0.20],
    [0.28, 0.52, 0.20],
    [0.45, 0.35, 0.20],[1]
  ]}}
        self.points_record = []
        self.ymal_path = "/media/magic-4090/DATA2/zzb/vlfm/outputs/path_record.yaml"
        self.room_candidate = ["Living Room", "Home Office", "Dining Room", "Bathroom","stairs", "Storage Room", "Kitchen","Bedroom","Toilet","corridor"]
        self.path_planner = SA(nparticles=400, T_init=10, T_final=5.5, alpha=0.97)
        self.room_coords = None
        self.room_categories = None
        
    def _reset(self) -> None:
        super()._reset()
        self._traj_vis.reset()
        self.step_num = 0
        self._value_map.reset()
        self.slam.reset()
        self.room_embedding = self.slam.semantic_query(self.room_candidate,flat = False)
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_point_num = 0
        self.stuck_count = 0
        self.infeasible_frontiers = []
        self._objects_attributes = None
        self.attributes_embeddings = None
        self.query_probability = None
        self._objects_observation_dict = {"bed":[1.0,3.0],"tv":[1.0,2.0],"couch":[1.0,3.0],"toilet":[0.7,1.5],"potted plant":[1.0,1.8],"chair":[1.0,2.0]}
        self._last_frontier = np.zeros(2)
        self.task_num+=1
        self.gain_dict = {}
        self.room_coords = None
        self.room_categories = None

        self.points_record = []
        

    def append_to_yaml_list(self, file_path, new_data):
        if not isinstance(new_data, list):
            raise ValueError("Input data must be a list.")
        with open(file_path, "a") as file:
            yaml.dump([new_data], file, default_flow_style=False)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        initial_frontiers = self._observations_cache["frontier_sensor"] 
        frontiers = [
            frontier for frontier in initial_frontiers if tuple(frontier) not in self._obstacle_map._disabled_frontiers
        ]
        def remove_matching_points(N: np.ndarray, M: np.ndarray) -> np.ndarray:
            M_set = set(map(tuple, M.tolist()))
            mask = np.array([tuple(point) not in M_set for point in N])
            return N[mask]
        
        if len(self.infeasible_frontiers) > 0 and len(frontiers) > 0:
            infeasible_frontiers_numpy = np.array(self.infeasible_frontiers)
            frontiers_numpy = np.array([np.array(f) for f in frontiers])
            frontiers_filter = remove_matching_points(frontiers_numpy,infeasible_frontiers_numpy)
            frontiers = frontiers_filter
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) < 1: # no frontier in this floor, check if there is stair
            if self._obstacle_map._reinitialize_flag == False and self._obstacle_map._floor_num_steps < 50 and (self._obstacle_map._explored_up_stair == False or self._obstacle_map._explored_down_stair == False): 

                if self._compute_frontiers:
                    temp_has_up_stair = self._obstacle_map._has_up_stair
                    if temp_has_up_stair:
                        temp_up_stair_map = self._obstacle_map._up_stair_map.copy()
                        temp_up_stair_start = self._obstacle_map._up_stair_start.copy()
                        temp_up_stair_end = self._obstacle_map._up_stair_end.copy()
                        temp_up_stair_frontiers = self._obstacle_map._up_stair_frontiers.copy()
                        temp_explored_up_stair = self._obstacle_map._explored_up_stair # .copy()
                    
                    temp_has_down_stair = self._obstacle_map._has_down_stair
                    if temp_has_down_stair:
                        temp_down_stair_map = self._obstacle_map._down_stair_map.copy()
                        temp_down_stair_start = self._obstacle_map._down_stair_start.copy()
                        temp_down_stair_end = self._obstacle_map._down_stair_end.copy()
                        temp_down_stair_frontiers = self._obstacle_map._down_stair_frontiers.copy()
                        temp_explored_down_stair = self._obstacle_map._explored_down_stair # .copy()
                                                
                    self._obstacle_map.reset()

                    if temp_has_up_stair:
                        self._obstacle_map._has_up_stair = temp_has_up_stair # .copy()
                        self._obstacle_map._up_stair_map = temp_up_stair_map.copy()
                        self._obstacle_map._up_stair_start = temp_up_stair_start.copy()
                        self._obstacle_map._up_stair_end = temp_up_stair_end.copy()
                        self._obstacle_map._up_stair_frontiers = temp_up_stair_frontiers.copy() 
                        self._obstacle_map._explored_up_stair = temp_explored_up_stair # .copy()
                    if temp_has_down_stair:
                        self._obstacle_map._has_down_stair = temp_has_down_stair # .copy()
                        self._obstacle_map._down_stair_map = temp_down_stair_map.copy()
                        self._obstacle_map._down_stair_start = temp_down_stair_start.copy()
                        self._obstacle_map._down_stair_end = temp_down_stair_end.copy()
                        self._obstacle_map._down_stair_frontiers = temp_down_stair_frontiers.copy() 
                        self._obstacle_map._explored_down_stair = temp_explored_down_stair # .copy()

                    self._obstacle_map._reinitialize_flag = True

                self._obstacle_map._tight_search_thresh = True 
                self._climb_stair_over = True
                self._reach_stair = False
                self._reach_stair_centroid = False
                self._stair_dilate_flag = False
                self._pitch_angle = 0
                self._done_initializing = False
                self._initialize_step = 0
                pointnav_action = self._initialize()
                return pointnav_action
            
            self._obstacle_map._this_floor_explored = True 
            has_unexplored_up_stair = (self._obstacle_map._has_up_stair and
                                    not self._obstacle_map._explored_up_stair)
            has_unexplored_down_stair = (self._obstacle_map._has_down_stair and
                                        not self._obstacle_map._explored_down_stair)

            if has_unexplored_up_stair or has_unexplored_down_stair:
                stair_direction = 1 if has_unexplored_up_stair else 2
                stair_frontiers = (self._obstacle_map._up_stair_frontiers if has_unexplored_up_stair
                                else self._obstacle_map._down_stair_frontiers)
                self._climb_stair_over = False
                self._climb_stair_flag = stair_direction
                self._stair_frontier = stair_frontiers
                floors_to_check = range(self._cur_floor_index + 1, len(self._obstacle_map_list)) if has_unexplored_up_stair else \
                                range(self._cur_floor_index - 1, -1, -1)

                for ith_floor in floors_to_check:
                    if not self._obstacle_map_list[ith_floor]._this_floor_explored:
                        pointnav_action = self._pointnav(self._stair_frontier[0])
                        return pointnav_action
                print("In all floors, no frontiers found during exploration, stopping.")
                return self._stop_action
            elif not self._obstacle_map._tight_search_thresh:
                self._obstacle_map._tight_search_thresh = True
                return TorchActionIDs_plook.MOVE_FORWARD
            else:
                print("No frontiers found during exploration, stopping.")
                return self._stop_action
        else:
            best_frontier,best_value, self.gain_dict = self._get_max_information_gain(observations, frontiers)
            os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
            pointnav_action = self._pointnav(best_frontier, stop=False)
            if pointnav_action.item() == 0:
                print("Might stop, change to move forward.")
                pointnav_action[0] = 1
                self.infeasible_frontiers.append(best_frontier)
            return pointnav_action

    def _get_object_attributes(self,target):
        completion = self.gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful robot to find an object in an unkown environment."},
            {
                "role": "user",
                "content": f'''Now that we need to find a/an {target}, please provide information about how it might be found in a private house, and these information will be embedded with clip and the information must be useful for robot to recognize the object.
        1. Provide what room it is possible to find.
        2. Please provide where it is likely to be found in some specific room, please add the room type in frount of the location. 
        3. Please provide what other items it is likely to be around.
        At the same time, the output meets the following requirements:
        1. Output three related strings for each level, and each string only contain one information about the level. without 'or' description.
        2. each information only contain the most relevant phrases instead of complete sentences, keep phrases simple and common, and remove words like maybe.
        3. Only one two-dimensional list is output, the first dimension is the information level, and the second dimension is the three related strings of each information level.
        4. Output only string and no other text.'''
            }
        ],
        )
        attributes_list = ast.literal_eval(completion.choices[0].message.content)
        return attributes_list
    
    
    def _get_observation_range_attributes(self,target):
        completion = self.gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful robot to find an object in an unkown environment."},
            {
                "role": "user",
                "content": f'''when we use a yolov7 to detect a/an {target} in a simulate mesh private house environment, The quality is bad. what is the best distance from the camera pose to the {target} to detect the {target}. THe input image are 640*480 resolution.
1. please output the distance in meters.
2. please only output the distance range with a list [distance min,distance max] without any other text.'''
            }
        ],
        )
        attributes_list = ast.literal_eval(completion.choices[0].message.content)
        return attributes_list   
    
    
    def get_max_numbered_folder(self,directory):
        folders = glob.glob(os.path.join(directory, "[0-9]*"))  # 只匹配以数字开头的文件夹
        numbered_folders = [int(os.path.basename(folder)) for folder in folders if os.path.isdir(folder) and os.path.basename(folder).isdigit()]
        return max(numbered_folders) if numbered_folders else int(0)

    import math

    def calculate_yaw(self, ego_pose,target_pose, is_navigation=False):
        dx = target_pose[0] - ego_pose[0]
        dy = target_pose[1] - ego_pose[1]
        theta_rad = math.atan2(dy, dx)
        return theta_rad


    def extract_room_map(self):
        room_features = self.slam.semantic_query(self.room_candidate)
    
    def _get_max_information_gain(self,observation,frontiers): 
        if self._target_object not in self._objects_attributes_dict.keys():
            objects_attributes = self._get_object_attributes(self._target_object)
            objects_attributes += [[self._target_object]]
            self._objects_attributes_dict.update({self._target_object:objects_attributes})
        
        query_attributes = self._objects_attributes_dict[self._target_object]["attributes"]
        query_probability = self._objects_attributes_dict[self._target_object]["probability"]
        query_probability_flat = list(itertools.chain(*query_probability))
        query_probability_flat = torch.tensor(query_probability_flat)
        
        self.attributes_embeddings = self.slam.semantic_query(query_attributes)
        self.query_probability = query_probability_flat
        camera_yaw = [0, 0.5 * np.pi, -0.5*np.pi,np.pi]
        extrinsics_list = []
        ego_pose = self._observations_cache["robot_xy"]
        best_observation_range = self._objects_observation_dict[self._target_object]
        agent_height = self._observations_cache["robot_height"]
        for frontier in frontiers:
            frontier = np.append(frontier, self._camera_height + agent_height)
            base_yaw = 0
            for yaw in camera_yaw:
                search_yaw = base_yaw + yaw
                extrinsic = xyz_yaw_to_extrinsic(frontier,search_yaw)
                extrinsics_list.append(extrinsic)
        information_gain_list ,explore_gain_list, semantic_gain_list, fov_mask_list, points_in_fov_list, semantic_points_list = self.slam.get_information_gain(self.attributes_embeddings,self.query_probability,extrinsics_list,best_observation_range)
        frontier_yaw_info = []
        idx = 0
        gain_dict = {}
        frontier_map = defaultdict(list)
        frontier_map_explore = defaultdict(list)
        frontier_map_semantic = defaultdict(list)
        frontier_map_yaw = defaultdict(list)
        frontier_map_fov = defaultdict(list)
        frontier_map_fov_points = defaultdict(list)
        frontier_map_semantic_points = defaultdict(list)
        for frontier in frontiers:
            base_yaw = 0
            distance = math.sqrt((frontier[0] - ego_pose[0]) ** 2 + (frontier[1] - ego_pose[1]) ** 2)
            for yaw in camera_yaw:
                information_gain = information_gain_list[idx]
                explore_gain = explore_gain_list[idx]
                semantic_gain = semantic_gain_list[idx]
                fov_mask = fov_mask_list[idx]
                frontier_map[tuple(frontier)].append(information_gain)
                frontier_map_explore[tuple(frontier)].append(explore_gain)
                frontier_map_semantic[tuple(frontier)].append(semantic_gain)
                frontier_map_yaw[tuple(frontier)].append(yaw + base_yaw)
                frontier_map_fov[tuple(frontier)].append(fov_mask)
                frontier_map_fov_points[tuple(frontier)].append(points_in_fov_list[idx])
                frontier_map_semantic_points[tuple(frontier)].append(semantic_points_list[idx])
                idx += 1
        path_dict = {"robot_xy":ego_pose.tolist(),"robot_heading":camera_yaw}
        for frontier, gains in frontier_map.items():
            max_index = gains.index(max(gains)) 
            max_information_gain = max(gains)
            max_explore = frontier_map_explore[frontier][max_index]
            max_semantic = frontier_map_semantic[frontier][max_index]
            max_yaw = frontier_map_yaw[frontier][max_index]
            final_mask = frontier_map_fov[frontier][max_index]
            final_mask_points = frontier_map_fov_points[frontier][max_index]
            semantic_points = frontier_map_semantic_points[frontier][max_index]
            frontier_yaw_info.append((max_information_gain, list(frontier)))
            gain_dict.update({tuple(frontier):[max_information_gain,max_explore,max_semantic,max_yaw,final_mask,final_mask_points,semantic_points]})
        frontier_yaw_info.sort(key=lambda x: x[0], reverse=True)
        sorted_pts = []
        sorted_values = []
        sorted_pts_with_yaw = []
        for info_gain, frontier in frontier_yaw_info:
            sorted_pts.append(np.array([frontier[0], frontier[1]]))  
            sorted_pts_with_yaw.append(np.array([frontier[0], frontier[1]]))  
            sorted_values.append(np.array(info_gain.cpu()))  
        sorted_pts = np.array(sorted_pts)
        sorted_values = np.array(sorted_values)  
             
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])
        sorted_pts_current = np.vstack([ego_pose[:2], sorted_pts])
        sorted_values_current = np.hstack([0, sorted_values])
        path_result = self.path_planner.infer(sorted_pts_current, sorted_values_current)
        sorted_pts = path_result['sorted_pts'][1:]
        sorted_values = path_result['sorted_values'][1:]
        sorted_pts = np.array(sorted_pts)
        sorted_values = np.array(sorted_values)
        os.environ["DEBUG_INFO"] = ""
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p[:2], self._last_frontier):
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)
                if closest_index != -1:
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_index < 2:
                    os.environ["DEBUG_INFO"] += "Sticking to last point."
                    distance_ego_target = math.sqrt((sorted_pts[curr_index][0] - robot_xy[0]) ** 2 + (sorted_pts[curr_index][1] - robot_xy[1]) ** 2)
                    if self.stuck_count <= 50 and distance_ego_target < 1.5 and np.array_equal(sorted_pts[curr_index], self._last_goal):
                        self.stuck_count+=1
                    
                    if self._last_point_num <= 120:
                        self._last_point_num+=1                        
                    
                    if self._last_point_num >=80 or self.stuck_count>=25:
                        self.infeasible_frontiers.append(sorted_pts[curr_index])
                        self._last_point_num = 0
                        self.stuck_count = 0
                    else:
                        best_frontier_idx = curr_index
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                self._last_point_num = 0
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )
            self._last_point_num = 0

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value:.2f}%"

        return best_frontier,best_value,gain_dict

        
    
    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)
        score_image = None
        if not self._visualize:
            return policy_info
        markers = []
        markers_gain = []
        frontiers = self._observations_cache["frontier_sensor"]
        fov_mask = None
        mask_point = None
        semantic_points = None
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            if tuple(frontier) in self.gain_dict.keys():
                gain = self.gain_dict[tuple(frontier)]
                text = f"info: {int(gain[0])}, e: {int(gain[1])}, s: {int(gain[2])}"
                yaw_view = gain[3]
                markers_gain.append((frontier[:2], text, yaw_view, marker_kwargs))
                if np.array_equal(frontier, self._last_goal):
                    fov_mask = gain[4]
                    fov_mask = fov_mask.cpu().numpy()
                    fov_mask = (fov_mask * 255).astype(np.uint8)
                    # print("mask: ",fov_mask.shape)
                    mask_point = gain[5]
                    semantic_points = gain[6]
            else:
                markers.append((frontier[:2], marker_kwargs))
                
        if self.attributes_embeddings is not None:
            buf_coords, score_map, semantic_gain, active_uncertanty_value, cluster_points_list,self.room_coords,self.room_categories = self.slam.get_uninspected_points(self.attributes_embeddings, self.room_embedding,self.query_probability)
            if len(buf_coords.shape) < 2:
                return policy_info
            
            observed_points_2d = buf_coords[:, :2]
            observed_unique_points, observed_inverse_indices = torch.unique(observed_points_2d, dim=0, return_inverse=True)
            observed_M = observed_unique_points.shape[0]
            score_unique_values = torch.empty(observed_M, dtype=score_map.dtype, device=score_map.device)
            semantic_gain_unique_values = torch.empty(observed_M, dtype=semantic_gain.dtype, device=semantic_gain.device)
            active_uncertanty_unique_values = torch.empty(observed_M, dtype=active_uncertanty_value.dtype, device=active_uncertanty_value.device)
            
        
            room_coords_2d = self.room_coords[:,:2]
            observed_unique_points_room, observed_inverse_indices_room = torch.unique(room_coords_2d, dim=0, return_inverse=True)
            observed_M_room = observed_unique_points_room.shape[0]
            score_unique_values_room = torch.empty(observed_M_room, dtype=self.room_categories.dtype, device=self.room_categories.device)
            for i in range(observed_M_room):
                mask_room = (observed_inverse_indices_room == i)
                categories_room = self.room_categories[mask_room]
                unique_categories, counts = categories_room.unique(return_counts=True)
                most_common_category_idx = torch.argmax(counts)
                most_common_category = unique_categories[most_common_category_idx]
                score_unique_values_room[i] = most_common_category
            for i in range(observed_M):
                mask = (observed_inverse_indices == i)
                score_unique_values[i] = score_map[mask].max()
                semantic_gain_unique_values[i] = semantic_gain[mask].max()
                active_uncertanty_unique_values[i] = active_uncertanty_value[mask].min()
            if semantic_points is not None and semantic_points.shape[0] > 0:
                semantic_points_image = self._traj_vis.draw_points(points= semantic_points, values= torch.ones(semantic_points.shape[0],dtype=torch.uint8))
                semantic_gain_unique_points = torch.cat((observed_unique_points,semantic_points),dim=0)
                semantic_gain_unique_values_with_semantic_points = torch.cat((semantic_gain_unique_values,torch.ones(semantic_points.shape[0],dtype=torch.uint8).to(semantic_gain_unique_values.device)),dim=0)
            else:
                semantic_gain_unique_points = observed_unique_points
                semantic_gain_unique_values = semantic_gain_unique_values
                semantic_points_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
                semantic_gain_unique_values_with_semantic_points = semantic_gain_unique_values
            score_image = self._traj_vis.draw_points(points= observed_unique_points, values= score_unique_values)
            uncertanty_image = self._traj_vis.draw_points(points= observed_unique_points, values= active_uncertanty_unique_values)
            semantic_gain_image = self._traj_vis.draw_points(points= observed_unique_points, values= semantic_gain_unique_values)
            semantic_gain_image_with_semantic_points = self._traj_vis.draw_points(points= semantic_gain_unique_points, values= semantic_gain_unique_values_with_semantic_points)
            room_categories_image = self._traj_vis.draw_points_room(points=observed_unique_points_room,indices= score_unique_values_room)
        if not np.array_equal(self._last_goal, np.zeros(2)):
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        if markers_gain is not None and score_image is not None:
            for pos,text,yaw_view, marker_kwargs in markers_gain:
                score_image = self._traj_vis.draw_circle(score_image, pos, text = text, **marker_kwargs)
                uncertanty_image = self._traj_vis.draw_circle(uncertanty_image, pos, text = text, **marker_kwargs)
                semantic_gain_image = self._traj_vis._draw_yaw(semantic_gain_image,pos,yaw_view)
        
        if markers is not None and score_image is not None:
            for pos, marker_kwargs in markers:
                score_image = self._traj_vis.draw_circle(score_image, pos, **marker_kwargs)
                uncertanty_image = self._traj_vis.draw_circle(uncertanty_image, pos, **marker_kwargs)
                
            semantic_flat = list(itertools.chain(* self._objects_attributes_dict[self._target_object]))
            title_text = f"target object: {self._target_object}, last_goal: {self._last_goal[:2]}, mode: {self.mode}"
            new_height = policy_info["obstacle_map"].shape[0]
            orig_height, orig_width = policy_info["annotated_rgb"].shape[:2]
            new_width = int(orig_width * new_height / orig_height)
            resized_rgb = cv2.resize(policy_info["annotated_rgb"], (new_width, new_height))
            
            
            if fov_mask is not None and fov_mask.shape[0] > 0:
                resized_mask = cv2.resize(fov_mask, (new_width, new_height))
            else:
                resized_mask = np.zeros((new_height,new_width),dtype=np.uint8)   
            if mask_point is not None and mask_point.shape[0] > 0:
                resozed_mask_point = cv2.resize(mask_point, (new_width, new_height))
            else:
                resozed_mask_point = np.zeros((new_height,new_width,3),dtype=np.uint8)    
            
            resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
            
            if len(self.slam._camera_positions) > 0:
                self._traj_vis.draw_trajectory(
                    score_image,
                    self.slam._camera_positions,
                    self.slam._last_camera_yaw,
                )
                self._traj_vis.draw_trajectory(
                    uncertanty_image,
                    self.slam._camera_positions,
                    self.slam._last_camera_yaw,
                )
            
            merged_image = cv2.hconcat([cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR), policy_info["obstacle_map"],score_image, uncertanty_image,semantic_gain_image,semantic_gain_image_with_semantic_points,room_categories_image])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            color = (0, 0, 0)  
            (text_width, text_height), baseline = cv2.getTextSize(title_text, font, font_scale, font_thickness)
            merged_image_width = merged_image.shape[1]
            merged_image_height = merged_image.shape[0]
            text_x = (merged_image_width - text_width) // 2  
            text_y = text_height + 20  
            cv2.putText(merged_image, title_text, (text_x, text_y), font, font_scale, color, font_thickness)
            os.makedirs(f'{self.result_path}/merge_images/{self.task_num}', exist_ok=True)
            cv2.imwrite(f'{self.result_path}/merge_images/{self.task_num}/{self.step_num}.jpg', merged_image)
            self.step_num+=1
        else:
            title_text = f"target object: {self._target_object}, last_goal: {self._last_goal[:2]}, mode: {self.mode}"
            new_height = policy_info["obstacle_map"].shape[0]
            orig_height, orig_width = policy_info["annotated_rgb"].shape[:2]
            new_width = int(orig_width * new_height / orig_height)
            resized_rgb = cv2.resize(policy_info["annotated_rgb"], (new_width, new_height))
            merged_image = cv2.hconcat([cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR), policy_info["obstacle_map"]])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            color = (0, 0, 0) 
            (text_width, text_height), baseline = cv2.getTextSize(title_text, font, font_scale, font_thickness)
            merged_image_width = merged_image.shape[1]
            merged_image_height = merged_image.shape[0]
            text_x = (merged_image_width - text_width) // 2  
            text_y = text_height + 20 
            
            
            # 在图像上添加文本
            cv2.putText(merged_image, title_text, (text_x, text_y), font, font_scale, color, font_thickness)
            # print("save path: ",f'/media/magic-4090/DATA2/zzb/vlfm/outputs/final_result/merge_images{self.task_num}/{self.step_num}.jpg')
            os.makedirs(f'{self.result_path}/merge_images/{self.task_num}', exist_ok=True)
            cv2.imwrite(f'{self.result_path}/merge_images/{self.task_num}/{self.step_num}.jpg', merged_image)
            self.step_num+=1
            # print("save image in: ",f'/media/magic-4090/44ee543b-82db-4f62-aa8d-c1ad5dd806dc2/zzb/vlfm/images/merged_map/{self.start_time}/{self.task_num}/{self.step_num}.jpg' )
        # self.video_writer.write(cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR))
        
        
        return policy_info

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError



    def _update_openspace(self) -> None:
        batch_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        batch_depth = [i[1] for i in self._observations_cache["value_map_rgbd"]]
        batch_extrinsic = [i[3] for i in self._observations_cache["value_map_rgbd"]]
        agent_height = [i[4] for i in self._observations_cache["value_map_rgbd"]]
        if self._target_object not in self._objects_observation_dict.keys():
             self._objects_observation_dict.update({self._target_object:self._get_observation_range_attributes(self._target_object)})
             print(f"self._objects_observation_dict: ",self._objects_observation_dict)
        best_observation_range = self._objects_observation_dict[self._target_object]
        self.slam.compute_state(batch_rgb, batch_depth, batch_extrinsic, agent_height,observation_range = best_observation_range)
        self.slam.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError




class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        # if self._visualize:
        #     # self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        print("_sort_frontiers_by_valuev1")
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        # obaservation
        self._pre_step(observations, masks)
        ########## TODO update  the openspace map ############
        # self._update_value_map()
        
        self._update_openspace()
        ######################################################
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)


    # def _sort_frontiers_by_value(
    #     self, observations: "TensorDict", frontiers: np.ndarray
    # ) -> Tuple[np.ndarray, List[float]]:
    #     sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
    #     return sorted_frontiers, sorted_values
    # # def _sort_frontiers_by_information_gain(self,observations: "TensorDict", frontiers: np.ndarray)-> Tuple[np.ndarray, List[float]]:
        





class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore


    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
