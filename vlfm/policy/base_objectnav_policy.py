import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union
import base64
import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor
from vlfm.utils.geometry_utils import transform_points
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.map_utils import ObstacleMapUpdater
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client
from vlfm.vlm.openai_api import OpenAI_API
from vlfm.RedNet.RedNet_model import load_rednet
import re
from pathlib import Path
try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  
        pass

STAIR_CLASS_ID = 17
def check_stairs_in_upper_50_percent(mask):
    height = mask.shape[0]
    upper_50_height = int(height * 0.5)
    upper_50_mask = mask[:upper_50_height, :]
    
    print(f"Stair upper 50% points: {np.sum(upper_50_mask)}")
    if np.sum(upper_50_mask) > 50:  
        return True
    return False

class TorchActionIDs_plook:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
    LOOK_UP = torch.tensor([[4]], dtype=torch.long)
    LOOK_DOWN = torch.tensor([[5]], dtype=torch.long)

class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  
    _stop_action: Union[Tensor, Any] = None  
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        gpu_id = int(os.environ.get("gpu_id"))
        gpu_num = int(os.environ.get("gpu_num"))
        super().__init__()
        self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", f"{12181 + (gpu_id) * gpu_num}")))
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", f"{12184+ (gpu_id) * gpu_num}")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", f"{12183 + (gpu_id) * gpu_num}")))
        self._use_vqa = use_vqa
        self.counter = 0
        self.min_distance_xy = np.inf
        self.navigation_steps = 0
        self.invalied_navigation_goal = []
        self.result_path = os.environ.get("result_path")
        project_root = Path.cwd()
        outputs_dir = project_root / 'outputs'
        self.result_path = outputs_dir / self.result_path
        print("self.result_path: ", self.result_path)
        target_dir = os.path.dirname(self.result_path)
        os.makedirs(target_dir, exist_ok=True)
        vlm_dir = self.result_path / 'VLM'
        vlm_dir.mkdir(parents=True, exist_ok=True)
        self.look_steps = 0
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = True
        self._last_object_goal = None
        self._last_object_goal_distance = None
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold
        self._objects_attributes = None
        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._try_to_navigate_step = 0
        self._compute_frontiers = compute_frontiers
        self.red_semantic_pred = None
        self._last_frontier_distance = 0
        self.min_obstacle_height = min_obstacle_height
        self.max_obstacle_height = max_obstacle_height
        self.obstacle_map_area_threshold = obstacle_map_area_threshold
        self.agent_radius = agent_radius
        self.hole_area_thresh  = hole_area_thresh
        if "full_config" in kwargs:
            self.device = (
                torch.device("cuda:{}".format(kwargs["full_config"].habitat_baselines.torch_gpu_id))
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self._pitch_angle_offset = kwargs["full_config"].habitat.task.actions.look_down.tilt_angle
        else:
            self.device = (
                torch.device("cuda:0") 
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self._pitch_angle_offset = 30
        project_root = Path.cwd()
        data_dir = project_root / 'data'
        self.red_sem_pred = load_rednet(
            self.device, ckpt=f'{data_dir}/rednet_semmap_mp3d_40.pth', resize=True, 
        )
        self.red_sem_pred.eval()

        self._climb_stair_flag = 0
        if self._compute_frontiers:
            self._obstacle_map_list = [ObstacleMapUpdater(
                    min_height=self.min_obstacle_height,
                    max_height=self.max_obstacle_height,
                    area_thresh=self.obstacle_map_area_threshold,
                    agent_radius=self.agent_radius,
                    hole_area_thresh=self.hole_area_thresh,
                    size=1000,
                )]
        self._climb_stair_over = True
        self._reach_stair = False
        self._reach_stair_centroid = False
        self._stair_frontier = None
        self._cur_floor_index = 0
        self._climb_stair_flag = 0
        self._stair_dilate_flag = False
        self.seg_map_color_list = []
        self._temp_stair_map = []
        self._last_carrot_xy = []
        self._last_carrot_px = []
        self._carrot_goal_xy = []
        self._pitch_angle = 0
        self._obstacle_map = [self._obstacle_map_list[self._cur_floor_index]]
        self.floor_num = len(self._obstacle_map_list)
        self.openai_client = OpenAI_API()


    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._last_object_goal = None
        self._last_object_goal_distance = None
        self.navigation_steps = 0
        self.invalied_navigation_goal = []
        self.look_steps = 0
        self.mode = None
        self._done_initializing = False
        self._called_stop = False
        self.min_distance_xy = np.inf
        self._did_reset = True
        
        self._cur_floor_index = 0
        self.fist_detect = True
        self.counter +=1
        self._try_to_navigate_step = 0
        self.double_detect = True
        self._object_map.reset()
        self._cur_floor_index = 0
        if self._compute_frontiers:
            self._obstacle_map = self._obstacle_map_list[0]
            self._obstacle_map.reset()
            del self._obstacle_map_list[1:]
        self.floor_num = len(self._obstacle_map_list)
        self._initialize_step = 0
        self._climb_stair_over = True
        self._reach_stair = False
        self._reach_stair_centroid = False
        self._stair_dilate_flag = False
        self._pitch_angle = 0
        self._climb_stair_flag = 0
        self._get_close_to_stair_step = 0
        self._frontier_stick_step = 0
        self._last_frontier_distance = 0
        self._has_climbed_once = False
        self._stair_masks = []
        self._last_carrot_xy = []
        self._last_carrot_px = []
        self._carrot_goal_xy = []
        self._temp_stair_map = []
    
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)
        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        rgb = torch.unsqueeze(observations["rgb"][0], dim=0).float()
        depth = torch.unsqueeze(observations["depth"][0], dim=0).float()
        
        with torch.no_grad():
            self.red_semantic_pred = self.red_sem_pred(rgb, depth)
            self.red_semantic_pred = self.red_semantic_pred.squeeze().cpu().detach().numpy().astype(np.uint8)
        self._update_obstacle_map()
        
        
        
        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_target_object_location(robot_xy)
        robot_xy_2d = np.atleast_2d(robot_xy) 
        robot_px = self._obstacle_map._xy_to_px(robot_xy_2d)
        x, y = robot_px[0, 0], robot_px[0, 1]
        if goal is not None:
            g = np.asarray(goal[:2], dtype=np.float32)
            r = np.asarray(robot_xy[:2], dtype=np.float32)
            if len(self.invalied_navigation_goal) > 0:
                invalid_xy = np.atleast_2d(np.array(self.invalied_navigation_goal, dtype=np.float32))
                if np.min(np.linalg.norm(invalid_xy - g, axis=1)) < 2.0:
                    self.navigation_steps = 0
                    self._last_object_goal = None
                    self._try_to_navigate_step = 0
                    self.min_distance_xy = np.inf
                    self._object_map.reset()
                    goal = None
                else:
                    if self._last_object_goal is None or np.linalg.norm(g - self._last_object_goal) > 0.2:
                        self._last_object_goal = g
                        self.navigation_steps = 0  
                    if np.linalg.norm(g - r) < 2.0:
                        if self.mode == "navigate":
                            self.navigation_steps += 1

                    if self.navigation_steps > 20 or self._try_to_navigate_step > 80:
                        print("invalid navigation goal, explore")
                        self.invalied_navigation_goal.append(g)  
                        self.navigation_steps = 0
                        self._last_object_goal = None
                        self._try_to_navigate_step = 0
                        self.min_distance_xy = np.inf
                        self._object_map.reset()
                        goal = None
            else:
                if self._last_object_goal is None or np.linalg.norm(g - self._last_object_goal) > 0.2:
                    self._last_object_goal = g
                    self.navigation_steps = 0

                if np.linalg.norm(g - r) < 2.0:
                    if self.mode == "navigate":
                        self.navigation_steps += 1
                        print("self.navigation_steps: ",self.navigation_steps)

                if self.navigation_steps > 20:
                    print("invalid navigation goal, explore")
                    self.invalied_navigation_goal.append(g)
                    self.navigation_steps = 0
                    self._last_object_goal = None
                    self._try_to_navigate_step = 0
                    self.min_distance_xy = np.inf
                    self._object_map.reset()
                    goal = None
        else:
            self._try_to_navigate_step = 0
                
            

        if self._climb_stair_over == True and self._obstacle_map._down_stair_map[y,x] == 1 and len(self._obstacle_map._down_stair_frontiers) > 0 and self._obstacle_map_list[self._cur_floor_index - 1]._explored_up_stair == False:
            self._reach_stair = True
            self._get_close_to_stair_step = 0
            self._climb_stair_over = False
            self._climb_stair_flag = 2
            self._obstacle_map._down_stair_start = robot_px[0].copy()


        elif self._climb_stair_over == True and self._obstacle_map._up_stair_map[y,x] == 1 and len(self._obstacle_map._up_stair_frontiers) > 0 and self._obstacle_map_list[self._cur_floor_index + 1]._explored_down_stair == False:
            self._reach_stair = True
            self._get_close_to_stair_step = 0
            self._climb_stair_over = False
            self._climb_stair_flag = 1
            self._obstacle_map._up_stair_start = robot_px[0].copy()
        if self._climb_stair_over == False:
            if self._reach_stair == True:
                if self._pitch_angle == 0 and self._climb_stair_flag == 2: 
                        self._pitch_angle -= self._pitch_angle_offset
                        self.mode = "look_down"
                        pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                elif self._climb_stair_flag == 2 and self._pitch_angle >= -30 and self._reach_stair_centroid == False: 
                        self._pitch_angle -= self._pitch_angle_offset
                        self.mode = "look_down_twice"
                        pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                else:
                    if self._obstacle_map._climb_stair_paused_step < 30:
                        self.mode = "climb_stair"
                        pointnav_action = self._climb_stair(observations, masks)
                    else:
                        if self._climb_stair_flag == 1 and self._obstacle_map_list[self._cur_floor_index+1]._done_initializing == False:
                            self._done_initializing = False
                            self._initialize_step = 0
                            self._obstacle_map._explored_up_stair = True
                            self._cur_floor_index += 1
                            self._obstacle_map = self._obstacle_map_list[self._cur_floor_index]
                            ori_up_stair_map = self._obstacle_map_list[self._cur_floor_index-1]._up_stair_map.copy()
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_up_stair_map.astype(np.uint8), connectivity=8)
                            closest_label = -1
                            min_distance = float('inf')
                            for i in range(1, num_labels):  
                                centroid_px = centroids[i]  
                                centroid = self._obstacle_map._px_to_xy(np.atleast_2d(centroid_px))
                                distance = np.linalg.norm(self._stair_frontier - centroid)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_label = i
                            if closest_label != -1:
                                ori_up_stair_map[labels != closest_label] = 0 
                            
                            self._obstacle_map_list[self._cur_floor_index]._down_stair_map = ori_up_stair_map
                            self._obstacle_map_list[self._cur_floor_index]._down_stair_start = self._obstacle_map_list[self._cur_floor_index - 1]._up_stair_end.copy()
                            self._obstacle_map_list[self._cur_floor_index]._down_stair_end = self._obstacle_map_list[self._cur_floor_index - 1]._up_stair_start.copy()
                            self._obstacle_map_list[self._cur_floor_index]._down_stair_frontiers = self._obstacle_map_list[self._cur_floor_index - 1]._up_stair_frontiers.copy()

                        elif self._climb_stair_flag == 2 and self._obstacle_map_list[self._cur_floor_index-1]._done_initializing == False:

                            self._done_initializing = False
                            self._initialize_step = 0 
                            self._obstacle_map._explored_down_stair = True
                            self._cur_floor_index -= 1 
                            self._obstacle_map = self._obstacle_map_list[self._cur_floor_index]
                            ori_down_stair_map = self._obstacle_map_list[self._cur_floor_index+1]._down_stair_map.copy()                            
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_down_stair_map.astype(np.uint8), connectivity=8)
                            closest_label = -1
                            min_distance = float('inf')
                            for i in range(1, num_labels):  
                                centroid_px = centroids[i]  
                                centroid = self._obstacle_map._px_to_xy(np.atleast_2d(centroid_px))
                                distance = np.linalg.norm(self._stair_frontier - centroid)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_label = i
                            if closest_label != -1:
                                ori_down_stair_map[labels != closest_label] = 0                             
                            self._obstacle_map_list[self._cur_floor_index]._up_stair_map = ori_down_stair_map
                            self._obstacle_map_list[self._cur_floor_index]._up_stair_start = self._obstacle_map_list[self._cur_floor_index + 1]._down_stair_end.copy()
                            self._obstacle_map_list[self._cur_floor_index]._up_stair_end = self._obstacle_map_list[self._cur_floor_index + 1]._down_stair_start.copy()
                            self._obstacle_map_list[self._cur_floor_index]._up_stair_frontiers = self._obstacle_map_list[self._cur_floor_index + 1]._down_stair_frontiers.copy()
                            
                        self.mode = "climb_stair_initialize"
                        if self._pitch_angle > 0: 
                            self._pitch_angle -= self._pitch_angle_offset
                            pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                        elif self._pitch_angle < 0:
                            self._pitch_angle += self._pitch_angle_offset
                            pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
                        else:  
                            self._obstacle_map._done_initializing = False 
                            self._initialize_step = 0
                            pointnav_action = self._initialize()
                        self._obstacle_map._climb_stair_paused_step = 0
                        self._climb_stair_over = True
                        self._climb_stair_flag = 0
                        self._reach_stair = False
                        self._reach_stair_centroid = False
                        self._stair_dilate_flag = False
            else:
                if self._obstacle_map._look_for_downstair_flag == True:
                    self.mode = "look_for_downstair"
                    pointnav_action = self._look_for_downstair(observations, masks)
                elif self._climb_stair_flag == 1 and self._pitch_angle == 0 and np.sum(self._obstacle_map._up_stair_map)>0:
                    up_stair_points = np.argwhere(self._obstacle_map._up_stair_map)
                    robot_xy = self._observations_cache["robot_xy"]
                    robot_xy_2d = np.atleast_2d(robot_xy) 
                    robot_px = self._obstacle_map._xy_to_px(robot_xy_2d)
                    distances = np.abs(up_stair_points[:, 0] - robot_px[0][0]) + np.abs(up_stair_points[:, 1] - robot_px[0][1])
                    min_dis_to_upstair = np.min(distances)
                    print(f"min_dis_to_upstair: {min_dis_to_upstair}")
                    if min_dis_to_upstair <= 2.0 * self._obstacle_map.pixels_per_meter and check_stairs_in_upper_50_percent(self.red_semantic_pred == STAIR_CLASS_ID):
                        self._pitch_angle += self._pitch_angle_offset
                        self.mode = "look_up"
                        pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
                    else:
                        self.mode = "get_close_to_stair"
                        pointnav_action = self._get_close_to_stair(observations, masks)
                elif self._climb_stair_flag == 2 and self._pitch_angle == 0 and np.sum(self._obstacle_map._down_stair_map)>0 :
                    down_stair_points = np.argwhere(self._obstacle_map._down_stair_map)
                    robot_xy = self._observations_cache["robot_xy"]
                    robot_xy_2d = np.atleast_2d(robot_xy) 
                    robot_px = self._obstacle_map._xy_to_px(robot_xy_2d)
                    distances = np.abs(down_stair_points[:, 0] - robot_px[0][0]) + np.abs(down_stair_points[:, 1] - robot_px[0][1])
                    min_dis_to_downstair = np.min(distances)
                    print(f"min_dis_to_downstair: {min_dis_to_downstair}")
                    if min_dis_to_downstair <= 2.0 * self._obstacle_map.pixels_per_meter:
                        self._pitch_angle -= self._pitch_angle_offset
                        self.mode = "look_down"
                        pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                    else:
                        self.mode = "get_close_to_stair"
                        pointnav_action = self._get_close_to_stair(observations, masks)
                else:
                    self.mode = "get_close_to_stair"
                    pointnav_action = self._get_close_to_stair(observations, masks)

        else:
            if self._pitch_angle > 0: 
                self.mode = "look_down_back"
                self._pitch_angle -= self._pitch_angle_offset
                pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
            elif self._pitch_angle < 0 and self._obstacle_map._look_for_downstair_flag == False:
                self.mode = "look_up_back"
                self._pitch_angle += self._pitch_angle_offset
                pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
            elif not self._done_initializing:
                self.mode = "initialize"
                pointnav_action = self._initialize()
            elif goal is not None:
                self.mode = "navigate"
                self._try_to_navigate_step += 1
                robot_xy = self._observations_cache["robot_xy"]
                heading = self._observations_cache["robot_heading"]
                cur_dis_to_goal = np.linalg.norm(goal[:2] - robot_xy[:2])
                rho, theta = rho_theta(robot_xy, heading, goal)
                print("origin theta: ",theta)
                print("cur_dis_to_goal: ",cur_dis_to_goal)
                theta = (theta + np.pi) % (2*np.pi) - np.pi
                if cur_dis_to_goal < 1.0: 
                    if cur_dis_to_goal <= 0.6 or np.abs(cur_dis_to_goal - self.min_distance_xy) < 0.1:
                        self._called_stop = True
                        pointnav_action = self._stop_action
                    else:
                        self.min_distance_xy = cur_dis_to_goal
                        pointnav_action = TorchActionIDs_plook.MOVE_FORWARD
                else:
                    self.min_distance_xy = cur_dis_to_goal        
                    pointnav_action = self._pointnav(goal[:2], stop=True)
            else:
                if self._obstacle_map._look_for_downstair_flag == True:
                    self.mode = "look_for_downstair"
                    pointnav_action = self._look_for_downstair(observations, masks)
                else:
                    explore_action = self._explore(observations)
                    self.mode = "explore"
                    pointnav_action = explore_action
        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {self.mode} | Action: {action_numpy}")
        
        if self._climb_stair_over == False:
            print(f"Reach_stair_centroid: {self._reach_stair_centroid}")

        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1
        self._obstacle_map._floor_num_steps += 1
        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)
        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)
        )
        detections.filter_by_class(target_classes)
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and detections.num_detections == 0:
            detections = self._object_detector.predict(img, caption=target_classes[0])
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)
        
        stairs_caption = "stair"
        stair_detections = self._object_detector.predict(img, caption=stairs_caption)
        stair_detections.filter_by_class([stairs_caption])
        stair_detections.filter_by_conf(self._non_coco_threshold)

        return detections, stair_detections

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections, stair_detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        self._stair_masks = np.zeros((height, width), dtype=bool)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        
        object_idx = None
        close_to_gloal = False
        if self._object_map.has_object(self._target_object) and (len(detections.logits) > 0):
            print("check has the target object for navigation")
            robot_xy = self._observations_cache["robot_xy"]
            goal_point = self._object_map.get_best_object(self._target_object, robot_xy)
            print("np.linalg.norm(goal_point[:2] - robot_xy[:2]): ",np.linalg.norm(goal_point[:2] - robot_xy[:2]))
            if np.linalg.norm(goal_point[:2] - robot_xy[:2]) < 1.5:
                close_to_gloal = True
            print("not close_to_gloal: ",not close_to_gloal)
            print("self.double_detect: ",self.double_detect)
        if (4 > len(detections.logits) > 1) and (not close_to_gloal) and (self.double_detect):
            print("compare the detections")
            color_list = [(255, 0, 0),(0, 255, 0),(0, 0, 255)]
            color_string_list = ['red','green','blue']
            mask_list = []
            bbox_denorm_list = []
            annotated_rgb_multi = rgb.copy()
            for idx in range(len(detections.logits)):
                bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
                object_mask = self._mobile_sam.segment_bbox(annotated_rgb_multi, bbox_denorm.tolist())
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb_multi = cv2.drawContours(annotated_rgb_multi, contours, -1, color_list[idx], 2)
            answer_color = self.openai_client.detection_choise(annotated_rgb_multi, detections.phrases[idx], color_string_list[:len(detections.logits)])
            print("answer_color: ",answer_color)
            annotated_rgb_multi = cv2.cvtColor(annotated_rgb_multi, cv2.COLOR_RGB2BGR)
            height, width, _ = annotated_rgb_multi.shape
            text = answer_color
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(width, height) / 500  
            font_thickness = 2
            color = (0, 0, 0)  
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2  
            cv2.putText(annotated_rgb_multi, text, (text_x, text_y), font, font_scale, color, font_thickness)
            cv2.imwrite(f'{self.result_path}/VLM/annotated_rgb_{self.counter}.jpg', annotated_rgb_multi)
            if "red" in answer_color.lower():
                object_idx = 0
                print("red: ",object_idx)
            elif "green" in answer_color.lower():
                object_idx = 1
                print("green: ",object_idx)
            elif "blue" in answer_color.lower():
                object_idx = 2
                print("blue: ",object_idx)
            else:
                object_idx = 0
                print("else: ",object_idx)
            self.fist_detect = True
            self._object_map.reset()
            self.double_detect = False
            
        if object_idx is None:
            for idx in range(len(detections.logits)):
                bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
                object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())
                same_detected_object = True
                if self._object_map.has_object(self._target_object):
                    local_cloud = self._object_map._extract_object_cloud(depth, object_mask, min_depth, max_depth, fx, fy)
                    global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
                    if global_cloud.shape[0] == 0:
                        continue
                    detected_cloud = self._object_map.get_target_cloud(self._target_object)
                    mean_global = np.mean(global_cloud[:, :2], axis=0)
                    mean_detected = np.mean(detected_cloud[:, :2], axis=0)
                    diff = np.linalg.norm(mean_global - mean_detected)
                    if diff > 1.5:
                        same_detected_object = False
                if (self._use_vqa and self.fist_detect) or (not same_detected_object):
                    contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                    answer = self.openai_client.detection_refinement(annotated_rgb, detections.phrases[idx])
                    print("answer nomal: ",answer)
                    answer_yn = re.search(r'(?:\*\*Answer:\*\*|Answer:)\s*(yes|no)', answer, re.IGNORECASE)
                    if not answer_yn:
                        raise ValueError("找不到格式为 'answer: yes/no' 的行")
                    determine = answer_yn.group(1).lower()
                    if "yes" in determine.lower():
                        self.fist_detect = False
                        
                    else:
                        print("ignore the detection")
                        self.double_detect = True
                        continue

                self._object_masks[object_mask > 0] = 1
                print("update the object map")
                if not self.too_offset(object_mask):
                    self._object_map.update_map(
                        self._target_object,
                        depth,
                        object_mask,
                        tf_camera_to_episodic,
                        min_depth,
                        max_depth,
                        fx,
                        fy,
                    )
                    print("have the target object? ",self._object_map.has_object(self._target_object))
        else:
            for idx in [object_idx]:
                print("object_idx: ",idx," ",color_string_list[idx])
                bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
                object_mask = self._mobile_sam.segment_bbox(rgb.copy(), bbox_denorm.tolist())

                if self._use_vqa and self.fist_detect:
                    contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                    answer = self.openai_client.detection_refinement(annotated_rgb, detections.phrases[idx])
                    print("answer choose: ",answer)
                    answer_yn = re.search(r'(?:\*\*Answer:\*\*|Answer:)\s*(yes|no)', answer, re.IGNORECASE)
                    if not answer_yn:
                        raise ValueError("找不到格式为 'answer: yes/no' 的行")
                    determine = answer_yn.group(1).lower()
                    if not ("yes" in determine.lower()):
                        print("ignore the detection")
                        self.double_detect = True
                        continue
                    else:
                        self.fist_detect = False

                self._object_masks[object_mask > 0] = 1
                if not self.too_offset(object_mask):
                    print("update the object_map")
                    self._object_map.reset()
                    self._object_map.update_map(
                        self._target_object,
                        depth,
                        object_mask,
                        tf_camera_to_episodic,
                        min_depth,
                        max_depth,
                        fx,
                        fy,
                    )
                    print("have the target object? ",self._object_map.has_object(self._target_object))


        for idx in range(len(stair_detections.logits)):
            stair_bbox_denorm = stair_detections.boxes[idx] * np.array([width, height, width, height])
            stair_mask = self._mobile_sam.segment_bbox(rgb, stair_bbox_denorm.tolist())
            self._stair_masks[stair_mask > 0] = 1  
        
        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError
    
    def too_offset(self,mask: np.ndarray) -> bool:
        """
        This will return true if the entire bounding rectangle of the mask is either on the
        left or right third of the mask. This is used to determine if the object is too far
        to the side of the image to be a reliable detection.

        Args:
            mask (numpy array): A 2D numpy array of 0s and 1s representing the mask of the
                object.
        Returns:
            bool: True if the object is too offset, False otherwise.
        """
        x, y, w, h = cv2.boundingRect(mask)

        fourth = mask.shape[1] // 4

        if x + w <= fourth:
            return x <= int(0.05 * mask.shape[1])
        elif x >= 3 * fourth:
            return x + w >= int(0.95 * mask.shape[1])
        else:
            return False
    
    def _update_obstacle_map(self) -> None: 
        if self._compute_frontiers:
            if self._climb_stair_over == False:
                if self._climb_stair_flag == 1: 
                    self._temp_stair_map= self._obstacle_map._up_stair_map
                elif self._climb_stair_flag == 2: 
                    self._temp_stair_map = self._obstacle_map._down_stair_map
                if self._stair_dilate_flag == False:
                    self._temp_stair_map = cv2.dilate(
                    self._temp_stair_map.astype(np.uint8),
                    (7,7), 
                    iterations=1,
                    )
                    self._stair_dilate_flag = True
                robot_xy = self._observations_cache["robot_xy"]
                robot_xy_2d = np.atleast_2d(robot_xy) 
                robot_px = self._obstacle_map._xy_to_px(robot_xy_2d)
                x, y = robot_px[0, 0], robot_px[0, 1]
                if self._reach_stair == False: 
                    already_reach_stair, reach_yx = self.is_robot_in_stair_map_fast(robot_px, self._temp_stair_map) 
                    if already_reach_stair:
                        self._reach_stair = True
                        self._get_close_to_stair_step = 0
                        if self._climb_stair_flag == 1: 
                            self._obstacle_map._up_stair_start = robot_px[0].copy()
                        elif self._climb_stair_flag == 2: 
                            self._obstacle_map._down_stair_start = robot_px[0].copy()
                if self._reach_stair == True and self._reach_stair_centroid == False:
                    if self._stair_frontier is not None and np.linalg.norm(self._stair_frontier - robot_xy_2d) <= 0.3:
                        self._reach_stair_centroid = True
                if self._reach_stair_centroid == True:

                    if self.is_robot_in_stair_map_fast(robot_px, self._temp_stair_map)[0]:
                        pass
                    elif self._obstacle_map._climb_stair_paused_step >= 30:
                        self._obstacle_map._climb_stair_paused_step = 0
                        self._last_carrot_xy = []
                        self._last_carrot_px = []
                        self._reach_stair = False
                        self._reach_stair_centroid = False
                        self._stair_dilate_flag = False
                        self._climb_stair_over = True
                        self._obstacle_map._disabled_frontiers.add(tuple(self._stair_frontier[0]))
                        print(f"Frontier {self._stair_frontier} is disabled due to no movement.")
                        if  self._climb_stair_flag == 1:
                            self._obstacle_map._disabled_stair_map[self._obstacle_map._up_stair_map == 1] = 1
                            self._obstacle_map._up_stair_map.fill(0)
                            self._obstacle_map._has_up_stair = False
                            del self._obstacle_map_list[self._cur_floor_index + 1]
                            self.floor_num -= 1
                        elif  self._climb_stair_flag == 2:
                            self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
                            self._obstacle_map._down_stair_frontiers.fill(0)
                            self._obstacle_map._has_down_stair = False
                            self._obstacle_map._look_for_downstair_flag = False
                            del self._obstacle_map_list[self._cur_floor_index - 1]
                            self.floor_num -= 1
                            self._cur_floor_index -= 1 
                        self._climb_stair_flag = 0
                        self._stair_dilate_flag = False
                    else:
                        self._climb_stair_over = True
                        self._reach_stair = False
                        self._reach_stair_centroid = False
                        self._stair_dilate_flag = False
                        if self._climb_stair_flag == 1: 
                            self._obstacle_map._up_stair_end = robot_px[0].copy()
                            if self._obstacle_map_list[self._cur_floor_index+1]._done_initializing == False:
                                self._done_initializing = False
                                self._initialize_step = 0
                                self._obstacle_map._explored_up_stair = True
                                self._obstacle_map_list[self._cur_floor_index+1]._explored_down_stair = True
                                self._cur_floor_index += 1
                                self._obstacle_map = self._obstacle_map_list[self._cur_floor_index]
                                ori_up_stair_map = self._obstacle_map_list[self._cur_floor_index-1]._up_stair_map.copy()
                                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_up_stair_map.astype(np.uint8), connectivity=8)
                                closest_label = -1
                                min_distance = float('inf')
                                for i in range(1, num_labels):  
                                    centroid_px = centroids[i]  
                                    centroid = self._obstacle_map._px_to_xy(np.atleast_2d(centroid_px))
                                    distance = np.abs(self._obstacle_map_list[self._cur_floor_index-1]._up_stair_frontiers[0][0] - centroid[0][0]) + np.abs(self._obstacle_map_list[self._cur_floor_index-1]._up_stair_frontiers[0][1] - centroid[0][1])
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_label = i
                                if closest_label != -1:
                                    ori_up_stair_map[labels != closest_label] = 0 
                                self._obstacle_map_list[self._cur_floor_index]._down_stair_map = ori_up_stair_map
                                self._obstacle_map_list[self._cur_floor_index]._down_stair_start = self._obstacle_map_list[self._cur_floor_index - 1]._up_stair_end.copy()
                                self._obstacle_map_list[self._cur_floor_index]._down_stair_end = self._obstacle_map_list[self._cur_floor_index - 1]._up_stair_start.copy()
                                self._obstacle_map_list[self._cur_floor_index]._down_stair_frontiers = self._obstacle_map_list[self._cur_floor_index - 1]._up_stair_frontiers.copy()
                            else:
                                self._cur_floor_index += 1
                                self._obstacle_map = self._obstacle_map_list[self._cur_floor_index]
                        elif self._climb_stair_flag == 2:
                            self._obstacle_map._down_stair_end = robot_px[0].copy()
                            if self._obstacle_map_list[self._cur_floor_index-1]._done_initializing == False:
                                self._done_initializing = False
                                self._initialize_step = 0
                                self._obstacle_map._explored_down_stair = True
                                self._obstacle_map_list[self._cur_floor_index-1]._explored_up_stair = True
                                self._cur_floor_index -= 1 
                                self._obstacle_map = self._obstacle_map_list[self._cur_floor_index]
                                ori_down_stair_map = self._obstacle_map_list[self._cur_floor_index+1]._down_stair_map.copy()
                                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_down_stair_map.astype(np.uint8), connectivity=8)
                                closest_label = -1
                                min_distance = float('inf')
                                for i in range(1, num_labels):  
                                    centroid_px = centroids[i]  
                                    centroid = self._obstacle_map._px_to_xy(np.atleast_2d(centroid_px))
                                    distance = np.abs(self._obstacle_map_list[self._cur_floor_index+1]._down_stair_frontiers[0][0] - centroid[0][0]) + np.abs(self._obstacle_map_list[self._cur_floor_index+1]._down_stair_frontiers[0][1] - centroid[0][1])
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_label = i
                                if closest_label != -1:
                                    ori_down_stair_map[labels != closest_label] = 0 
                                self._obstacle_map_list[self._cur_floor_index]._up_stair_map = ori_down_stair_map
                                self._obstacle_map_list[self._cur_floor_index]._up_stair_start = self._obstacle_map_list[self._cur_floor_index + 1]._down_stair_end.copy()
                                self._obstacle_map_list[self._cur_floor_index]._up_stair_end = self._obstacle_map_list[self._cur_floor_index + 1]._down_stair_start.copy()
                                self._obstacle_map_list[self._cur_floor_index]._up_stair_frontiers = self._obstacle_map_list[self._cur_floor_index + 1]._down_stair_frontiers.copy()
                            else:
                                self._cur_floor_index -= 1
                                self._obstacle_map = self._obstacle_map_list[self._cur_floor_index]
                        self._climb_stair_flag = 0
                        self._has_climbed_once = True
                        self._obstacle_map._climb_stair_paused_step = 0
                        self._last_carrot_xy = []
                        self._last_carrot_px = [] 
                        
                        print("climb stair success!!!!")
            self._obstacle_map.update_map_with_stair(
                self._observations_cache["object_map_rgbd"][0][1],
                self._observations_cache["object_map_rgbd"][0][2],
                self._min_depth, self._max_depth, self._fx, self._fy,
                self._camera_fov,  self._stair_masks,
                self.red_semantic_pred, self._pitch_angle,
                self._climb_stair_over, self._reach_stair,
                self._climb_stair_flag
            )
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(self._observations_cache["robot_xy"], self._observations_cache["robot_heading"])

        else:
            if "frontier_sensor" in self._observations_cache:
                frontiers = self._observations_cache["frontier_sensor"].cpu().numpy()
            else:
                frontiers = np.array([])
        
        self._observations_cache["frontier_sensor"] = frontiers
        if self._obstacle_map._has_up_stair and self._cur_floor_index + 1 >= len(self._obstacle_map_list):
            self._obstacle_map_list.append(ObstacleMapUpdater(
                min_height=self.min_obstacle_height,
                max_height=self.max_obstacle_height,
                area_thresh=self.obstacle_map_area_threshold,
                agent_radius=self.agent_radius,
                hole_area_thresh=self.hole_area_thresh,
                size=1000,
            ))
        if self._obstacle_map._has_down_stair and self._cur_floor_index == 0:
            self._obstacle_map_list.insert(0, ObstacleMapUpdater(
                min_height=self.min_obstacle_height,
                max_height=self.max_obstacle_height,
                area_thresh=self.obstacle_map_area_threshold,
                agent_radius=self.agent_radius,
                hole_area_thresh=self.hole_area_thresh,
                size=1000,
            ))
            self._cur_floor_index += 1 
        self.floor_num = len(self._obstacle_map_list)
        self._obstacle_map.project_frontiers_to_rgb_hush(self._observations_cache["object_map_rgbd"][0][0])
   




    def _update_stair_endpoints(self, robot_px: np.ndarray) -> None:
        if self._climb_stair_flag == 1:
            self._update_up_stair_endpoints(robot_px)
        elif self._climb_stair_flag == 2:
            self._update_down_stair_endpoints(robot_px)
       
    def _update_down_stair_endpoints(self, robot_px: np.ndarray) -> None:
        self._obstacle_map._down_stair_end = robot_px[0].copy()
        if not self._obstacle_map_list[self._cur_floor_index - 1]._done_initializing:
            self._initialize_new_floor(self._cur_floor_index - 1, "down")
            self._downstair_to_upstair()
        else:
            self._cur_floor_index -= 1
            self._update_current_floor_maps()
   
    def _update_up_stair_endpoints(self, robot_px: np.ndarray) -> None:
        self._obstacle_map._up_stair_end = robot_px[0].copy()
        if not self._obstacle_map_list[self._cur_floor_index + 1]._done_initializing:
            self._initialize_new_floor(self._cur_floor_index + 1, "up")
            self._upstair_to_downstair()
        else:
            self._cur_floor_index += 1
            self._update_current_floor_maps() 
   
    def _update_current_floor_maps(self) -> None:
        self._obstacle_map = self._obstacle_map_list[self._cur_floor_index]
   
    def _initialize_new_floor(self, floor_index: int, stair_type: str) -> None:
        self._done_initializing = False
        self._initialize_step = 0
        if stair_type == "up":
            self._obstacle_map._explored_up_stair = True
            self._obstacle_map_list[floor_index]._explored_down_stair = True
        else:
            self._obstacle_map._explored_down_stair = True
            self._obstacle_map_list[floor_index]._explored_up_stair = True

        self._cur_floor_index = floor_index
        self._update_current_floor_maps()
        
    def _downstair_to_upstair(self) -> None:
        ori_down_stair_map = self._obstacle_map_list[self._cur_floor_index+1]._down_stair_map.copy()                   
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_down_stair_map.astype(np.uint8), connectivity=8)
        closest_label = -1
        min_distance = float('inf')
        for i in range(1, num_labels):  
            centroid_px = centroids[i]  
            centroid = self._obstacle_map._px_to_xy(np.atleast_2d(centroid_px))
            distance = np.abs(self._obstacle_map_list[self._cur_floor_index+1]._down_stair_frontiers[0][0] - centroid[0][0]) + np.abs(self._obstacle_map_list[self._cur_floor_index+1]._down_stair_frontiers[0][1] - centroid[0][1])
            if distance < min_distance:
                min_distance = distance
                closest_label = i
        if closest_label != -1:
            ori_down_stair_map[labels != closest_label] = 0 
        
        self._obstacle_map_list[self._cur_floor_index]._up_stair_map = ori_down_stair_map
        self._obstacle_map_list[self._cur_floor_index]._up_stair_start = self._obstacle_map_list[self._cur_floor_index + 1]._down_stair_end.copy()
        self._obstacle_map_list[self._cur_floor_index]._up_stair_end = self._obstacle_map_list[self._cur_floor_index + 1]._down_stair_start.copy()
        self._obstacle_map_list[self._cur_floor_index]._up_stair_frontiers = self._obstacle_map_list[self._cur_floor_index + 1]._down_stair_frontiers.copy()
        
    def _upstair_to_downstair(self) -> None:
        ori_up_stair_map = self._obstacle_map_list[self._cur_floor_index-1]._up_stair_map.copy()                    
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_up_stair_map.astype(np.uint8), connectivity=8)
        closest_label = -1
        min_distance = float('inf')
        for i in range(1, num_labels):  
            centroid_px = centroids[i]  
            centroid = self._obstacle_map._px_to_xy(np.atleast_2d(centroid_px))
            distance = np.abs(self._obstacle_map_list[self._cur_floor_index-1]._up_stair_frontiers[0][0] - centroid[0][0]) + np.abs(self._obstacle_map_list[self._cur_floor_index-1]._up_stair_frontiers[0][1] - centroid[0][1])
            if distance < min_distance:
                min_distance = distance
                closest_label = i
        if closest_label != -1:
            ori_up_stair_map[labels != closest_label] = 0 
        
        self._obstacle_map_list[self._cur_floor_index]._down_stair_map = ori_up_stair_map
        self._obstacle_map_list[self._cur_floor_index]._down_stair_start = self._obstacle_map_list[self._cur_floor_index - 1]._up_stair_end.copy()
        self._obstacle_map_list[self._cur_floor_index]._down_stair_end = self._obstacle_map_list[self._cur_floor_index - 1]._up_stair_start.copy()
        self._obstacle_map_list[self._cur_floor_index]._down_stair_frontiers = self._obstacle_map_list[self._cur_floor_index - 1]._up_stair_frontiers.copy()
    
    def _end_of_climbing_stair(self) -> None:
        self._obstacle_map._climb_stair_paused_step = 0
        self._last_carrot_xy = []
        self._last_carrot_px = []
        self._reach_stair = False
        self._reach_stair_centroid = False
        self._stair_dilate_flag = False
        self._climb_stair_over = True
        self._climb_stair_flag = 0
   
    def _reset_stair_state(self) -> None:

        self._obstacle_map._disabled_frontiers.add(tuple(self._stair_frontier[0]))
        print(f"Frontier {self._stair_frontier} is disabled due to no movement.")

        if self._climb_stair_flag == 1:
            self._disable_up_stair()
        elif self._climb_stair_flag == 2:
            self._disable_down_stair()

        self._climb_stair_flag = 0

    def _disable_up_stair(self) -> None:
        self._obstacle_map._disabled_stair_map[self._obstacle_map._up_stair_map == 1] = 1
        self._obstacle_map._up_stair_map.fill(0)
        self._obstacle_map._has_up_stair = False
        self._remove_floor_from_map_list(self._cur_floor_index + 1)
        self.floor_num -= 1
        
    def _disable_down_stair(self) -> None:
        self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
        self._obstacle_map._down_stair_frontiers.fill(0)
        self._obstacle_map._has_down_stair = False
        self._obstacle_map._look_for_downstair_flag = False
        self._remove_floor_from_map_list(self._cur_floor_index - 1)
        self.floor_num -= 1
        self._cur_floor_index -= 1
    
    def _remove_floor_from_map_list(self, floor_index: int) -> None:
        del self._obstacle_map_list[floor_index]
        
    def is_robot_in_stair_map_fast(self, robot_px:np.ndarray, stair_map: np.ndarray):
        x, y = robot_px[0, 0], robot_px[0, 1]


        radius_px = self.agent_radius * self._obstacle_map.pixels_per_meter
        rows, cols = stair_map.shape
        x_min = max(0, int(x - radius_px))
        x_max = min(cols - 1, int(x + radius_px))
        y_min = max(0, int(y - radius_px))
        y_max = min(rows - 1, int(y + radius_px))
        sub_matrix = stair_map[y_min:y_max + 1, x_min:x_max + 1]
        y_indices, x_indices = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
        mask = (y_indices - y) ** 2 + (x_indices - x) ** 2 <= radius_px ** 2
        if np.any(sub_matrix[mask]):  
            true_coords_in_sub_matrix = np.column_stack(np.where(sub_matrix))
            true_coords_filtered = true_coords_in_sub_matrix[mask[true_coords_in_sub_matrix[:, 0], true_coords_in_sub_matrix[:, 1]]]
            true_coords_in_stair_map = true_coords_filtered + [y_min, x_min]
            
            return True, true_coords_in_stair_map
        else:
            return False, None

    def _look_for_downstair(self, observations: Union[Dict[str, Tensor], "TensorDict"], masks: Tensor) -> Tensor:
        self.look_steps +=1
        if self._pitch_angle >= 0:
            self._pitch_angle -= self._pitch_angle_offset
            pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
        else:
            robot_xy = self._observations_cache["robot_xy"]
            robot_xy_2d = np.atleast_2d(robot_xy) 
            dis_to_potential_stair = np.linalg.norm(self._obstacle_map._potential_stair_centroid - robot_xy_2d)
            if dis_to_potential_stair > 0.2 and self.look_steps < 30:
                pointnav_action = self._pointnav(self._obstacle_map._potential_stair_centroid[0], stop=True) 
                if pointnav_action.item() == 0:
                    print("Might false recognize down stairs, change to other mode.")
                    self._obstacle_map._disabled_frontiers.add(tuple(self._obstacle_map._potential_stair_centroid[0]))
                    print(f"Frontier {self._obstacle_map._potential_stair_centroid[0]} is disabled due to no movement.")
                    self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
                    self._obstacle_map._down_stair_map.fill(0)
                    self._obstacle_map._has_down_stair = False
                    self._pitch_angle += self._pitch_angle_offset
                    self._obstacle_map._look_for_downstair_flag = False
                    pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
                    self.look_steps = 0
            else:
                print("Might false recognize down stairs, change to other mode.")
                self._obstacle_map._disabled_frontiers.add(tuple(self._obstacle_map._potential_stair_centroid[0]))
                print(f"Frontier {self._obstacle_map._potential_stair_centroid[0]} is disabled due to no movement.")
                self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
                self._obstacle_map._down_stair_map.fill(0)
                self._obstacle_map._has_down_stair = False
                self._pitch_angle += self._pitch_angle_offset
                self._obstacle_map._look_for_downstair_flag = False
                self.look_steps = 0
                pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
        return pointnav_action
    
    def _get_close_to_stair(self, observations: "TensorDict", ori_masks: Tensor) -> Tensor:

        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        if  self._climb_stair_flag == 1:
            self._stair_frontier = self._obstacle_map._up_stair_frontiers
        elif  self._climb_stair_flag == 2:
            self._stair_frontier = self._obstacle_map._down_stair_frontiers
        if self._stair_frontier.shape[0] == 0:
            action = self._explore(observations)
            return action
        if np.array_equal(self._last_frontier, self._stair_frontier[0]):
            if self._frontier_stick_step == 0:
                self._last_frontier_distance = np.linalg.norm(self._stair_frontier - robot_xy)
                self._frontier_stick_step += 1
            else:
                current_distance = np.linalg.norm(self._stair_frontier - robot_xy)
                print(f"Distance Change: {np.abs(self._last_frontier_distance - current_distance)} and Stick Step {self._frontier_stick_step}")
                if np.abs(self._last_frontier_distance - current_distance) > 0.3:
                    self._frontier_stick_step = 0
                    self._last_frontier_distance = current_distance 
                else:
                    if self._frontier_stick_step >= 50:
                        self._obstacle_map._disabled_frontiers.add(tuple(self._stair_frontier[0]))
                        print(f"Frontier {self._stair_frontier} is disabled due to no movement.")
                        if  self._climb_stair_flag == 1:
                            self._obstacle_map._disabled_stair_map[self._obstacle_map._up_stair_map == 1] = 1
                            self._obstacle_map._up_stair_map.fill(0)
                            self._climb_stair_flag = 0
                            self._stair_dilate_flag = False
                            self._climb_stair_over = True
                            self._obstacle_map._has_up_stair = False
                            self._obstacle_map._look_for_downstair_flag = False
                        elif  self._climb_stair_flag == 2:
                            self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
                            self._obstacle_map._down_stair_frontiers.fill(0)
                            self._climb_stair_flag = 0
                            self._stair_dilate_flag = False
                            self._climb_stair_over = True
                            self._obstacle_map._has_down_stair = False
                        action = self._explore(observations) 
                        return action
                    elif current_distance < 1.0:
                        self._frontier_stick_step += 1
        else:
            self._frontier_stick_step = 0
            self._last_frontier_distance = 0
        self._last_frontier = self._stair_frontier[0]
        rho, theta = rho_theta(robot_xy, heading, self._stair_frontier[0])
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True) 
        if action.item() == 0:
            self._obstacle_map._disabled_frontiers.add(tuple(self._stair_frontier[0]))
            print(f"Frontier {self._stair_frontier} is disabled due to no movement.")
            if  self._climb_stair_flag == 1:
                self._obstacle_map._disabled_stair_map[self._obstacle_map._up_stair_map == 1] = 1
                self._obstacle_map._up_stair_map.fill(0)
                self._obstacle_map._up_stair_frontiers = np.array([])
                self._climb_stair_over = True
                self._obstacle_map._has_up_stair = False
                self._obstacle_map._look_for_downstair_flag = False
            elif  self._climb_stair_flag == 2:
                self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
                self._obstacle_map._down_stair_map.fill(0)
                self._obstacle_map._down_stair_frontiers = np.array([])
                self._climb_stair_over = True
                self._obstacle_map._has_down_stair = False
                self._obstacle_map._look_for_downstair_flag = False
            self._climb_stair_flag = 0
            self._stair_dilate_flag = False
            action = self._explore(observations) 
        return action

    def _climb_stair(self, observations: "TensorDict", ori_masks: Tensor) -> Tensor:
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        robot_xy = self._observations_cache["robot_xy"]
        robot_xy_2d = np.atleast_2d(robot_xy) 
        robot_px = self._obstacle_map._xy_to_px(robot_xy_2d)
        heading = self._observations_cache["robot_heading"]  

        if  self._climb_stair_flag == 1:
            self._stair_frontier = self._obstacle_map._up_stair_frontiers
        elif  self._climb_stair_flag == 2:
            self._stair_frontier = self._obstacle_map._down_stair_frontiers
        current_distance = np.linalg.norm(self._stair_frontier - robot_xy)
        print(f"Climb Stair -- Distance Change: {np.abs(self._last_frontier_distance - current_distance)} and Stick Step {self._obstacle_map._climb_stair_paused_step}")
        if np.abs(self._last_frontier_distance - current_distance) > 0.2:
            self._obstacle_map._climb_stair_paused_step = 0
            self._last_frontier_distance = current_distance  
        else:
            self._obstacle_map._climb_stair_paused_step += 1
        
        if self._obstacle_map._climb_stair_paused_step > 15:
            self._obstacle_map._disable_end = True
        if self._reach_stair_centroid == False:
            stair_frontiers = self._stair_frontier[0]
            rho, theta = rho_theta(robot_xy, heading, stair_frontiers)
            rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
            obs_pointnav = {
                "depth": image_resize(
                    self._observations_cache["nav_depth"],
                    (self._depth_image_shape[0], self._depth_image_shape[1]),
                    channels_last=True,
                    interpolation_mode="area",
                ),
                "pointgoal_with_gps_compass": rho_theta_tensor,
            }
            self._policy_info["rho_theta"] = np.array([rho, theta])
            action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            if action.item() == 0:
                self._reach_stair_centroid = True
                print("Might close, change to move forward.") 
                action[0] = 1
            return action

        elif self._climb_stair_flag == 2 and self._pitch_angle < -30: 
            self._pitch_angle += self._pitch_angle_offset
            print("Look up a little for downstair!")
            action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
            return action
        
        else:

            distance = 0.8 
            depth_map = self._observations_cache["nav_depth"].squeeze(0).cpu().numpy()
            max_value = np.max(depth_map)
            max_indices = np.argwhere(depth_map == max_value)  
            center_point = np.mean(max_indices, axis=0).astype(int)
            v, u = center_point[0], center_point[1] 
            normalized_u = (u - self._cx) / self._cx 
            normalized_u = np.clip(normalized_u, -1, 1)
            angle_offset = normalized_u * (self._camera_fov / 2)
            target_heading = heading - angle_offset 
            target_heading = target_heading % (2 * np.pi)
            x_target = robot_xy[0] + distance * np.cos(target_heading)
            y_target = robot_xy[1] + distance * np.sin(target_heading)
            target_point = np.array([x_target, y_target])
            target_point_2d = np.atleast_2d(target_point) 
            temp_target_px = self._obstacle_map._xy_to_px(target_point_2d) 
            if  self._climb_stair_flag == 1:
                this_stair_end = self._obstacle_map._up_stair_end
            elif  self._climb_stair_flag == 2:
                this_stair_end = self._obstacle_map._down_stair_end
            if len(self._last_carrot_xy) == 0 or len(this_stair_end) == 0: 
                self._carrot_goal_xy = target_point
                self._obstacle_map._carrot_goal_px = temp_target_px
                self._last_carrot_xy = target_point
                self._last_carrot_px = temp_target_px
            elif np.linalg.norm(this_stair_end - robot_px) <= 0.5 * self._obstacle_map.pixels_per_meter or self._obstacle_map._disable_end == True:
                self._carrot_goal_xy = target_point
                self._obstacle_map._carrot_goal_px = temp_target_px
                self._last_carrot_xy = target_point
                self._last_carrot_px = temp_target_px
            else: 
                l1_distance = np.abs(this_stair_end[0] - temp_target_px[0][0]) + np.abs(this_stair_end[1] - temp_target_px[0][1])
                last_l1_distance = np.abs(this_stair_end[0] - self._last_carrot_px[0][0]) + np.abs(this_stair_end[1] - self._last_carrot_px[0][1])
                if last_l1_distance > l1_distance:
                    self._carrot_goal_xy = target_point
                    self._obstacle_map._carrot_goal_px = temp_target_px
                    self._last_carrot_xy = target_point
                    self._last_carrot_px = temp_target_px
                else: 
                    self._carrot_goal_xy = self._last_carrot_xy
                    self._obstacle_map._carrot_goal_px = self._last_carrot_px

            rho, theta = rho_theta(robot_xy, heading, self._carrot_goal_xy) 
            rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
            obs_pointnav = {
                "depth": image_resize(
                    self._observations_cache["nav_depth"],
                    (self._depth_image_shape[0], self._depth_image_shape[1]),
                    channels_last=True,
                    interpolation_mode="area",
                ),
                "pointgoal_with_gps_compass": rho_theta_tensor,
            }
            self._policy_info["rho_theta"] = np.array([rho, theta])
            action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            if action.item() == 0:
                print("Might stop, change to move forward.")
                action[0] = 1
            return action
        
@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.85
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 0.8
    min_obstacle_height: float = 0.50
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = True
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  
    @property
    def kwaarg_names(cls) -> List[str]:
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]



cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
