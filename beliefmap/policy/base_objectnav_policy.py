# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union
import base64
import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor
from beliefmap.utils.geometry_utils import transform_points
from beliefmap.mapping.object_point_cloud_map import ObjectPointCloudMap
from beliefmap.mapping.obstacle_map import ObstacleMap
from beliefmap.obs_transformers.utils import image_resize
from beliefmap.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from beliefmap.utils.geometry_utils import get_fov, rho_theta
from beliefmap.vlm.blip2 import BLIP2Client
from beliefmap.vlm.coco_classes import COCO_CLASSES
from beliefmap.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from beliefmap.vlm.sam import MobileSAMClient
from beliefmap.vlm.yolov7 import YOLOv7Client
from beliefmap.vlm.openai_api import OpenAI_API
try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from beliefmap.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
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
        self.result_path = os.environ.get("result_path")
        # if use_vqa:
        #     self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = True
        # self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold
        self._objects_attributes = None
        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )
        self.openai_client = OpenAI_API()
    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True
        self.fist_detect = True
        self.counter +=1
        self.double_detect = True

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
        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_target_object_location(robot_xy)
        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
        elif goal is None:  # Haven't found target object yet
            mode = "explore"
            # TODO change the explore strategy########
            pointnav_action = self._explore(observations)
        else:
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True)

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1

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
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
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

        return detections

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
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        
        object_idx = None
        close_to_gloal = False
        if self._object_map.has_object(self._target_object) and (len(detections.logits) > 0):
            robot_xy = self._observations_cache["robot_xy"]
            goal_point = self._object_map.get_best_object(self._target_object, robot_xy)
            if np.linalg.norm(goal_point[:2] - robot_xy[:2]) < 1.5:
                close_to_gloal = True
        if (4 > len(detections.logits) > 1) and (not close_to_gloal) and (self.double_detect):
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
            annotated_rgb_multi = cv2.cvtColor(annotated_rgb_multi, cv2.COLOR_RGB2BGR)
            height, width, _ = annotated_rgb_multi.shape

            # 选择字体、大小、颜色和厚度
            text = answer_color
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(width, height) / 500  # 根据图片大小调整字体
            font_thickness = 2
            color = (0, 0, 0)  # 黑色 (B, G, R)

            # 获取文本大小
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # 计算文本位置（居中）
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2  # OpenCV 文字的 y 坐标是基线

            # 添加文本
            cv2.putText(annotated_rgb_multi, text, (text_x, text_y), font, font_scale, color, font_thickness)
            cv2.imwrite(f'{self.result_path}/VLM/annotated_rgb_{self.counter}.jpg', annotated_rgb_multi)
            if "red" in answer_color.lower():
                object_idx = 0
            elif "green" in answer_color.lower():
                object_idx = 1
            elif "blue" in answer_color.lower():
                object_idx = 2
            else:
                object_idx = 0
            self.fist_detect = True
            self._object_map.reset()
            self.double_detect = False
            
        if object_idx is None:
            for idx in range(len(detections.logits)):
                bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
                object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

                # If we are using vqa, then use the BLIP2 model to visually confirm whether
                # the contours are actually correct.
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
                    if "yes" in answer.lower():
                        self.fist_detect = False
                        
                    else:
                        print("ignore the detection")
                        self.double_detect = True
                        continue

                self._object_masks[object_mask > 0] = 1
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
        else:
            for idx in [object_idx]:
                bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
                object_mask = self._mobile_sam.segment_bbox(rgb.copy(), bbox_denorm.tolist())

                # If we are using vqa, then use the BLIP2 model to visually confirm whether
                # the contours are actually correct.

                if self._use_vqa and self.fist_detect:
                    contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                    answer = self.openai_client.detection_refinement(annotated_rgb, detections.phrases[idx])
                    if not ("yes" in answer.lower()):
                        print("ignore the detection")
                        self.double_detect = True
                        continue
                    else:
                        self.fist_detect = False

                self._object_masks[object_mask > 0] = 1
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
        # Find the bounding rectangle of the mask
        x, y, w, h = cv2.boundingRect(mask)

        # Calculate the thirds of the mask
        fourth = mask.shape[1] // 4

        # Check if the entire bounding rectangle is in the left or right third of the mask
        if x + w <= fourth:
            # Check if the leftmost point is at the edge of the image
            # return x == 0
            return x <= int(0.05 * mask.shape[1])
        elif x >= 3 * fourth:
            # Check if the rightmost point is at the edge of the image
            # return x + w == mask.shape[1]
            return x + w >= int(0.95 * mask.shape[1])
        else:
            return False

@dataclass
class BeliefmapConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.85
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 0.8 # in square meters
    min_obstacle_height: float = 0.35
    max_obstacle_height: float = 0.6
    hole_area_thresh: int = 100000
    use_vqa: bool = True
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(BeliefmapConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="beliefmap_config_base", node=BeliefmapConfig())
