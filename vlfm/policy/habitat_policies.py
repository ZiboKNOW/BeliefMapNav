

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from frontier_exploration.base_explorer import BaseExplorer
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import (
    PolicyConfig,
)
from habitat_baselines.rl.ppo.policy import PolicyActionData
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import Tensor

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix, xyz_yaw_to_extrinsic, xyz_yaw_pitch_roll_to_extrinsic
from vlfm.vlm.grounding_dino import ObjectDetections
from config.openspace.build import get_config
from ..mapping.obstacle_map import ObstacleMap
from .base_objectnav_policy import BaseObjectNavPolicy, VLFMConfig
from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3
from ..mapping.openfusion.slam import build_slam
HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
MP3D_ID_TO_NAME = [
    "chair",
    "table|dining table|coffee table|side table|desk",  
    "framed photograph",  
    "cabinet",
    "pillow",  
    "couch",  
    "bed",
    "nightstand",  
    "potted plant",  
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",  
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)

def xyz_yaw_pitch_roll_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Converts a given position and yaw, pitch, roll angles to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians (rotation around Z-axis).
        pitch (float): The pitch angle in radians (rotation around Y-axis).
        roll (float): The roll angle in radians (rotation around X-axis).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    
    R = R_yaw @ R_pitch @ R_roll
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R  
    transformation_matrix[:3, 3] = [x, y, z]  

    return transformation_matrix

class HabitatMixin:
    """This Python mixin only contains code relevant for running a BaseObjectNavPolicy
    explicitly within Habitat (vs. the real world, etc.) and will endow any parent class
    (that is a subclass of BaseObjectNavPolicy) with the necessary methods to run in
    Habitat.
    """

    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = False

    def __init__(
        self,
        camera_height: float,
        min_depth: float,
        max_depth: float,
        camera_fov: float,
        image_width: int,
        image_height:int,
        dataset_type: str = "hm3d",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        self._cx = image_width /2
        self._cy = image_height /2 
        self._dataset_type = dataset_type
        intrinsic = np.array([[self._fx, 0,  self._cx], [0, self._fy, self._cy], [0, 0, 1]])
        params = get_config(dataset = "hm3d")
        params["img_size"] = (image_width,image_height)
        params["input_size"] = params["img_size"]
        params["depth_max"] = self._max_depth
        params["depth_min"] = self._min_depth
        self.slam = build_slam(intrinsic, params)
        self.flat_height = 0.0

        

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMixin":
        policy_config: VLFMPolicyConfig = config.habitat_baselines.rl.policy
        kwargs = {k: policy_config[k] for k in VLFMPolicyConfig.kwaarg_names}  
        sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width
        kwargs["image_height"] = sim_sensors_cfg.depth_sensor.height
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0

        if "hm3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "hm3d"
        elif "mp3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "mp3d"
        else:
            raise ValueError("Dataset type could not be inferred from habitat config")

        return cls(**kwargs)

    def act(
        self: Union["HabitatMixin", BaseObjectNavPolicy],
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        """Converts object ID to string name, returns action as PolicyActionData"""
        object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        obs_dict = observations.to_tree()
        if self._dataset_type == "hm3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = HM3D_ID_TO_NAME[object_id]
        elif self._dataset_type == "mp3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = MP3D_ID_TO_NAME[object_id]
            self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")
        parent_cls: BaseObjectNavPolicy = super()  
        try:
            action, rnn_hidden_states = parent_cls.act(obs_dict, rnn_hidden_states, prev_actions, masks, deterministic)
        except StopIteration:
            action = self._stop_action
            print("error in habitat_policy")
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        if self._initialize_step > 11: 
            self._done_initializing = True
            self._obstacle_map._tight_search_thresh = False 
        else:
            self._initialize_step += 1 
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        parent_cls: BaseObjectNavPolicy = super()  
        parent_cls._reset()
        self.flat_height = 0.0
        self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """Get policy info for logging"""
        parent_cls: BaseObjectNavPolicy = super()  
        info = parent_cls._get_policy_info(detections)

        if not self._visualize:  
            return info

        if self._start_yaw is None:
            self._start_yaw = self._observations_cache["habitat_start_yaw"]
        info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        if len(self._observations_cache) > 0:
            return
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        y, z, x = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        camera_position = np.array([-x, -y, self._camera_height])
        camera_position_height = np.array([-x, -y, self._camera_height + z])
        robot_xy = camera_position[:2]
        camera_pitch = np.radians(-self._pitch_angle) 
        camera_roll = 0
        
        tf_camera_to_episodic = xyz_yaw_pitch_roll_to_tf_matrix(camera_position, camera_yaw, camera_pitch, camera_roll)
        extrinsic = xyz_yaw_pitch_roll_to_extrinsic(camera_position_height, camera_yaw, camera_pitch, camera_roll)
        frontiers = np.array([])
        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": observations["depth"],  
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "robot_height":z,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    extrinsic,
                    z,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][0].item(),
        }

    
@baseline_registry.register_policy
class OracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def _explore(self, observations: TensorDict) -> Tensor:
        explorer_key = [k for k in observations.keys() if k.endswith("_explorer")][0]
        pointnav_action = observations[explorer_key]
        return pointnav_action


@baseline_registry.register_policy
class SuperOracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,  
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        return PolicyActionData(
            actions=observations[BaseExplorer.cls_uuid],
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )


@baseline_registry.register_policy
class HabitatITMPolicy(HabitatMixin, ITMPolicy):
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV2(HabitatMixin, ITMPolicyV2):
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV3(HabitatMixin, ITMPolicyV3):
    pass


@dataclass
class VLFMPolicyConfig(VLFMConfig, PolicyConfig):
    pass


cs = ConfigStore.instance()
cs.store(group="habitat_baselines/rl/policy", name="vlfm_policy", node=VLFMPolicyConfig)
