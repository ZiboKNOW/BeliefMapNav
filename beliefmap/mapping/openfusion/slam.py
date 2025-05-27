import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c
import time
import threading
import sys
import itertools
from matplotlib.colors import ListedColormap
from ultralytics import SAM
import cv2
from beliefmap.mapping.openfusion.state import BaseState, CFState
from beliefmap.mapping.openfusion.utils import rand_cmap, get_cmap_legend

from matplotlib import pyplot as plt
from typing import List, Tuple, Union, Optional, Dict, Any
from lavis.models import load_model_and_preprocess
from PIL import Image
try:
    import torchshow as ts
except:
    print("[*] install `torchshow` for easier debugging")


class BaseSLAM(object):
    def __init__(
        self,
        intrinsic,
        point_state: BaseState,
        with_pose=True,
        img_size=(640, 480),
        live_mode=False,
        capture=False
    ) -> None:
        self.intrinsic = intrinsic
        self.point_state = point_state
        self.with_pose = with_pose
        self.height = img_size[1]
        self.width = img_size[0]
        self.export_path = None
        self.live_mode = live_mode
        self.capture = capture
        self.selected_points = None
        self.selected_colors = None
        self.control_thread_enabled = False
        self.monitor_thread_enabled = False
        self._camera_positions = []
        self._last_camera_yaw = 0

    def save(self, path=None):
        if path is None:
            path = self.export_path
        self.point_state.save(path)

    def load(self, path):
        self.point_state.load(path)


    def compute_state(self,batch_color,batch_depth,batch_extrinsic,observation_range, **kwargs):
            self.point_state.update(batch_color, batch_depth, batch_extrinsic, observation_range)


    def monitor(self):
        if len(self.point_state.poses) <= 1:
            return
        time.sleep(0.02)
        points, colors = self.point_state.get_pc(500000)
        if len(points) == 0:
            return
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        if self.selected_points is not None:
            # NOTE: highlight selected points
            self.selected_pcd.points = o3d.utility.Vector3dVector(self.selected_points)
            self.selected_pcd.colors = o3d.utility.Vector3dVector(self.selected_colors)
        pose = np.linalg.inv(self.point_state.get_last_pose())

        if not self.is_monitor_init:
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.selected_pcd)
            self.last_pose = pose
            self.cur_camera_cf.transform(pose)
            self.cur_camera_lines.transform(pose)
            self.vis.add_geometry(self.cur_camera_cf)
            self.vis.add_geometry(self.cur_camera_lines)
            self.is_monitor_init = True
        else:
            self.vis.update_geometry(self.pcd)
            self.vis.update_geometry(self.selected_pcd)

            if np.any(pose != self.last_pose):
                transform = pose @ np.linalg.inv(self.last_pose)
                self.cur_camera_cf.transform(transform)
                self.cur_camera_lines.transform(transform)
                self.vis.update_geometry(self.cur_camera_cf)
                self.vis.update_geometry(self.cur_camera_lines)
                self.last_pose = pose

        # Force the visualizer to redraw
        self.vis.poll_events()
        self.vis.update_renderer()


class VLSLAM(BaseSLAM):
    def __init__(
        self,
        intrinsic,
        point_state:BaseState,
        with_pose=True,
        img_size=(640, 480),
        vl_model=None,
        host_ip="127.0.0.1",
        query_port=5000,
        live_mode=False
    ) -> None:
        super().__init__(intrinsic, point_state, with_pose, img_size, live_mode)
        if isinstance(point_state, CFState):
            self.mode = "emb"
        else:
            self.mode = "default"
        self.host_ip = host_ip
        self.query_port = query_port
        self.clip_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.clip_model, self.vis_processors, self.text_processors = \
            load_model_and_preprocess("clip_feature_extractor",model_type="ViT-B-32",is_eval=True, device=self.clip_device)
        
        self.sam2_model = SAM("/home/ubuntu/DATA2/zzb/sam2.1_b.pt")

    @torch.no_grad()
    def reset(self):
        self.point_state.reset()
        self._camera_positions = []
        self._last_camera_yaw = 0
    def get_pyramid_clip_embedding(self, images, vis_processors, txt_processors, score_processor, clip_devices=None, scales=[1, 2, 4]):
        """
        Crops the image into multiple patches at different scales, normalizes these embedding, and maps them back to the original image pixels.

        Parameters:
        - image: The PIL Image to be processed.
        - depth: The depth image to calculate the weight .
        - vis_processors: A function that processes each patch and returns a tensor suitable for scoring.
        - score_processor: A function that takes a batch of patches and returns scores.
        - scales: A list of scales at which to crop the image.

        Returns:
        - A tuple of (cropped_images, scores_map) where scores_map is a numpy array of the same shape as the original image with scores assigned to each pixel.
        """  
        def crop_numpy(img, num_x, num_y):
            np_img = np.array(img)
            h, w, _ = np_img.shape
            iamge_crop = Image.fromarray(np_img)
            return [vis_processors(Image.fromarray(np_img[i * h // num_y:(i + 1) * h // num_y,
                                                j * w // num_x:(j + 1) * w // num_x, :])).to(clip_devices)
                    for i in range(num_y) for j in range(num_x)]
        def crop_mask(img,num_x,num_y):
            np_img = img
            h, w, _ = np_img.shape
            return [np_img[i * h // num_y:(i + 1) * h // num_y,
                    j * w // num_x:(j + 1) * w // num_x, :]
                    for i in range(num_y) for j in range(num_x)]
        res_list = []
        for seg_image in images:
            plt.clf()
            seg_results = self.sam2_model(seg_image,verbose=False)
            H,W,_= seg_image.shape
            if not seg_results[0].masks:
                merged_mask = np.zeros((H, W), dtype=np.int32)
                continue

            masks = seg_results[0].masks.data.cpu().numpy()  
            N, H, W = masks.shape

            merged_mask = np.zeros((H, W), dtype=np.int32)
            for i in range(N):
                merged_mask[masks[i]] = i + 1
          


        for image in images:
            image = Image.fromarray(image[...,0:3].astype(np.uint8))
            cropped_images = []
            cropped_batches = []
            # Define the crop function using numpy slicing
            # Process each scale
            for scale in scales:
                num_x, num_y = scale, scale
                patches = crop_numpy(image, num_x, num_y)
                cropped_images.extend(patches)
                # batch = torch.stack(cropped_images)
            sample = {"image": torch.stack(cropped_images), "text_input": None}
            clip_features = score_processor.extract_features(sample)
            features_image = clip_features

            embedding_key_map = np.zeros(np.array(image).shape[:2] + (len(scales),), dtype=np.float32)
            patch_idx = 0
            for scale_index, scale in enumerate(scales):
                num_x, num_y = scale, scale
                patch_height = embedding_key_map.shape[0] // num_y
                patch_width = embedding_key_map.shape[1] // num_x
                for i in range(num_y):
                    for j in range(num_x):
                        # scores_map[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] += normalized_scores[patch_idx]
                        embedding_key_map[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, scale_index] = \
                        np.ones_like(embedding_key_map[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, scale_index]) * patch_idx
                        patch_idx += 1

            embedding_key_map = embedding_key_map
            res_dict = {"cropped_images": cropped_images, "embedding_keys": embedding_key_map, "mebbedings": features_image,"object_mask":torch.tensor(merged_mask)}
            res_list.append(res_dict)
        return res_list
    def compute_state(self,batch_color,batch_depth, batch_extrinsic,batch_agent_height,observation_range =[1.0,1.5], encode_image=True):
        if encode_image:
            res_list = self.get_pyramid_clip_embedding(batch_color, self.vis_processors["eval"],self.text_processors["eval"],self.clip_model,clip_devices = self.clip_device)
        else:
            res_list = [{} for _ in range(len(batch_color))]
        for color, depth, extrinsic,agent_height, res_dict in zip(
            batch_color, batch_depth, batch_extrinsic,batch_agent_height, res_list
        ):
            if depth.max() == depth.min():
                print("invalid depth")
                return
            self.point_state.update(color, depth, extrinsic, agent_height, res_dict, observation_range = observation_range)
    
    def get_score_map(self,attributes_embeddings):
        points, score,best_scale, best_attributes = self.point_state.get_score_map(attributes_embeddings)
        return points, score,best_scale,best_attributes
    def get_observed_area(self):
        observation_point, confidence = self.point_state.get_observed_area()
        return observation_point, confidence

    def update_agent_traj(self, robot_xy: np.ndarray, robot_heading: float) -> None:
        self._camera_positions.append(robot_xy)
        self._last_camera_yaw = robot_heading
    def semantic_query(
        self, query,flat = True
    ):
        """ perform semantic segmentation on the point cloud
        Args:
            query (List[str]): strings of semantic classes

        Returns:
            points (np.array): xyz coordinates
            colors (np.array): colored points
        """
        if flat:
            query_flat = list(itertools.chain(*query))
        else:
            query_flat = query
        text_input = self.text_processors["eval"](query_flat)
        # Create a sample dictionary
        sample = {"text_input": text_input}
        # Extract text features
        features = self.clip_model.extract_features(sample)
        # Get the text embeddings
        text_embedding = features
        return text_embedding

    def get_information_gain(self,attributes_embedding, attributes_probability,extrinsics,observation_range):
        information_gain_list = []
        exolore_gain_list = []
        semantic_gain_list = []
        fov_mask_list = []
        points_in_fov_list = []
        semantic_points_list = []
        if len(extrinsics) > 0:
            for extrinsic in extrinsics:
                information_gain,exolore_gain,semantic_gain,observed_in_fov_mask, points_in_fov,semantic_points = self.point_state.compute_info_gain(attributes_embedding,attributes_probability, extrinsic, observation_range)
                information_gain_list.append(information_gain)
                exolore_gain_list.append(exolore_gain)
                semantic_gain_list.append(semantic_gain)
                fov_mask_list.append(observed_in_fov_mask)
                points_in_fov_list.append(points_in_fov)
                semantic_points_list.append(semantic_points)
            return information_gain_list,exolore_gain_list,semantic_gain_list,fov_mask_list,points_in_fov_list,semantic_points_list
        else:
            return information_gain_list,exolore_gain_list,semantic_gain_list,fov_mask_list,points_in_fov_list,semantic_points_list
    def get_uninspected_points(self, query_embedding,room_embedding,query_probability):
        buf_coords, score_map, semantic_gain, active_confidence_value, cluster_points_list, room_coords,room_categories = self.point_state.get_uninspected_points(query_embedding,room_embedding,query_probability)
        return buf_coords, score_map, semantic_gain, active_confidence_value, cluster_points_list, room_coords, room_categories



def build_slam(intrinsic, params):
    with_pose = True
    assert params["img_size"][0] / params["input_size"][0] == \
            params["img_size"][1] / params["input_size"][1], \
            "[*] img_size and input_size should have the same aspect ratio"
    assert params["img_size"][0] % params["input_size"][0] == 0, \
            "[*] img_size should be divisible by input_size"

    if params['algo'] == "default":
        point_state = BaseState(
            intrinsic, params["depth_scale"], params["depth_max"],
            params["voxel_size"], params["block_resolution"], params["block_count"],
            device=params['device'].upper(), img_size=params["input_size"]
        )
        return BaseSLAM(intrinsic, io, point_state, with_pose, params["input_size"], live_mode=False)
    elif params['algo'] == "cfusion":
        point_state = CFState(
            intrinsic, params["depth_scale"], params["depth_max"],params["depth_min"],
            params["voxel_size"], params["block_resolution"], params["block_count"],
            dim=512, device= params['device'].upper(), img_size=params["input_size"],semantic_weight = 1.5,explore_weight = 0.007
        )
        return VLSLAM(intrinsic, point_state, with_pose, params["input_size"], live_mode=False)
    else:
        raise ValueError("Unknown SLAM algorithm: {}".format(args.algo))
