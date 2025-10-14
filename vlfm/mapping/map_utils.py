import numpy as np
import os
import cv2
from vlfm.mapping.obstacle_map import ObstacleMap, filter_points_by_height
from vlfm.mapping.value_map import ValueMap
from habitat_baselines.common.tensor_dict import TensorDict
from vlfm.mapping.base_map import BaseMap
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap, too_offset, get_random_subarray, open3d_dbscan_filtering 
from PIL import Image
from vlfm.utils.geometry_utils import get_fov, within_fov_cone
from torch.nn import functional as F
from torch.autograd import Variable as V
from vlfm.utils.img_utils import fill_small_holes
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from typing import Dict
from collections import deque
STAIR_CLASS_ID = 17
def filter_points_by_height_below_ground_0(points: np.ndarray) -> np.ndarray:
    data = points[(points[:, 2] < -0.25) & (points[:, 2] > -1.5)] 
    return data

def clear_connected_region(map_array, start_y, start_x):
    rows, cols = map_array.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  


    queue = deque([(start_y, start_x)])
    map_array[start_y, start_x] = False  

    while queue:
        y, x = queue.popleft()
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and map_array[ny, nx]:  
                map_array[ny, nx] = False  
                queue.append((ny, nx))  

class ObstacleMapUpdater(ObstacleMap):

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  
        hole_area_thresh: int = 100000,  
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(min_height,max_height,agent_radius,area_thresh,hole_area_thresh,size,pixels_per_meter)

        self._map_dtype = np.dtype(bool)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self.radius_padding_color = (100, 100, 100)

        self._map_size = size
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)

        self._up_stair_map = np.zeros((size, size), dtype=bool)  
        self._down_stair_map = np.zeros((size, size), dtype=bool)  
        self._disabled_stair_map = np.zeros((size, size), dtype=bool)  
        self._min_height = min_height
        self._max_height = max_height
        self._agent_radius = agent_radius
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2 
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)

        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

        self._has_up_stair = False
        self._has_down_stair = False
        self._done_initializing = False
        self._this_floor_explored = False

        self._up_stair_frontiers_px = np.array([])
        self._up_stair_frontiers = np.array([])
        self._down_stair_frontiers_px = np.array([])
        self._down_stair_frontiers = np.array([])

        self._up_stair_start = np.array([])
        self._up_stair_end = np.array([])
        self._down_stair_start = np.array([])
        self._down_stair_end = np.array([])

        self._carrot_goal_px = np.array([])
        self._explored_up_stair = False
        self._explored_down_stair = False

        self.stair_boundary = np.zeros((size, size), dtype=bool)
        self.stair_boundary_goal = np.zeros((size, size), dtype=bool)
        self._floor_num_steps = 0
        self._disabled_frontiers = set()
        self._disabled_frontiers_px =  np.array([], dtype=np.float64).reshape(0, 2) 
        self._climb_stair_paused_step = 0
        self._disable_end = False
        self._look_for_downstair_flag = False
        self._potential_stair_centroid_px = np.array([])
        self._potential_stair_centroid = np.array([])
        self._reinitialize_flag = False
        self._tight_search_thresh = True
        self._best_frontier_selection_count = {}

        self.previous_frontiers = []  
        self.frontier_visualization_info = {}  
        self._each_step_rgb = {} 
        self._finish_first_explore = False
        self._neighbor_search = False
        self._climb_stair_over = True 
        self._reach_stair = False 
        self._reach_stair_centroid = False
        self._stair_frontier = None
        
    def reset(self) -> None:
        super().reset()
        self._map_dtype = np.dtype(bool)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self.radius_padding_color = (100, 100, 100)
        self._navigable_map.fill(0)
        self._up_stair_map.fill(0) 
        self._down_stair_map.fill(0) 
        self._disabled_stair_map.fill(0) 
        self.explored_area.fill(0)
        self.stair_boundary.fill(0)
        self.stair_boundary_goal.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self._has_up_stair = False
        self._has_down_stair = False
        self._explored_up_stair = False
        self._explored_down_stair = False
        self._done_initializing = False
        self._up_stair_frontiers_px = np.array([])
        self._up_stair_frontiers = np.array([])
        self._down_stair_frontiers_px = np.array([])
        self._down_stair_frontiers = np.array([])
        self._up_stair_start = np.array([])
        self._up_stair_end = np.array([])
        self._down_stair_start = np.array([])
        self._down_stair_end = np.array([])
        self._carrot_goal_px = np.array([])
        self._floor_num_steps = 0      
        self._disabled_frontiers = set()
        self._disabled_frontiers_px =  np.array([], dtype=np.float64).reshape(0, 2) 
        self._climb_stair_paused_step = 0
        self._disable_end = False
        self._look_for_downstair_flag = False
        self._potential_stair_centroid_px = np.array([])
        self._potential_stair_centroid = np.array([])
        self._reinitialize_flag = False
        self._tight_search_thresh = True
        self._best_frontier_selection_count = {}
        self.previous_frontiers = []  
        self.frontier_visualization_info = {}  
        self._each_step_rgb = {}
        self._each_step_rgb_phash = {}
        self._finish_first_explore = False
        self._neighbor_search = False
    
    def is_robot_in_stair_map_fast(self, robot_px:np.ndarray, stair_map: np.ndarray):
        x, y = robot_px[0, 0], robot_px[0, 1]
        radius_px = self._agent_radius * self.pixels_per_meter
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

    def visualize(self) -> np.ndarray:
        temp_disabled_frontiers = np.atleast_2d(np.array(list(self._disabled_frontiers)))
        if len(temp_disabled_frontiers[0]) > 0:
            self._disabled_frontiers_px = self._xy_to_px(temp_disabled_frontiers)
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        vis_img[self.explored_area == 1] = (200, 255, 200)
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        vis_img[self._map == 1] = (0, 0, 0)
        vis_img[self._up_stair_map == 1] = (128,0,128)
        vis_img[self._down_stair_map == 1] = (139, 26, 26)
        
        for carrot in self._carrot_goal_px:
            cv2.circle(vis_img, tuple([int(i) for i in carrot]), 5, (42, 42, 165), 2) 
        if len(self._down_stair_end) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._down_stair_end]), 5, (0, 255, 255), 2) 
        if len(self._up_stair_end) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._up_stair_end]), 5, (0, 255, 255), 2) 
        if len(self._down_stair_start) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._down_stair_start]), 5, (101, 96, 127), 2) 
        if len(self._up_stair_start) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._up_stair_start]), 5, (101, 96, 127), 2) 
        for frontier in self._frontiers_px:
            temp = np.array([int(i) for i in frontier])
            if temp not in self._disabled_frontiers_px:
                cv2.circle(vis_img, tuple(temp), 5, (200, 0, 0), 2) 
        for up_stair_frontier in self._up_stair_frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in up_stair_frontier]), 5, (255, 128, 0), 2) 
        for down_stair_frontier in self._down_stair_frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in down_stair_frontier]), 5, (19, 69, 139), 2) 
                    
        for potential_downstair in self._potential_stair_centroid_px:
            cv2.circle(vis_img, tuple([int(i) for i in potential_downstair]), 5, (128, 69, 128), 2) 

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img
    def visualize_and_save_frontiers(self, save_dir="debug/20250124/test_frontier_rgb/v7"):
        """
        Visualizes frontiers on the RGB images using the stored information in self._each_step_rgb
        and self.frontier_visualization_info, and saves the images to the specified directory.

        Args:
            save_dir (str): Directory to save the visualized images.
        """
        os.makedirs(save_dir, exist_ok=True)

        for floor_num_steps, rgb_image in self._each_step_rgb.items():
            for frontier, info in self.frontier_visualization_info.items():
                visualized_rgb = rgb_image.copy()
                if info['floor_num_steps'] == floor_num_steps:
                    arrow_end_pixel = info['arrow_end_pixel']
                    arrow_start_pixel = (visualized_rgb.shape[1] // 2, int(visualized_rgb.shape[0] * 0.9))  
                    center = (arrow_start_pixel[0], arrow_start_pixel[1])
                    axes = (5, 5)  
                    cv2.ellipse(visualized_rgb, center, axes, 0, 0, 360, (0, 255, 0), -1)
                    cv2.arrowedLine(
                        visualized_rgb,
                        arrow_start_pixel,
                        arrow_end_pixel,
                        color=(0, 0, 255),  
                        thickness=4,
                        tipLength=0.08
                    )
                    filename = f"floor_{floor_num_steps}_x_{int(frontier[0])}_y_{int(frontier[1])}.png"
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, visualized_rgb)
                    print(f"Saved: {save_path}")
    def update_map_with_stair(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        stair_mask: np.ndarray, 
        seg_mask: np.ndarray,
        agent_pitch_angle: int,
        search_stair_over: bool,
        reach_stair: bool,
        climb_stair_flag: int,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """

        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
                         
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            if (np.any(stair_mask) > 0 and np.sum(seg_mask == STAIR_CLASS_ID) > 20) or np.sum(seg_mask == STAIR_CLASS_ID) > 2000: 
                stair_map = (seg_mask == STAIR_CLASS_ID)
                if np.any(stair_mask) > 0:
                    fusion_stair_mask = stair_mask & stair_map
                else:
                    fusion_stair_mask = stair_map
                if np.any(fusion_stair_mask) > 0: 
                    stair_depth = np.full_like(depth, max_depth)
                    scaled_depth_stair = scaled_depth.copy()
                    stair_depth[fusion_stair_mask] = scaled_depth_stair[fusion_stair_mask]
                    stair_cloud_camera_frame = get_point_cloud(stair_depth, fusion_stair_mask, fx, fy)
                    stair_cloud_episodic_frame = transform_points(tf_camera_to_episodic, stair_cloud_camera_frame)
                    stair_xy_points = stair_cloud_episodic_frame[:, :2]
                    stair_pixel_points = self._xy_to_px(stair_xy_points)
                    if agent_pitch_angle >= 0 and climb_stair_flag != 2: 
                        for x, y in stair_pixel_points:
                            if 0 <= x < self._up_stair_map.shape[1] and 0 <= y < self._up_stair_map.shape[0] and self._up_stair_map[y, x] == 0:
                                self._up_stair_map[y, x] = 1
                        self._map[self._up_stair_map == 1] = 1
                    elif agent_pitch_angle < 0 and climb_stair_flag != 1: 
                        for x, y in stair_pixel_points:
                            if 0 <= x < self._down_stair_map.shape[1] and 0 <= y < self._down_stair_map.shape[0] and self._down_stair_map[y, x] == 0:
                                self._down_stair_map[y, x] = 1 
                        self._map[self._down_stair_map == 1] = 1 
            if agent_pitch_angle <= 0 and reach_stair == False: 
                filled_depth_for_stair = fill_small_holes(depth, self._hole_area_thresh)
                inverted_depth_for_stair = max_depth - filled_depth_for_stair * (max_depth - min_depth)
                inverted_mask = inverted_depth_for_stair < 2.4 
                inverted_point_cloud_camera_frame = get_point_cloud(inverted_depth_for_stair, inverted_mask, fx, fy)
                inverted_point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, inverted_point_cloud_camera_frame)
                if inverted_point_cloud_episodic_frame.shape[0] > 0:
                    below_ground_obstacle_cloud_0 = filter_points_by_height_below_ground_0(inverted_point_cloud_episodic_frame)
                    below_ground_xy_points = below_ground_obstacle_cloud_0[:, :2] 
                    below_ground_pixel_points = self._xy_to_px(below_ground_xy_points)
                    self._down_stair_map[below_ground_pixel_points[:, 1], below_ground_pixel_points[:, 0]] = 1
            if search_stair_over == True: 
                mask = scaled_depth < max_depth
                point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
                point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
                obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

                xy_points = obstacle_cloud[:, :2]
                pixel_points = self._xy_to_px(xy_points)

                self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1
            
            self._up_stair_map = self._up_stair_map & (~self._disabled_stair_map)
            self._down_stair_map = self._down_stair_map & (~self._disabled_stair_map)
            
            stair_dilated_mask = (self._up_stair_map == 1) | (self._down_stair_map == 1)
            self._map[stair_dilated_mask] = 0

            dilated_map = cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1
            )
            dilated_map[stair_dilated_mask] = 1
            self._map[stair_dilated_mask] = 1
            self._navigable_map = 1 - dilated_map.astype(bool)

        if not explore:
            return
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        ) 
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)

        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) >= 1:
            largest = max(contours, key=cv2.contourArea)  
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, [largest], -1, 1, -1)
            self.explored_area = new_area.astype(bool)
        self._frontiers_px = self._get_frontiers() 
        if len(self._frontiers_px) == 0: 
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)
            
        if np.sum(self._down_stair_map == 1) > 20:
            self._down_stair_map = cv2.morphologyEx(self._down_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) 
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map, connectivity=8)
            min_area_threshold = 10  
            filtered_map = np.zeros_like(self._down_stair_map)
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):  
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area_threshold:
                    filtered_map[labels == i] = 1  
                    if area > max_area:
                        max_area = area
                        max_label = i

            self._down_stair_map = filtered_map
            self._down_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

            self._down_stair_frontiers = self._px_to_xy(self._down_stair_frontiers_px)
            self._has_down_stair = True
            self._look_for_downstair_flag = False
            self._potential_stair_centroid_px = np.array([])
            self._potential_stair_centroid = np.array([])
        else:
            if np.sum(self._down_stair_map == 1) > 0:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map.astype(np.uint8), connectivity=8)
                max_area = 0
                max_label = 1
                for i in range(1, num_labels):  
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > max_area:
                        max_area = area
                        max_label = i
                self._potential_stair_centroid_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])
                self._potential_stair_centroid = self._px_to_xy(self._potential_stair_centroid_px)
                
            if np.sum(self._down_stair_map == 1) == 0:
                self._down_stair_frontiers_px = np.array([])
                self._down_stair_frontiers = np.array([])
                self._has_down_stair = False

        if np.sum(self._up_stair_map == 1) > 20:
            self._up_stair_map = cv2.morphologyEx(self._up_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) 
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._up_stair_map, connectivity=8)
            min_area_threshold = 10  
            filtered_map = np.zeros_like(self._up_stair_map)
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):  
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area_threshold:
                    filtered_map[labels == i] = 1  
                    if area > max_area:
                        max_area = area
                        max_label = i

            self._up_stair_map = filtered_map
            self._up_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

            self._up_stair_frontiers = self._px_to_xy(self._up_stair_frontiers_px)
            self._has_up_stair = True
        else:
            self._up_stair_frontiers_px = np.array([]) 
            self._has_up_stair = False

        if len(self._down_stair_frontiers) == 0 and np.sum(self._down_stair_map) > 0:
            self._look_for_downstair_flag = True 
    def project_frontiers_to_rgb_hush(self, rgb: np.ndarray) -> dict: 
        """
        Projects the frontiers from the map to the corresponding positions in the RGB image,
        and visualizes them on the RGB image.

        Args:
            rgb (np.ndarray): The RGB image (H x W x 3).
            robot_xy (np.ndarray): The robot's position in the map coordinates (x, y).
            min_arrow_length (float): The minimum length of the arrow in meters. Default is 4.0 meter.
            max_arrow_length (float): The maximum length of the arrow in meters. Default is 10.0 meter.

        Returns:
            dict: A dictionary containing the visualized RGB images with frontiers marked for each new frontier.
        """
        if len(self.frontiers) == 0:
            return {} 

        new_frontiers = [f for f in self.frontiers if f.tolist() not in self.previous_frontiers]
        if len(new_frontiers) == 0:
            return {}
        
        self.previous_frontiers.extend([f.tolist() for f in new_frontiers])

        visualized_rgb_ori = rgb.copy()
        self._each_step_rgb[self._floor_num_steps] = visualized_rgb_ori

        for frontier in new_frontiers:

            visualization_info = {
                'floor_num_steps': self._floor_num_steps,
            }
            self.frontier_visualization_info[tuple(frontier)] = visualization_info

    def extract_frontiers_with_image(self, frontier):
        """
        Visualizes frontiers on the RGB images using the stored information in self._each_step_rgb
        and self.frontier_visualization_info. Draws a blue circle with index at the end of a line.
        """
        floor_num_steps = self.frontier_visualization_info[tuple(frontier)]['floor_num_steps']
        visualized_rgb = self._each_step_rgb[floor_num_steps].copy()
        return floor_num_steps, visualized_rgb
    
class ValueMapUpdater(ValueMap):
    def _update_value_map(self, observations_cache, text_prompt, target_object, itm) -> None:
        all_rgb = []
        all_texts = []
        rgb = observations_cache["value_map_rgbd"][0][0]
        all_rgb.append(rgb)
        text = text_prompt.replace("target_object", target_object.replace("|", "/"))
        all_texts.append(text)
        all_cosines = itm.cosine(all_rgb, all_texts)
        self.update_map(
            np.array([all_cosines]),  
            observations_cache["value_map_rgbd"][0][1],
            observations_cache["value_map_rgbd"][0][2],
            observations_cache["value_map_rgbd"][0][3],
            observations_cache["value_map_rgbd"][0][4],
            observations_cache["value_map_rgbd"][0][5]
        )
        self.update_agent_traj(
            observations_cache["robot_xy"],
            observations_cache["robot_heading"],
        )
        self._blip_cosine = all_cosines

    def reset(self) -> None:
        super().reset()
        self._value_map.fill(0)
        self._confidence_masks = {}
        self._camera_positions = []
        self._last_camera_yaw = 0.0
        self._min_confidence = 0.25
        self._decision_threshold = 0.35

class ObjectMapUpdater(ObjectPointCloudMap,BaseMap):
    def __init__(self, erosion_size: float, size: int = 1000) -> None:
        ObjectPointCloudMap.__init__(self,erosion_size=erosion_size)
        BaseMap.__init__(self,size=size)
        self._map = np.zeros((size, size), dtype=bool)
        self._disabled_object_map = np.zeros((size, size), dtype=bool)  
        self.clouds = {}
        self.use_dbscan = True
        self.stair_clouds: Dict[str, np.ndarray] = {}
        self.visualization = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255  
        self.each_step_objects = {}
        self.each_step_rooms = {}
        self.this_floor_rooms = set()
        self.this_floor_objects = set()

    def reset(self) -> None:
        ObjectPointCloudMap.reset(self)
        BaseMap.reset(self)
        self._map.fill(0)
        self._disabled_object_map.fill(0)
        self.use_dbscan = True
        self.clouds = {}
        self.last_target_coord = None
        self.stair_clouds = {}

    def update_map(
        self,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None:
        """Updates the object map with the latest information from the agent."""
        local_cloud = self._extract_object_cloud(depth_img, object_mask, min_depth, max_depth, fx, fy)
        if len(local_cloud) == 0:
            return
        if too_offset(object_mask):
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
        else:
            within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0 
            within_range = within_range.astype(np.float32)
            within_range[within_range == 0] = np.random.rand()
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)

        xy_points = global_cloud[:, :2]
        pixel_points = self._xy_to_px(xy_points)
        valid_points_mask = ~self._disabled_object_map[pixel_points[:, 1], pixel_points[:, 0]]

        global_cloud = global_cloud[valid_points_mask]
        if len(global_cloud) == 0:
            return  
            
        xy_points = global_cloud[:, :2]
        pixel_points = self._xy_to_px(xy_points)
        self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1
        self._map = self._map & (~self._disabled_object_map)

        curr_position = tf_camera_to_episodic[:3, 3]
        closest_point = self._get_closest_point(global_cloud, curr_position)
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        if dist <= 0.5: 
            return

        if object_name in self.clouds:
            self.clouds[object_name] = np.concatenate((self.clouds[object_name], global_cloud), axis=0)
        else:
            self.clouds[object_name] = global_cloud
    def visualize(self) -> np.ndarray:  
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        vis_img[self._map == 1] = (0, 0, 128)
        vis_img = cv2.flip(vis_img, 0)
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )
        return vis_img

    def _extract_object_cloud(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        final_mask = object_mask * 255
        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)  

        valid_depth = depth.copy()
        valid_mask = (valid_depth > 0) & final_mask
        
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        cloud = get_point_cloud(valid_depth, valid_mask, fx, fy) 
        cloud = get_random_subarray(cloud, 5000)
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud)

        return cloud
    def update_explored(self, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float) -> None:
        """
        This method will remove all point clouds in self.clouds that were originally
        detected to be out-of-range, but are now within range. This is just a heuristic
        that suppresses ephemeral false positives that we now confirm are not actually
        target objects.

        Args:
            tf_camera_to_episodic: The transform from the camera to the episode frame.
            max_depth: The maximum distance from the camera that we consider to be
                within range.
            cone_fov: The field of view of the camera.
        """
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        for obj in self.clouds:
            within_range = within_fov_cone(
                camera_coordinates,
                camera_yaw,
                cone_fov,
                max_depth * 0.5,
                self.clouds[obj],
            )
            range_ids = set(within_range[..., -1].tolist())
            for range_id in range_ids:
                if range_id == 1:
                    continue
                self.clouds[obj] = self.clouds[obj][self.clouds[obj][..., -1] != range_id]
                
        for obj in list(self.clouds.keys()):
            if self.clouds[obj].size == 0:
                del self.clouds[obj]