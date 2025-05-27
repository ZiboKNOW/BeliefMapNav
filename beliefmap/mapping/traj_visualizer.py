# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Union

import cv2
import numpy as np
import torch

class TrajectoryVisualizer:
    _num_drawn_points: int = 1
    _cached_path_mask: Union[np.ndarray, None] = None
    _origin_in_img: Union[np.ndarray, None] = None
    _pixels_per_meter: Union[float, None] = None
    agent_line_length: int = 10
    agent_line_thickness: int = 3
    path_color: tuple = (0, 255, 0)
    path_thickness: int = 3
    scale_factor: float = 1.0

    def __init__(self, origin_in_img: np.ndarray, pixels_per_meter: float):
        self._origin_in_img = origin_in_img
        self._pixels_per_meter = pixels_per_meter
        self.categories = ["Living Room", "Home Office", "Dining Room", "Bathroom","stairs", "Storage Room", "Kitchen","Bedroom","Toilet","corridor"]
        cmap = cv2.applyColorMap(np.linspace(0, 255, len(self.categories), dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
        self.room_colors = {cat: tuple(map(int, cmap[i, 0])) for i, cat in enumerate(self.categories)}

    def _generate_legend_room(self):
        """ 生成类别对应的颜色图例 """
        legend = np.ones((len(self.categories) * 30, 200, 3), dtype=np.uint8) * 255  # 竖直图例
        for i, (cat, color) in enumerate(self.room_colors.items()):
            cv2.rectangle(legend, (5, i * 30 + 5), (35, i * 30 + 25), color, -1)
            cv2.putText(legend, cat, (45, i * 30 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return legend
    

    def reset(self) -> None:
        self._num_drawn_points = 1
        self._cached_path_mask = None

    def draw_trajectory(
        self,
        img: np.ndarray,
        camera_positions: Union[np.ndarray, List[np.ndarray]],
        camera_yaw: float,
    ) -> np.ndarray:
        """Draws the trajectory on the image and returns it"""
        img = self._draw_path(img, camera_positions)
        img = self._draw_agent(img, camera_positions[-1], camera_yaw)
        return img

    def _draw_path(self, img: np.ndarray, camera_positions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Draws the path on the image and returns it"""
        if len(camera_positions) < 2:
            return img
        if self._cached_path_mask is not None:
            path_mask = self._cached_path_mask.copy()
        else:
            path_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for i in range(self._num_drawn_points - 1, len(camera_positions) - 1):
            path_mask = self._draw_line(path_mask, camera_positions[i], camera_positions[i + 1])

        img[path_mask == 255] = self.path_color

        self._cached_path_mask = path_mask
        self._num_drawn_points = len(camera_positions)

        return img

    def _draw_line(self, img: np.ndarray, pt_a: np.ndarray, pt_b: np.ndarray) -> np.ndarray:
        """Draws a line between two points and returns it"""
        # Convert metric coordinates to pixel coordinates
        px_a = self._metric_to_pixel(pt_a)
        px_b = self._metric_to_pixel(pt_b)

        if np.array_equal(px_a, px_b):
            return img

        cv2.line(
            img,
            tuple(px_a[::-1]),
            tuple(px_b[::-1]),
            255,
            int(self.path_thickness * self.scale_factor),
        )

        return img

    def _draw_agent(self, img: np.ndarray, camera_position: np.ndarray, camera_yaw: float) -> np.ndarray:
        """Draws the agent on the image and returns it"""
        px_position = self._metric_to_pixel(camera_position)
        cv2.circle(
            img,
            tuple(px_position[::-1]),
            int(8 * self.scale_factor),
            (255, 192, 15),
            -1,
        )
        heading_end_pt = (
            int(px_position[0] - self.agent_line_length * self.scale_factor * np.cos(camera_yaw)),
            int(px_position[1] - self.agent_line_length * self.scale_factor * np.sin(camera_yaw)),
        )
        cv2.line(
            img,
            tuple(px_position[::-1]),
            tuple(heading_end_pt[::-1]),
            (0, 0, 0),
            int(self.agent_line_thickness * self.scale_factor),
        )

        return img
    
    def _draw_yaw(self, img: np.ndarray, camera_position: np.ndarray, camera_yaw: float) -> np.ndarray:
        """Draws the agent on the image and returns it"""
        px_position = self._metric_to_pixel(camera_position)
        cv2.circle(
            img,
            tuple(px_position[::-1]),
            int(8 * self.scale_factor),
            (255, 0, 0),
            -1,
        )
        heading_end_pt = (
            int(px_position[0] - self.agent_line_length * self.scale_factor * np.cos(camera_yaw)),
            int(px_position[1] - self.agent_line_length * self.scale_factor * np.sin(camera_yaw)),
        )
        cv2.line(
            img,
            tuple(px_position[::-1]),
            tuple(heading_end_pt[::-1]),
            (0, 0, 0),
            int(self.agent_line_thickness * self.scale_factor),
        )

        return img

    def draw_circle(self, img: np.ndarray, position: np.ndarray,text = None, **kwargs: Any) -> np.ndarray:
        """Draws the point as a circle on the image and returns it"""
        px_position = self._metric_to_pixel(position)
        cv2.circle(img, tuple(px_position[::-1]), **kwargs)
        if text is not None:
            text_position = (px_position[1] - 60, px_position[0]- 10)
            cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img
    
    def draw_points(self, points: torch.Tensor, values: torch.Tensor) -> np.ndarray:
        points_np = points.cpu().numpy()  # (M,2)
        values_np = values.cpu().numpy()  # (M,)
        

        pixel_coords = self._metric_to_pixel(points_np)  # (M,2)
        

        img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        

        min_val = values_np.min()
        max_val = values_np.max()
        if max_val == min_val:
            norm_vals = np.zeros_like(values_np, dtype=np.uint8)
        else:
            norm_vals = ((values_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        

        norm_vals_img = norm_vals.reshape(-1, 1)  # (M,1)

        colors = cv2.applyColorMap(norm_vals_img, cv2.COLORMAP_JET)  # (M,1,3)
        colors = colors.reshape(-1, 3)  # (M,3) ，颜色格式为 BGR
        
        xs = pixel_coords[:, 0]
        ys = pixel_coords[:, 1]
        
        valid = (xs >= 0) & (xs < 1000) & (ys >= 0) & (ys < 10000)
        xs = xs[valid]
        ys = ys[valid]
        colors = colors[valid]
        
        # 使用向量化索引将颜色赋值到对应像素
        # 注意：OpenCV 图像的索引格式为 img[y, x]
        # img[ys, xs] = colors
        
        
        # 使用 cv2.circle 绘制圆点，radius 用来控制圆点大小
        for x, y, color in zip(xs, ys, colors):
            cv2.circle(img, (y, x), 2, color.tolist(), -1)  # 绘制填充的圆点
        colorbar = self._generate_colorbar(min_val, max_val)
        final_img = np.hstack([img, colorbar])
        return final_img
    
    
    
    def draw_points_room(self, points: torch.Tensor, indices: torch.Tensor) -> np.ndarray:
        points_np = points.cpu().numpy()  # (M,2)
        indices_np = indices.cpu().numpy()  # (M,)

        pixel_coords = self._metric_to_pixel(points_np)  # (M,2)

        img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # 白色背景

        xs = pixel_coords[:, 0]
        ys = pixel_coords[:, 1]

        valid = (xs >= 0) & (xs < 1000) & (ys >= 0) & (ys < 1000)  # 保证点在图像范围内
        xs = xs[valid]
        ys = ys[valid]
        indices_np = indices_np[valid]

        # 绘制散点
        for x, y, idx in zip(xs, ys, indices_np):
            color = self.room_colors[self.categories[idx]]
            cv2.circle(img, (y, x), 3, color, -1)  # 画小圆点

        # 生成图例并拼接
        legend = self._generate_legend_room()
        if img.shape[0] > legend.shape[0]:
            padding = np.zeros((img.shape[0] - legend.shape[0], legend.shape[1], legend.shape[2]), dtype=legend.dtype)
            legend = np.vstack([legend, padding])
        elif img.shape[0] < legend.shape[0]:
            legend = legend[:img.shape[0]]

        final_img = np.hstack([img, legend])

        return final_img


    def _metric_to_pixel(self, pt: np.ndarray) -> np.ndarray:
        """Converts a metric coordinate to a pixel coordinate"""
        # Need to flip y-axis because pixel coordinates start from top left
        px = pt * self._pixels_per_meter * np.array([-1, -1]) + self._origin_in_img
        # px = pt * self._pixels_per_meter + self._origin_in_img
        px = px.astype(np.int32)
        return px
    
    def _generate_colorbar(self, min_val: float, max_val: float) -> np.ndarray:
        """
        生成颜色条，确保和点的颜色匹配
        :param min_val: 最小值
        :param max_val: 最大值
        :return: 颜色条图像
        """
        colorbar_height = 1000
        colorbar_width = 50
        colorbar_img = np.ones((colorbar_height, colorbar_width, 3), dtype=np.uint8) * 255

        # 生成从 max_val 到 min_val 的数值（翻转顺序）
        if max_val == min_val:
            norm_vals = np.zeros((colorbar_height,), dtype=np.uint8)
        else:
            norm_vals = ((np.linspace(min_val, max_val, colorbar_height)[::-1] - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # 进行颜色映射
        norm_vals_img = norm_vals.reshape(-1, 1)  # (H,1)
        colorbar_colors = cv2.applyColorMap(norm_vals_img, cv2.COLORMAP_JET)  # (H,1,3)

        # 复制到整个 colorbar 区域
        colorbar_img[:] = colorbar_colors

        # 添加数值标记（从下到上）
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        cv2.putText(colorbar_img, f'{min_val:.2f}', (5, colorbar_height - 10), font, scale, (0, 0, 0), thickness)
        cv2.putText(colorbar_img, f'{max_val:.2f}', (5, 25), font, scale, (0, 0, 0), thickness)

        return colorbar_img
