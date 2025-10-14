import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage.measure import label, regionprops

class PointLabelRegionProcessor:
    def __init__(self, cell_size=0.25, min_area=50, min_overlap_area=10):
        self.cell_size = cell_size
        self.min_area = min_area
        self.min_overlap_area = min_overlap_area
        self.origin = None  # 原始坐标系左上角
        self.canvas_shape = None
        self.label_polygons = {}

    def to_grid(self, points):
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        self.origin = (min_x, min_y)
        width = int(np.ceil((max_x - min_x) / self.cell_size)) + 1
        height = int(np.ceil((max_y - min_y) / self.cell_size)) + 1
        self.canvas_shape = (height, width)

        grid_coords = np.floor((points - [min_x, min_y]) / self.cell_size).astype(np.int32)
        return grid_coords

    def create_label_masks(self, grid_coords, labels):
        masks = {}
        for rid in np.unique(labels):
            if rid == 0:
                continue
            mask = np.zeros(self.canvas_shape, dtype=np.uint8)
            coords = grid_coords[labels == rid]
            for y, x in coords:
                mask[y, x] = 1
            masks[rid] = mask
        return masks

    def clean_mask(self, mask):
        cleaned = np.zeros_like(mask)
        labeled = label(mask)
        for region in regionprops(labeled):
            if region.area >= self.min_area:
                for coord in region.coords:
                    cleaned[coord[0], coord[1]] = 1
        # 闭运算填空洞
        cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return cleaned

    def get_convex_polygon(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        if len(hull) >= 3:
            return Polygon(hull.squeeze())
        return None

    def process(self, points, labels):
        grid_coords = self.to_grid(points)
        masks = self.create_label_masks(grid_coords, labels)

        # 获取多边形并去噪
        self.label_polygons.clear()
        for rid, mask in masks.items():
            cleaned = self.clean_mask(mask)
            poly = self.get_convex_polygon(cleaned)
            if poly is not None and poly.area > 0:
                self.label_polygons[rid] = poly

        self.resolve_overlap_keep_larger()
        return self.label_polygons

    def resolve_overlap_keep_larger(self):
        ids = list(self.label_polygons.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                poly1 = self.label_polygons.get(id1)
                poly2 = self.label_polygons.get(id2)
                if poly1 is None or poly2 is None:
                    continue
                if poly1.intersects(poly2):
                    inter = poly1.intersection(poly2)
                    if inter.area > self.min_overlap_area:
                        if poly1.area >= poly2.area:
                            self.label_polygons[id2] = poly2.difference(inter)
                        else:
                            self.label_polygons[id1] = poly1.difference(inter)

    def draw_polygons(self):
        canvas = np.zeros(self.canvas_shape, dtype=np.uint8)
        for rid, poly in self.label_polygons.items():
            if not poly.is_empty:
                pts = np.array(poly.exterior.coords).astype(np.int32)
                cv2.fillPoly(canvas, [pts], int(rid))
        return canvas

    def get_world_polygon(self, poly):
        """将 polygon 从像素坐标还原为世界坐标（米）"""
        offset = np.array(self.origin)
        return np.array(poly.exterior.coords) * self.cell_size + offset