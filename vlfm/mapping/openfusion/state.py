import numpy as np
import torch
from torch.nn import functional as F
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c
from scipy.spatial import ConvexHull
import scipy
import matplotlib.pyplot as plt
import bisect
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, GeometryCollection
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from mpl_toolkits.axes_grid1 import make_axes_locatable
import alphashape
from shapely.geometry import MultiPolygon, Polygon
import cv2
try:
    import torchshow as ts
    import time
    from openfusion.utils import rand_cmap
    DBG = False
except:
    print("[*] torchshow not found")


class BaseState(object):
    def __init__(
        self,
        intrinsic,
        depth_scale,
        depth_max,
        depth_min,
        voxel_size = 5.0 / 512,
        block_resolution = 8,
        block_count = 100000,
        device = "CUDA:0",
        img_size=(640, 480),
        scale_num = [1,2,4]
    ) -> None:
        self.img_size = img_size
        print("self.img_size: ",self.img_size)
        self.device = o3c.Device(device)
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.depth_min = depth_min
        self.voxel_size = voxel_size
        self.trunc = self.voxel_size * 4
        self.block_resolution = block_resolution
        self.intrinsic_np = intrinsic
        self.intrinsic_np[2,2] = 1
        self.intrinsic = o3c.Tensor.from_numpy(intrinsic)
        self.scale_number = scale_num
        self.block_count = block_count
        self.world = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'color'),
            (o3c.float32, o3c.float32, o3c.float32),
            ((1), (1), (3)),
            self.voxel_size,
            self.block_resolution,
            block_count, device=self.device
        )
        init_capacity = 1000  # Preallocate space for 1000 elements
        key_dtype = o3d.core.Dtype.Int32  # Keys are integers (for 3D voxel indices)
        key_element_shape = o3d.core.SizeVector([3])  # Each key is a 3D coordinate (x, y, z)

        value_dtype = o3d.core.Dtype.Float32  # Values are float scalars
        value_element_shape = o3d.core.SizeVector([1])  # Each value has a single float

        device = o3d.core.Device("CPU:0")  # Use CPU (Change to "CUDA:0" for GPU)

        # Initialize the HashMap
        self.conf_hash_map = o3c.HashMap(
            block_count * 1000, o3c.int32, o3d.core.SizeVector([3]), o3c.float32, o3d.core.SizeVector([1]), device=self.device
        )
        self.hash_scale_score = o3c.HashMap(
            self.block_count * 1000, o3c.int32, o3d.core.SizeVector([3]), o3c.float32, o3d.core.SizeVector([3]), device=self.device
        )
        print("capacity: ",self.conf_hash_map.capacity())
        print("block_resolution: ",block_resolution, "self.voxel_size: ",self.voxel_size," block_count: ",block_count)
        self.rgb_buffer = []
        self.depth_buffer = []
        self.poses_buffer = []
        self.poses = []
        
        
    
    def custom_intrinsic(self, w, h):
        """ rescales intrinsic matrix according to new image size
        Args:
            w (int): new width
            h (int): new height
        """
        intrinsic = self.intrinsic_np.copy()
        w0, h0 = self.img_size
        intrinsic[0] *= (w / w0)
        intrinsic[1] *= (h / h0)
        return o3c.Tensor.from_numpy(intrinsic)

    def save(self, path):
        self.world.save(path)
        data = np.load(path)
        np.savez(
            path,
            intrinsic = self.intrinsic_np,
            extrinsic = np.array(self.poses),
            **data
        )

    def load(self, path):
        self.world = self.world.load(path)
        data = np.load(path)
        self.intrinsic_np = data["intrinsic"]
        self.intrinsic = o3c.Tensor.from_numpy(self.intrinsic_np)
        self.poses = data["extrinsic"].tolist()

    def append(self, rgb, depth, extrinsic):
        self.rgb_buffer.append(rgb)
        self.depth_buffer.append(depth)
        self.poses_buffer.append(extrinsic)

    def get(self, bs=1):
        if len(self.rgb_buffer) < bs:
            return None, None, None
        if bs == 1:
            pose = self.poses_buffer.pop(0)
            self.poses.append(pose)
            return [self.rgb_buffer.pop(0),], [self.depth_buffer.pop(0),], [pose,]
        if bs > len(self.rgb_buffer):
            bs = len(self.rgb_buffer)
        rgb = [self.rgb_buffer.pop(0) for _ in range(bs)]
        depth = [self.depth_buffer.pop(0) for _ in range(bs)]
        poses = [self.poses_buffer.pop(0) for _ in range(bs)]
        self.poses.extend(poses)
        return rgb, depth, poses

    def get_last_pose(self):
        return self.poses[-1]

    def get_mesh(self, legacy=True):
        mesh = self.world.extract_triangle_mesh()
        return mesh.to_legacy() if legacy else mesh

    def get_pc(self, n=-1):
        if len(self.poses) < 1:
            return None, None
        pcd = self.world.extract_point_cloud()
        points = pcd.point.positions.cpu().numpy()
        colors = pcd.point.colors.cpu().numpy()
        if n > 0 and len(points) > n:
            sample_idx = np.random.choice(len(points), n)
            points = points[sample_idx]
            colors = colors[sample_idx]
        return points, colors

    def get_og2d(self, robot_height=0.72, camera_height=0.38, grid_size=0.02):
        """get 2D occupancy grid of the world
        Args:
            robot_height (float, optional): clearance to be considered in [m]. Defaults to 0.75.
            camera_height (float, optional): height of camera from ground in [m]. Defaults to 0.45.
        """
        pcd = self.world.extract_point_cloud().to_legacy()
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=grid_size)
        ux, _, uz = pcd.get_max_bound()
        lx, _, lz = pcd.get_min_bound()
        x_ = np.arange(lx, ux, 0.1)
        y_ = np.arange(camera_height-robot_height, camera_height-0.1, 0.05)
        z_ = np.arange(lz, uz, 0.1)
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        queries = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        output = np.array(voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))).reshape(x.shape)
        return np.any(output, axis=1), ((lx, ux), (lz, uz))

    def get_pos_in_og2d(self, lims=((0,0),(0,0)), pose=None, pos=None):
        """get 2d position in occupancy grid
        Args:
            lims (tuple, optional): limits of the occupancy grid (returned by get_og2d).
            pose (list, optional): camera pose
            pos (list, optional): point in world frame
        """
        if pos is None:
            if pose is None:
                pose = self.get_last_pose()
            pos = np.linalg.inv(pose)
        x, z = pos[0][3], pos[2][3]
        x_ = np.arange(*lims[0], 0.1)
        z_ = np.arange(*lims[1], 0.1)
        return (
            bisect.bisect_right(z_, z),
            bisect.bisect_right(x_, x),
        )

    @staticmethod
    def depth_to_point_cloud(depth, extrinsic, intrinsic, image_width, image_height, depth_max, depth_scale, observed_mask = None,embedding_keys = None):
        """
        Args:
            depth (np.array): depth image
            extrinsic (o3c.Tensor): shape of (4, 4)
            intrinsic (o3c.Tensor): shape of (3, 3). Use self.custom_intrinsic(image_width, image_height)
            image_width (int): image width
            image_height (int): image height
            depth_max (float): depth max
            depth_scale (float): depth scale
        Returns:
            coords (torch.Tensor): shape of (N, 3)
            mask (torch.Tensor): shape of (H, W)
        """
        depth = torch.from_numpy(depth.astype(np.float32)) / depth_scale
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0).float(),
            (image_height, image_width)
        ).view(image_height, image_width).cuda()
        extrinsic = torch.utils.dlpack.from_dlpack(extrinsic.to_dlpack()).cuda().float()
        intrinsic = torch.utils.dlpack.from_dlpack(intrinsic.to_dlpack()).cuda().float()
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

        v, u = torch.meshgrid(torch.arange(image_height).cuda(), torch.arange(image_width).cuda(), indexing="ij")
        uvd = torch.stack([u, v, torch.ones_like(depth)], dim=0).float() # (3,H,W)
        # NOTE: don't use torch.inverse(intrinsic) as it is slow
        uvd[0] = (uvd[0] - cx) / fx
        uvd[1] = (uvd[1] - cy) / fy
        xyz = uvd.view(3, -1) * depth.view(1, -1) # (3, H*W)
        # NOTE: convert to world frame
        R = extrinsic[:3, :3].T
        coords =  (R @ xyz - R @ extrinsic[:3, 3:]).view(3, image_height, image_width).permute(1,2,0)
        if observed_mask is not None:
            observed_mask = observed_mask.to(depth.device)
            observed_mask = observed_mask == 0
            mask = [(0 < depth) & (depth <= depth_max) & observed_mask]
        else:
            mask = [(0 < depth) & (depth < depth_max)]
        
        # TODO: check 0.05 offset for +y direction (up)
        if embedding_keys is None:
            return coords[mask], mask
        else:
            coords_copy = coords.clone()
            volume_dict = {}
            density_dict = {}
            embedding_keys = torch.tensor(embedding_keys.copy())
            unique_ids = torch.unique(embedding_keys)
            for id_value in unique_ids:
                mask_region = torch.any(embedding_keys == id_value, dim=-1)  
                com_mask = mask_region & mask[0].cpu()
                croped_point_cloud = coords_copy[com_mask]
                points_np = croped_point_cloud.cpu().numpy()
                try:
                    hull = ConvexHull(points_np)
                    volume = hull.volume
                    density = points_np.shape[0]/hull.volume
                except Exception as e:  
                    print(f"Error in compute_volume_and_density: {str(e)}")
                    volume = 0
                    density = 0 

                volume_dict[id_value] = volume
                density_dict[id_value] = density
            return coords[mask], mask, volume_dict, density_dict
    @staticmethod
    def get_points_in_fov(coords, extrinsic, intrinsic, image_width, image_height, depth_max):
        """
        Args:
            coords (o3c.Tensor): shape of (N, 3)
            extrinsic (o3c.Tensor): shape of (4, 4)
            intrinsic (o3c.Tensor): shape of (3, 3). Use self.custom_intrinsic(image_width, image_height)
            image_width (int): width of the image
            image_height (int): height of the image
            depth_max (float): depth max
        Returns:
            v_proj (torch.Tensor): shape of (M)
            u_proj (torch.Tensor): shape of (M)
            d_proj (torch.Tensor): shape of (M)
            mask_proj (torch.Tensor): shape of (N)
        """
        coords = torch.utils.dlpack.from_dlpack(coords.to_dlpack()).cuda().float()
        extrinsic = torch.utils.dlpack.from_dlpack(extrinsic.to_dlpack()).cuda().float()
        intrinsic = torch.utils.dlpack.from_dlpack(intrinsic.to_dlpack()).cuda().float()
        # NOTE: apply camera pose
        xyz = extrinsic[:3, :3] @ coords.T + extrinsic[:3, 3:]
        # NOTE: perform projection using the camera intrinsic matrix (W,H,D)
        uvd = intrinsic @ xyz
        d = uvd[2]
        # NOTE: divide by third coordinate to obtain 2D pixel locations
        u = (uvd[0] / d).long() # W
        v = (uvd[1] / d).long() # H
        # NOTE: filter out points outside the image plane (outside FoV)
        mask_proj = (depth_max > d) & (
            (d > 0) &
            (u >= 0) &
            (v >= 0) &
            (u < image_width) &
            (v < image_height)
        )
        v_proj = v[mask_proj] # H
        u_proj = u[mask_proj] # W
        d_proj = d[mask_proj] # D

        return v_proj, u_proj, d_proj, mask_proj

    def active_buf_indices(self):
        # Find all active buf indices in the underlying engine
        buf_indices = self.world.hashmap().active_buf_indices()
        return buf_indices

    def active_buf_indices_in_fov(self, extrinsic, width, height, device):
        pcd = self.world.extract_point_cloud()
        pcd_cpu = pcd.to(o3d.core.Device("CPU:0")) 
        pcd_legacy = o3d.t.geometry.PointCloud.to_legacy(pcd_cpu)
        o3d.visualization.draw_geometries([pcd_legacy],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
        points = pcd.point.positions
        _, _, _, mask_proj = self.get_points_in_fov(
            points, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
        )
        pcd_ = o3d.t.geometry.PointCloud(device)
        pcd_.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(
            torch.utils.dlpack.from_dlpack(points.to_dlpack()).float()[mask_proj]
        )).to(device)
        frustum_block_coords = self.world.compute_unique_block_coordinates(
            pcd_, trunc_voxel_multiplier=8.0
        )
        buf_indices, _ = self.world.hashmap().find(frustum_block_coords)
        o3c.cuda.synchronize()
        return buf_indices

    def buf_coords(self, buf_indices):
        voxel_coords, _ = self.world.voxel_coordinates_and_flattened_indices(
            buf_indices
        )
        buf_coords = voxel_coords.reshape((-1, self.block_resolution**3, 3)).mean(1)
        return buf_coords

    def update(self, color, depth, extrinsic):
        color = o3d.t.geometry.Image(color).to(o3c.float32).to(self.device)
        depth = o3d.t.geometry.Image(depth).to(o3c.float32).to(self.device)
        extrinsic = o3c.Tensor.from_numpy(extrinsic)
        # Get active frustum block coordinates from input
        frustum_block_coords = self.world.compute_unique_block_coordinates(
            depth, self.intrinsic, extrinsic, self.depth_scale, self.depth_max, trunc_voxel_multiplier=self.block_resolution
        )
        self.world.integrate(
            frustum_block_coords, depth, color, self.intrinsic,
            extrinsic, self.depth_scale, self.depth_max
        )
        return frustum_block_coords, extrinsic

    def take_snapshot(self, height, width, extrinsic, show=False, save_path=None, level="voxel"):
        assert level in ["block", "voxel", "pc"]
        img = torch.zeros(height, width, 3, dtype=torch.uint8)

        if level in ["block", "voxel"]:
            buf_indices = self.world.hashmap().active_buf_indices()
            voxel_coords, voxel_indices = self.world.voxel_coordinates_and_flattened_indices(
                buf_indices
            )
            o3c.cuda.synchronize()
            if level == "block":
                buf_coords = voxel_coords.reshape((-1, self.block_resolution**3, 3)).mean(1)
                v_proj, u_proj, _, mask_proj = self.get_points_in_fov(
                    buf_coords, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
                )
                color = self.world.attribute('color').reshape((-1, self.block_resolution**3, 3)).mean(1)
                indices = buf_indices.cpu().numpy()[mask_proj.cpu().numpy()]
            elif level == "voxel":
                v_proj, u_proj, _, mask_proj = self.get_points_in_fov(
                    voxel_coords, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
                )
                color = self.world.attribute('color').reshape((-1, 3))
                indices = voxel_indices.cpu().numpy()[mask_proj.cpu().numpy()]
            color = torch.utils.dlpack.from_dlpack(color.to_dlpack()).cpu()
            v_proj = v_proj.cpu()
            u_proj = u_proj.cpu()

            unique_indices, inverse_indices = torch.unique(v_proj * width + u_proj, return_inverse=True)
            sum_colors = torch.zeros_like(unique_indices, dtype=torch.float32).repeat(3,1).T
            sum_colors.index_add_(0, inverse_indices, color[indices])
            counts = torch.bincount(inverse_indices, minlength=len(unique_indices))
            avg_colors = sum_colors / counts.unsqueeze(1)
            img[(unique_indices // width), (unique_indices % width)] = avg_colors.to(torch.uint8)
        else:
            pcd = self.world.extract_point_cloud()
            points = pcd.point.positions
            color = pcd.point.colors * 255
            v_proj, u_proj, _, mask_proj = self.get_points_in_fov(
                points, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
            )
            v_proj, u_proj, indices = v_proj.cpu().numpy(), u_proj.cpu().numpy(), mask_proj.cpu().numpy()
            color = torch.utils.dlpack.from_dlpack(color.to_dlpack()).cpu()
            img[v_proj, u_proj] = color[indices].to(torch.uint8)

        if show or save_path is not None:
            ts.show([img], save=save_path is not None, file_path=save_path)
        return img

    @torch.no_grad()
    def fast_object_query(self, t_emb, points, colors=None, only_poi=False, topk=1, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def object_query(self, t_emb, points, colors=None, only_poi=False, topk=1, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def semantic_query(self, t_emb, points, colors=None, cmap=None):
        raise NotImplementedError

    @torch.no_grad()
    def instance_query(self, t_emb, points, colors=None, cmap=None):
        raise NotImplementedError

    @torch.no_grad()
    def panoptic_query(self, t_emb, points, colors=None, cmap=None):
        raise NotImplementedError

class CFState(BaseState):
    """ Point (Block) wise embedding fusion
    """
    def __init__(
        self,
        intrinsic,
        depth_scale,
        depth_max,
        depth_min,
        voxel_size = 5.0 / 512,
        block_resolution = 8,
        block_count = 100000,
        dim = 512,
        scales = 3,
        device = "CUDA:0",
        img_size=(640, 480),
        semantic_weight = 1,
        explore_weight = 0.01
    ) -> None:
        super().__init__(
            intrinsic, depth_scale, depth_max, depth_min, voxel_size,
            block_resolution, block_count, device, img_size
        )
        # store embeddings on each block to avoid GPU memory overflow
        self.emb_num = 100000
        self.scales = scales
        self.dim = dim
        self.emb_keys = torch.zeros(block_count, scales).to(torch.int64)
        self.embedding_book = torch.zeros(self.emb_num,dim)
        self.count = torch.zeros(self.emb_num)
        self.image_count = 0
        self.semantic_gain_weight = semantic_weight
        self.explore_gain_weight = explore_weight
        print("block_count: ",block_count)
    
    def reset(self):
        print("reset the map")
        del self.world
        del self.conf_hash_map
        del self.hash_scale_score
        del self.emb_keys
        del self.embedding_book
        del self.count
        del self.image_count
        self.compute_count = 0
        self.agent_height = 0
        self.world = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'color'),
            (o3c.float32, o3c.float32, o3c.float32),
            ((1), (1), (3)),
            self.voxel_size,
            self.block_resolution,
            self.block_count, device=self.device
        )
        self.emb_keys = torch.zeros(self.block_count, self.scales).to(torch.int64)
        self.embedding_book = torch.zeros(self.emb_num, self.dim)
        self.count = torch.zeros(self.emb_num)
        self.image_count = 0
        self.conf_hash_map = o3c.HashMap(
            self.block_count * 1000, o3c.int32, o3d.core.SizeVector([3]), o3c.float32, o3d.core.SizeVector([1]), device=self.device
        )
        self.hash_scale_score = o3c.HashMap(
            self.block_count * 1000, o3c.int32, o3d.core.SizeVector([3]), o3c.float32, o3d.core.SizeVector([3]), device=self.device
        )
        torch.cuda.empty_cache()
        
    def adjust_embed_capacity(self):
        if self.world.hashmap().capacity() > self.emb_keys.shape[0]:
            delta = self.world.hashmap().capacity() - self.emb_keys.shape[0]
            self.emb_keys = torch.cat([self.emb_keys, torch.zeros(delta,self.scales)], dim=0).to(torch.int64)

    def voxels_in_fov(self,obs_coords, extrinsics):
        extrinsics = torch.utils.dlpack.from_dlpack(extrinsics.to_dlpack()).cuda().float()
        R = extrinsics[:3, :3].T  
        cam_origin = (-R @ extrinsics[:3, 3:]).squeeze()  
        directions = obs_coords - cam_origin
        distances = torch.norm(directions, dim=1)  
        directions /= distances[:, None]  


        num_steps = (distances / self.voxel_size).ceil().long()  
        max_steps = num_steps.max().item()  
        step_sizes = torch.arange(0, max_steps + 1, dtype=torch.float32).to(obs_coords.device) * self.voxel_size  

        
        rays = cam_origin + directions[:, None, :] * step_sizes[None, :, None]  # (N, S, 3)
        

        mask = step_sizes[None, :] <= distances[:, None]  
        valid_rays = rays[mask]  

        
        observed_voxel_coords = torch.floor(valid_rays / self.voxel_size).to(torch.int32)
        
        edge_voxel_coords = torch.floor(obs_coords / self.voxel_size).to(torch.int32)
        
        unique_observed_voxel_coords = torch.unique(observed_voxel_coords, dim=0)
        unique_edge_voxel_coords = torch.unique(edge_voxel_coords, dim=0)
        mask = ~((unique_observed_voxel_coords.unsqueeze(1) == unique_edge_voxel_coords.unsqueeze(0)).all(dim=2).any(dim=1))
    
        
        unique_observed_voxel_coords = unique_observed_voxel_coords[mask.to(unique_observed_voxel_coords.device)]
        return unique_observed_voxel_coords
    
    def compute_confidence_from_depth_torch(self,
    depth_image,  
    fx, fy, cx, cy,
    best_distance = [1.0,1.5],
    min_distance = 0.1,
    max_distance = 10.0,
    alpha=0.2,
    gamma=0.05,
    w_d=0.8,
    w_yaw=0.1,
    w_pitch=0.1,
    horizontal_fov=None,  
    vertical_fov=None,  
    device="cuda",
    with_yaw_pitch = True,
    without_depth = False
    ): 
        depth_image = depth_image.to(device) / self.depth_scale
        H, W = depth_image.shape
        if horizontal_fov is None:
            horizontal_fov = 2.0 * torch.atan2(torch.tensor(cx, device=device), torch.tensor(fx, device=device))
        if vertical_fov is None:
            vertical_fov = 2.0 * torch.atan2(torch.tensor(cy, device=device), torch.tensor(fy, device=device))

        hfov_rad = horizontal_fov / 2.0 
        vfov_rad = vertical_fov / 2.0    
        v_coords = torch.arange(H, device=device).view(H, 1).expand(-1, W)  # (H, W)
        u_coords = torch.arange(W, device=device).view(1, W).expand(H, -1)  # (H, W)

        valid_mask = depth_image > 0  
        valid_v_proj = v_coords[valid_mask]  # (N,)
        valid_u_proj = u_coords[valid_mask]  # (N,)
        valid_d_proj = depth_image[valid_mask]  # (N,)


        Z = valid_d_proj  
        X = (valid_v_proj - cy) / fy * Z  
        Y = (cx - valid_u_proj) / fx * Z  
        distance = torch.sqrt(X**2 + Y**2 + Z**2)  
        yaw = torch.atan2(Y, Z)  
        pitch = torch.atan2(X, torch.hypot(Y, Z))  
        C_yaw = torch.cos((yaw / hfov_rad) * (torch.pi / 2)) ** 2
        C_pitch = torch.cos((pitch / vfov_rad) * (torch.pi / 2)) ** 2
        C_yaw[torch.abs(yaw) > hfov_rad] = 0
        C_pitch[torch.abs(pitch) > vfov_rad] = 0
        best_distance_mask = (distance > best_distance[0]) & (distance < best_distance[1])
        decay_values = torch.exp(-alpha * torch.min((distance - best_distance[0]) ** 2, 
                                                    (distance - best_distance[1]) ** 2))
        C_d = torch.where(best_distance_mask, torch.ones_like(distance, device=distance.device), decay_values)

        C_d[distance > max_distance] = 0  
        C_d[distance < min_distance] = 0
        if with_yaw_pitch:
            confidence = torch.clamp(C_d * C_yaw * C_pitch, 0, 1)
        else:
            confidence = torch.clamp(C_d, 0, 1)
        confidence_map = torch.zeros((H, W), dtype=torch.float32, device=device)
        confidence_map[valid_mask] = confidence
        if without_depth:
            confidence_without_depth = torch.clamp(C_yaw * C_pitch, 0, 1)
            confidence_map_without_depth = torch.zeros((H, W), dtype=torch.float32, device=device)
            confidence_map_without_depth[valid_mask] = confidence_without_depth
            return confidence_map, confidence_map_without_depth
        else:
            return confidence_map
    
    def update_embedding_book(self, new_embeddings, cur_keys, similarity_threshold=0.2, update_rate=1.0):
        change_dict = {}
        device = self.embedding_book.device
        M, C = new_embeddings.shape
        N = self.embedding_book.shape[0]

        new_embeddings = new_embeddings.to(device)
        pos_list = []  
        if cur_keys is not None:
            if not isinstance(cur_keys, torch.Tensor):
                cur_keys = torch.tensor(cur_keys, dtype=torch.int64, device=self.embedding_book.device)
            mask = torch.zeros(self.embedding_book.size(0), device=self.embedding_book.device, dtype=torch.bool)
            mask.index_fill_(0, cur_keys, True)
            self.embedding_book *= mask.unsqueeze(1).to(self.embedding_book.dtype)
            self.count *= mask
        free_mask = torch.all(self.embedding_book == 0, dim=1)  # shape: (N,)
        free_idx = torch.nonzero(free_mask, as_tuple=False).squeeze(1)  # 1D tensor
        free_idx, _ = torch.sort(free_idx)
        free_idx = free_idx[free_idx !=0]
        num_free = free_idx.numel()
        if num_free >= M:
            self.embedding_book[free_idx[:M]] = new_embeddings
            self.count[free_idx[:M]] += 1
            pos_list = free_idx[:M]
            return pos_list,{}

        self.embedding_book[free_idx] = new_embeddings[:num_free]
        self.count[free_idx] += 1
        pos_list.append(free_idx)  

        remain_new = new_embeddings[num_free:]  # shape: (R, C)
        R = remain_new.shape[0]
        for i in range(R):
            new_emb = remain_new[i]
            new_emb_norm = F.normalize(new_emb.unsqueeze(0), p=2, dim=1)  # (1, C)
            book_norm = F.normalize(self.embedding_book, p=2, dim=1)               # (N, C)
            cos_sim = torch.mm(book_norm, new_emb_norm.t()).squeeze(1)         # (N,)
            max_sim, max_idx = torch.max(cos_sim, dim=0)
            if max_sim >= similarity_threshold:
                self.embedding_book[max_idx] = (self.count[max_idx] * self.embedding_book[max_idx] + new_emb) / (self.count[max_idx] + 1)
                self.count[max_idx] += 1
                pos_list.append(torch.tensor([max_idx], device=device))
            else:
                norm_book = F.normalize(self.embedding_book, p=2, dim=1)  # (N, C)
                sim_matrix = torch.mm(norm_book, norm_book.t())         # (N, N)
                sim_matrix.fill_diagonal_(-float('inf'))
                row_max, col_indices = torch.max(sim_matrix, dim=1)
                overall_max, i_star = torch.max(row_max, dim=0)
                j_star = col_indices[i_star]
                sim_i = F.cosine_similarity(new_emb.unsqueeze(0), self.embedding_book[i_star].unsqueeze(0))
                sim_j = F.cosine_similarity(new_emb.unsqueeze(0), self.embedding_book[j_star].unsqueeze(0))
                if sim_i >= sim_j:
                    self.embedding_book[i_star] = new_emb
                    pos_list.append(torch.tensor([i_star], device=device))
                    change_dict.update({i_star:j_star})
                else:
                    self.embedding_book[j_star] = new_emb
                    pos_list.append(torch.tensor([j_star], device=device))
                    change_dict.update({j_star:i_star})

        new_positions = torch.cat(pos_list, dim=0)
        return new_positions, change_dict
    
    def visualize_voxels_and_points(self, points_world, unique_voxel_coords):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        sample_ratio = 0.01
        num_points = points_world.shape[0]
        sample_size_points = max(1, int(num_points * sample_ratio))  
        sampled_indices_points = torch.randperm(num_points)[:sample_size_points]  
        sampled_points = points_world[sampled_indices_points] 

        num_voxels = unique_voxel_coords.shape[0]
        sample_size_voxels = max(1, int(num_voxels * 1))  
        sampled_indices_voxels = torch.randperm(num_voxels)[:sample_size_voxels]  
        sampled_voxels = unique_voxel_coords[sampled_indices_voxels] * self.voxel_size 

        sampled_points_np = sampled_points.cpu().numpy()  
        sampled_voxels_np = sampled_voxels.cpu().numpy()  

        ax.scatter(sampled_points_np[:, 0], sampled_points_np[:, 1], sampled_points_np[:, 2], 
                c='r', marker='o', s=10, label="Sampled Original Points")  
        
        ax.scatter(sampled_voxels_np[:, 0], sampled_voxels_np[:, 1], sampled_voxels_np[:, 2], 
                c='k', marker='s', s=5, label="Sampled Observed Voxels")  

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("3D Visualization of Sampled Observed Voxels and Points")
        ax.legend()
        plt.show()

    def visualize_sampled_points(self, points_world, values,intrinsic,width,height,extrinsic,near = 0.1, far = 2, sample_ratio=0.1, save_path=None, colormap="viridis"):
        """
        仅对 points_world 进行采样，并基于 values (0~1) 着色点云。

        参数:
            points_world (torch.Tensor): 形状 `(N, 3)`，表示原始点云。
            values (torch.Tensor): 形状 `(N,)`，范围 [0,1]，控制点云的颜色映射。
            sample_ratio (float): 采样比例，默认 1%。
            save_path (str): 选填，若提供则保存图像，否则仅显示。
            colormap (str): Matplotlib 颜色映射，默认 "viridis"。
        """
        extrinsic = torch.utils.dlpack.from_dlpack(extrinsic.to_dlpack()).cpu().numpy()
        intrinsic = torch.utils.dlpack.from_dlpack(intrinsic.to_dlpack()).cpu().numpy()
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        num_points = points_world.shape[0]
        sample_size_points = max(1, int(num_points * sample_ratio))  
        sampled_indices_points = torch.randperm(num_points)[:sample_size_points]  
        sampled_points = points_world[sampled_indices_points] * self.voxel_size  
        sampled_values = values[sampled_indices_points]  
        sampled_points_np = sampled_points.cpu().numpy()
        sampled_values_np = sampled_values.cpu().numpy()
        cmap = plt.get_cmap(colormap)  
        point_colors = cmap(sampled_values_np)[:, :3]  
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        near_corners = np.array([
            [(0 - cx) * near / fx, (0 - cy) * near / fy, near],   
            [(width - cx) * near / fx, (0 - cy) * near / fy, near],  
            [(width - cx) * near / fx, (height - cy) * near / fy, near],  
            [(0 - cx) * near / fx, (height - cy) * near / fy, near],  
        ])
        far_corners = np.array([
            [(0 - cx) * far / fx, (0 - cy) * far / fy, far],  
            [(width - cx) * far / fx, (0 - cy) * far / fy, far],  
            [(width - cx) * far / fx, (height - cy) * far / fy, far],  
            [(0 - cx) * far / fx, (height - cy) * far / fy, far],  
        ])
        frustum_corners = np.vstack([near_corners, far_corners]) 

        R = extrinsic[:3, :3].T  
        t = (-R @ extrinsic[:3, 3:]).squeeze() 
        frustum_corners = (R @ frustum_corners.T).T + t  
        print("frustum_corners: ",frustum_corners.shape)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-1, 4)
        
        ax.set_box_aspect([1, 1, 1])  
        cam_pos = t
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='r', marker='o', s=50, label="Camera")
        edges = [
            [frustum_corners[0], frustum_corners[1]], [frustum_corners[1], frustum_corners[2]],
            [frustum_corners[2], frustum_corners[3]], [frustum_corners[3], frustum_corners[0]],  
            [frustum_corners[4], frustum_corners[5]], [frustum_corners[5], frustum_corners[6]],
            [frustum_corners[6], frustum_corners[7]], [frustum_corners[7], frustum_corners[4]],  
            [frustum_corners[0], frustum_corners[4]], [frustum_corners[1], frustum_corners[5]],
            [frustum_corners[2], frustum_corners[6]], [frustum_corners[3], frustum_corners[7]]  
        ]


        for edge in edges:
            edge = np.array(edge)  
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color='r', linewidth=1)
        scatter = ax.scatter(sampled_points_np[:, 0], sampled_points_np[:, 1], sampled_points_np[:, 2], 
                c=point_colors, marker='o', s=10, label="Sampled Points")
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Value')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("3D Visualization of Sampled Points with Value-based Coloring")
        ax.legend()
    
    def find_buf_indices_from_coord(self, buf_indices, voxel_coords, coordinates):
        """
        Finds the index of the cube that incorporates each coordinate in a batched manner.

        Args:
            buf_indices (torch.Tensor): N tensor of buf indices for the given voxel coords
            voxel_coords (torch.Tensor): Nx8^3x3 tensor where N is the number of cubes.
            coordinates (torch.Tensor): Mx3 tensor where M is the number of coordinates. (usually M >> N)

        Returns:
            tensor: M tensor that contains the index of the cube that incorporates each coordinate.
        """
        # NOTE: find min and max of x, y, z for each cube
        min_vals = voxel_coords[:, 0, :]  - self.voxel_size/2 # Shape: Nx3
        max_vals = voxel_coords[:, -1, :] + self.voxel_size/2 # Shape: Nx3
        # NOTE: check if each coordinate is inside each cube
        is_inside = (min_vals[:, None] <= coordinates[None]) & (coordinates[None] < max_vals[:, None]) # Shape: NxMx3
        
        # NOTE: all coordinates must be inside the cube along all 3 dimensions (x, y, z)
        is_inside_all_dims = torch.all(is_inside, dim=2).long()  # Shape: NxM
        # NOTE: find cube index for each coordinate
        cube_idx_for_each_coord = buf_indices[torch.argmax(is_inside_all_dims, dim=0)]  # Shape: M
        # NOTE: find valid mask where a cube was found for a coordinate
        valid_mask = torch.any(is_inside_all_dims, dim=0)  # Shape: M
        return cube_idx_for_each_coord[valid_mask], valid_mask

    def compute_similarity_and_entropy(self, image_features, text_features):

        image_features = image_features.cuda()
        text_features = text_features.cuda()
        image_features_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # (N, S, 512)
        text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True)  # (M, 512)
        

        text_features_T = text_features_normalized.T  
        score_map = torch.matmul(image_features_normalized, text_features_T)  # (N, S, M)
        entropy_values = []
        N, S, M = score_map.shape
        min_m, min_s = 0, 0
        min_entropy = float('inf')
        for m in range(M):
            for s in range(S):
                prob_map = torch.softmax(score_map[:, s, m], dim=0)  
                entropy = -(prob_map * torch.log(prob_map + 1e-7)).sum()  
                entropy_values.append(entropy.item())  
                if entropy < min_entropy:
                    min_entropy = entropy
                    min_m, min_s = m, s
        return min_s, min_m, score_map[:,min_s,min_m]


    def compute_map_catagories(self,image_features,text_features):
        image_features = image_features.cuda()
        text_features = text_features.cuda()
        _, S, _ = image_features.shape
        M,_ = text_features.shape
        image_features_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # (N, S, 512)
        text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True)  # (M, 512)
        text_features_T = text_features_normalized.T  
        score_map = torch.matmul(image_features_normalized, text_features_T)  
        max_indices = torch.argmax(score_map, dim=2)
        return max_indices

    def most_frequent_indices(self, max_indices):
        """
        在 S 维度上找到出现次数最多的索引，
        如果有多个索引出现次数相同，则选择 S 中第一个出现的值。

        参数:
        - max_indices: (N, S) 形状的张量，每个元素是索引

        返回:
        - most_frequent: (N,) 形状的张量，每个 N 选择一个出现最多的索引
        """
        N, S = max_indices.shape
        unique_vals, inverse_indices = torch.unique(max_indices, return_inverse=True)  
        inverse_indices = inverse_indices.view(N, S)  

        counts = torch.zeros((N, unique_vals.shape[0]), dtype=torch.long, device=max_indices.device)
        counts.scatter_add_(1, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.long))
        max_counts, _ = counts.max(dim=1, keepdim=True)
        mask = counts == max_counts
        first_occurrence = torch.full((N,), S, dtype=torch.long, device=max_indices.device)  
        most_frequent = torch.full((N,), -1, dtype=max_indices.dtype, device=max_indices.device)  

        for j in range(S):  
            mask_at_j = mask.gather(1, inverse_indices[:, j:j+1]).squeeze(1)  
            update_mask = (first_occurrence == S) & mask_at_j 
            first_occurrence[update_mask] = j  
            most_frequent[update_mask] = max_indices[update_mask, j]  
        return most_frequent  # (N,)

    def compute_sum_similarity(self, image_features, text_features, probabilites):
        image_features = image_features.cuda()
        text_features = text_features.cuda()
        probabilites = probabilites.cuda()
        _, S, _ = image_features.shape
        M,_ = text_features.shape
        image_features_norm = image_features.norm(dim=-1, keepdim=True)  
        epsilon = 1e-6
        mask = (image_features_norm == 0)
        image_features_norm = torch.where(mask, torch.ones_like(image_features_norm) * epsilon, image_features_norm)
        image_features_normalized = image_features / image_features_norm  # (N, S, 512)
        text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True)  # (M, 512)
        
        text_features_T = text_features_normalized.T  
        score_map = torch.matmul(image_features_normalized, text_features_T)  # (N, S, M)
        score_weight = probabilites
        score_map, _ = torch.max(score_map, dim=1)
        score_map = torch.sum(score_map * score_weight, dim=1)
        return 0, 0, score_map
    
    def instance_num(self,embedding_keys,object_mask):
        id_mask_count = {}
        embedding_keys = torch.tensor(embedding_keys.copy())
        unique_ids = torch.unique(embedding_keys)
        for id_value in unique_ids:
            mask_region = torch.any(embedding_keys == id_value, dim=-1)  
            object_mask_in_region = object_mask[mask_region]  
            unique_masks = object_mask_in_region.unique()  
            unique_masks = unique_masks[unique_masks != 0]
            id_mask_count[id_value] = len(unique_masks)

        return id_mask_count    
    def get_scene_level_score(self,embedding_keys,object_num_dict,volume_dict,confidence_map_without_depth):
        embedding_keys = torch.tensor(embedding_keys.copy()).to(torch.int64)
        object_num = torch.tensor([object_num_dict[i] for i in object_num_dict.keys()], dtype=torch.float32)
        volume = torch.tensor([volume_dict[i] for i in volume_dict.keys()], dtype=torch.float32)
        sum_values = object_num/20 + volume/10  
        pixel_sums = sum_values[embedding_keys]  
        max_sum_mask, max_id_indices = (pixel_sums*confidence_map_without_depth.to(pixel_sums.device).unsqueeze(-1)).max(dim=-1) 
        max_id = embedding_keys.gather(dim=-1, index=max_id_indices.unsqueeze(-1))  
        return max_id.squeeze(-1), max_sum_mask
    
    def get_area_level_score(self,embedding_keys, object_num_dict, volume_dict, density_dict,confidence_map_without_depth):
        embedding_keys = torch.tensor(embedding_keys.copy()).to(torch.int64)
        object_num = torch.tensor([object_num_dict[i] for i in object_num_dict.keys()], dtype=torch.float32)
        volume = torch.tensor([volume_dict[i] for i in volume_dict.keys()], dtype=torch.float32)
        density = torch.tensor([density_dict[i] for i in density_dict.keys()], dtype=torch.float32)
        volume = torch.where(volume == 0, torch.tensor(float('inf')), volume)
        object_volume = 2 * (object_num)/(volume**0.6)
        density = ((density)**0.5)/100
        sum_values = object_volume + density 
        pixel_sums = sum_values[embedding_keys]  
        max_sum_mask, max_id_indices = (pixel_sums*confidence_map_without_depth.to(pixel_sums.device).unsqueeze(-1)).max(dim=-1)  
        max_id = embedding_keys.gather(dim=-1, index=max_id_indices.unsqueeze(-1))  
        return max_id.squeeze(-1), max_sum_mask
    
    def get_objects_level_score(self,embedding_keys,object_num_dict,density_dict,confidence_map_without_depth):
        embedding_keys = torch.tensor(embedding_keys.copy()).to(torch.int64)
        object_num = torch.tensor([object_num_dict[i] for i in object_num_dict.keys()], dtype=torch.float32)
        object_num = torch.where(object_num == 0, torch.tensor(float('inf')), object_num)
        density = torch.tensor([density_dict[i] for i in density_dict.keys()], dtype=torch.float32)
        sum_values = density/(object_num)  
        pixel_sums = sum_values[embedding_keys]  
        max_sum_mask, max_id_indices = (pixel_sums*confidence_map_without_depth.to(pixel_sums.device).unsqueeze(-1)).max(dim=-1)  
        max_id = embedding_keys.gather(dim=-1, index=max_id_indices.unsqueeze(-1))  
        return max_id.squeeze(-1), max_sum_mask
    
    def update(self, color, depth, extrinsic, agent_height, res_dict:dict={}, observation_range = [1.0,1.5]):
        
        with torch.no_grad():
            self.agent_height = agent_height
            depth = depth * (self.depth_max - self.depth_min) + self.depth_min
            frustum_block_coords, extrinsic = super().update(color, depth, extrinsic)
            
            # NOTE: when the switch is off return without semantic integration
            if not res_dict:
                return
            self.adjust_embed_capacity()
            cur_buf_indices, _ = self.world.hashmap().find(frustum_block_coords)
            voxel_coords, _ = self.world.voxel_coordinates_and_flattened_indices(
                cur_buf_indices
            )
            cur_buf_indices = torch.utils.dlpack.from_dlpack(cur_buf_indices.to_dlpack())
            voxel_coords = torch.utils.dlpack.from_dlpack(voxel_coords.to_dlpack())
            height, width = res_dict["embedding_keys"].shape[:2]
            new_embedding = res_dict["mebbedings"]
            init_embedding_keys = res_dict["embedding_keys"]
            object_mask = res_dict["object_mask"]
            obs_coords, mask, volume_dict, density_dict = self.depth_to_point_cloud(
                    depth, extrinsic, self.custom_intrinsic(width, height),
                    width, height, self.depth_max, self.depth_scale, embedding_keys = init_embedding_keys
                )
            comb_buf_idx, valid = self.find_buf_indices_from_coord(
                    cur_buf_indices,
                    voxel_coords.view(-1, self.block_resolution**3, 3), # (M, 8^3, 3)
                    obs_coords # (N, 3)
                )
            comb_buf_idx ,_ = torch.unique(comb_buf_idx, dim=0, return_inverse=True)
            cur_buf_coords = self.buf_coords(o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(comb_buf_idx)))
            cur_buf_coords_tensor = torch.utils.dlpack.from_dlpack(cur_buf_coords.to_dlpack()).cuda().float()
            v_proj, u_proj, _, mask_proj = self.get_points_in_fov(
                cur_buf_coords, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
            )
            intrinsic_tensor  = torch.from_numpy(self.intrinsic_np)
            depth_torch = torch.from_numpy(depth.astype(np.int32))
            confidence_map,confidence_map_without_depth = self.compute_confidence_from_depth_torch(
            depth_torch,
            fx = self.intrinsic_np[0,0], fy = self.intrinsic_np[1,1], cx = self.intrinsic_np[0,2], cy = self.intrinsic_np[1,2],
            best_distance= observation_range,
            min_distance= 0.0,
            max_distance = self.depth_max * 0.5,
            alpha = 0.25,
            gamma = 0.05,
            w_d = 0.8,
            w_yaw = 0.1,
            w_pitch = 0.10,
            without_depth = True
            )
            cur_buf_coords = torch.utils.dlpack.from_dlpack(cur_buf_coords.to_dlpack())
            cur_buf_indices, cur_buf_coords, v_proj, u_proj = \
                comb_buf_idx[mask_proj], cur_buf_coords[mask_proj], v_proj.cpu(), u_proj.cpu()
            if cur_buf_coords.shape[0] == 0:
                return
            condidance_value = confidence_map[v_proj, u_proj]
            cur_buf_coords_voxel = torch.floor(cur_buf_coords / self.voxel_size).to(torch.int32)
            condidance_value = condidance_value.to(torch.float32)

            current_size = self.conf_hash_map.size()
            object_num = self.instance_num(init_embedding_keys,object_mask)
            scene_id, scene_level_score = self.get_scene_level_score(init_embedding_keys,object_num,volume_dict,confidence_map_without_depth)
            unique_scene_id, counts_scene_id = torch.unique(scene_id, return_counts=True)
            area_id, area_level_score = self.get_area_level_score(init_embedding_keys,object_num,volume_dict,density_dict,confidence_map_without_depth)
            unique_area_id, counts_area_id = torch.unique(area_id, return_counts=True)
            objects_id, objects_level_score = self.get_objects_level_score(init_embedding_keys,object_num,density_dict,confidence_map_without_depth)
            unique_objects_id, counts_objects_id = torch.unique(objects_id, return_counts=True)
            new_embedding_keys = torch.cat((scene_id.unsqueeze(-1), area_id.unsqueeze(-1), objects_id.unsqueeze(-1)), dim=-1)
            new_embedding_score = torch.cat((scene_level_score.unsqueeze(-1), area_level_score.unsqueeze(-1), objects_level_score.unsqueeze(-1)), dim=-1)
            current_keys = torch.unique(self.emb_keys)
            if len(current_keys) == 1 and current_keys[0] == 0:
                current_keys = None
            else:
                current_keys = current_keys[current_keys != 0]
            update_positions, change_dict = self.update_embedding_book(new_embedding, current_keys)
            new_embedding_keys_book = update_positions[new_embedding_keys]            
            cur_buf_coords_voxel = cur_buf_coords_voxel.cpu().pin_memory().cuda(non_blocking=True)
            unique_keys_o3d = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(cur_buf_coords_voxel.clone()))
            
            condidance_value = condidance_value.cpu().pin_memory().cuda(non_blocking=True)   
            condidance_value_o3d = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(condidance_value.clone()))
            
            found_indices, found_mask = self.conf_hash_map.find(unique_keys_o3d)
            found_indices = found_indices[found_mask]
            
            value_tensors = self.conf_hash_map.value_tensors()
            if value_tensors and found_indices.shape[0] > 0:
                existing_values = value_tensors[0][found_indices]  
            else:
                existing_values = None
            
            key_update_mask = torch.ones_like(condidance_value, dtype = torch.bool)
            if existing_values is not None:
                o3d.core.cuda.synchronize()  
                update_mask = found_mask  
                update_mask_converted = update_mask.to(o3d.core.Dtype.Int32)
                update_mask_tensor = torch.utils.dlpack.from_dlpack(update_mask_converted.to_dlpack()).to(torch.bool)
                existing_values_tensor = torch.utils.dlpack.from_dlpack(existing_values.to_dlpack()).view(-1)
                key_update_mask[update_mask_tensor] = existing_values_tensor < condidance_value[update_mask_tensor]
                condidance_value_o3d.contiguous()
                update_mask.contiguous()
                condidance_value_o3d_tensor = torch.utils.dlpack.from_dlpack(condidance_value_o3d.to_dlpack())
                origin_confidence = condidance_value_o3d_tensor[update_mask_tensor]
                condidance_value_o3d_tensor[update_mask_tensor] = torch.maximum(existing_values_tensor, condidance_value_o3d_tensor[update_mask_tensor])
                update_confidence = condidance_value_o3d_tensor[update_mask_tensor] - origin_confidence

                
                update_value = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(condidance_value_o3d_tensor.clone()))
            else:
                update_value = condidance_value_o3d

         
         
         
            new_embedding_score = new_embedding_score[v_proj, u_proj]
            new_embedding_score = new_embedding_score.to(torch.float32)
            current_score = new_embedding_score.cpu().pin_memory().cuda(non_blocking=True)
            current_score_o3d = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(current_score.clone()))
            found_score_indices, found_score_mask = self.hash_scale_score.find(unique_keys_o3d)
            found_score_indices = found_score_indices[found_score_mask]
            
            score_tensors = self.hash_scale_score.value_tensors()
            if score_tensors and found_score_indices.shape[0] > 0:
                existing_score = score_tensors[0][found_score_indices] 
            else:
                existing_score = None
            key_score_update_mask = torch.ones_like(current_score, dtype = torch.bool)
            if existing_score is not None:
                o3d.core.cuda.synchronize()  
                update_score_mask = found_score_mask 
                update_score_mask_converted = update_score_mask.to(o3d.core.Dtype.Int32)
                update_score_mask_tensor = torch.utils.dlpack.from_dlpack(update_score_mask_converted.to_dlpack()).to(torch.bool)
                existing_score_tensor = torch.utils.dlpack.from_dlpack(existing_score.to_dlpack())
                key_score_update_mask[update_score_mask_tensor] = existing_score_tensor < current_score[update_score_mask_tensor]
                current_score_o3d.contiguous()
                update_score_mask.contiguous()
                current_score_o3d_tensor = torch.utils.dlpack.from_dlpack(current_score_o3d.to_dlpack())
                origin_score = current_score_o3d_tensor[update_score_mask_tensor]
                current_score_o3d_tensor[update_score_mask_tensor] = torch.maximum(existing_score_tensor, current_score_o3d_tensor[update_score_mask_tensor])
                update_score= current_score_o3d_tensor[update_score_mask_tensor] - origin_score
                
                update_score = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(current_score_o3d_tensor.clone()))
            else:
                update_score = current_score_o3d
                
                

            cur_buf_indices_cpu = cur_buf_indices.cpu()
            key_update_mask_cpu = key_update_mask.cpu()
            key_score_update_mask_cpu = key_score_update_mask.cpu()
            v_proj_cpu = v_proj.cpu()
            u_proj_cpu = u_proj.cpu()
            scale_cpu = torch.tensor([0,1,2]).cpu()
            new_emb_keys = new_embedding_keys_book[v_proj_cpu, u_proj_cpu, :][key_score_update_mask_cpu].to(torch.int64)
            combined_mask = torch.zeros_like(self.emb_keys, dtype=torch.bool, device=self.emb_keys.device)
            combined_mask[cur_buf_indices_cpu,:] = key_score_update_mask_cpu
            self.emb_keys[combined_mask] = new_emb_keys.flatten()
            if unique_keys_o3d.shape[0] > 0:
                unique_keys_o3d = unique_keys_o3d.clone()
                update_value = update_value.clone()
                masks = self.conf_hash_map.erase(unique_keys_o3d)
                try:
                    torch.cuda.synchronize()  
                    o3d.core.cuda.synchronize()
                except RuntimeError as e:
                    print(f"synchronize: {e}")
                buf_indices, masks = self.conf_hash_map.insert(unique_keys_o3d, update_value)
                try:
                    torch.cuda.synchronize()  
                    o3d.core.cuda.synchronize()
                    if sum(masks.cpu().numpy()) != masks.shape[0]:
                        raise NotImplementedError
                except RuntimeError as e:
                    print(f"synchronize: {e}")
                    
                update_score = update_score.clone()
                masks = self.hash_scale_score.erase(unique_keys_o3d)
                try:
                    torch.cuda.synchronize()  
                    o3d.core.cuda.synchronize()
                except RuntimeError as e:
                    print(f"synchronize: {e}")
                buf_indices, masks = self.hash_scale_score.insert(unique_keys_o3d, update_score)
                try:
                    torch.cuda.synchronize()  
                    o3d.core.cuda.synchronize()
                    if sum(masks.cpu().numpy()) != masks.shape[0]:
                        raise NotImplementedError
                except RuntimeError as e:
                    print(f"synchronize: {e}")
                    
            torch.cuda.empty_cache()
            self.image_count += 1
        
    @torch.no_grad()
    def fast_object_query(
        self, t_emb, points, colors, only_poi=False, obj_thresh=0.5, **kwargs
    ):
        """ obtain heatmap of relevence between map and query
        Args:
            t_emb (torch.Tensor): text embedding
        """
        assert t_emb.shape[0] == 1, "[*] only support single query"
        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        buf_indices = self.active_buf_indices()
        buf_coords = self.buf_coords(buf_indices)
        buf_indices = torch.utils.dlpack.from_dlpack(
            buf_indices.to_dlpack()
        )
        buf_coords = torch.utils.dlpack.from_dlpack(
            buf_coords.to_dlpack()
        )
        embed = self.embed.to(t_emb.device)[buf_indices]
        mask_pred_caption = embed / (embed.norm(dim=-1, keepdim=True) + 1e-7)
        out = torch.einsum("cd,nd->cn", t_emb, mask_pred_caption).flatten().cpu() # (N,)

        if only_poi:
            return buf_coords[out > obj_thresh].cpu().numpy()

        out = out.cpu().numpy()
        out = (out - np.min(out)) / np.max(out)
        cmap = plt.get_cmap("plasma")
        colors = np.array([(np.array(cmap(v)[:3])) for v in out])
        return buf_coords.cpu().numpy(), colors

    @torch.no_grad()
    def object_query(
        self, t_emb, points, colors=None, cmap=None, obj_thresh=0.1, **kwargs
    ):
        if cmap is None:
            cmap = plt.get_cmap("plasma")
        buf_coords, colors = self.fast_object_query(
            t_emb, points, colors, only_poi=False, obj_thresh=obj_thresh
        )

        tree = KDTree(buf_coords.reshape(-1,3), leaf_size=10)
        dist, ind = tree.query(points, k=5)

        colors = np.array([np.mean(m, axis=0) for m in colors[ind]])
        return points, colors

    @torch.no_grad()
    def visualize_points_with_values(self, coords, values, colormap='viridis', sample_ratio=1.0, save_path=None):
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        
        N = coords.shape[0]
        if sample_ratio < 1.0:
            sample_size = max(1, int(N * sample_ratio))
            indices = np.random.permutation(N)[:sample_size]
            coords = coords[indices]
            values = values[indices]
        

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        

        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                            c=values, cmap=colormap, marker='o', s=20)
        

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Value')
        

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Points Visualization with Values")
        
    def save_depth_tensor_rgb_as_image(self,depth_img, tensor, rgb_img,u_values,v_values, filename="depth_tensor_rgb_output.png"):

        tensor = tensor.cpu().numpy()  
        H, W = tensor.shape
        depth_img = depth_img.cpu().numpy() 
        depth_img = np.clip(depth_img, 0, np.max(depth_img)) 
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8) 

        
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))  
        ax1, ax2, ax3,ax4 = axes

        
        ax1.imshow(depth_img, cmap='gray')  
        ax1.set_title("Depth Image")
        ax1.axis('off')

       
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)  
        im = ax2.imshow(tensor, cmap='jet')  
        ax2.set_title("Tensor Visualization")
        ax2.axis('off')

        
        fig.colorbar(im, cax=cax)
        img = np.ones((H, W, 3), dtype=np.uint8) * 255
        u_values = np.clip(u_values, 0, W-1)
        v_values = np.clip(v_values, 0, H-1)

        
        for i in range(len(u_values)):
            u = u_values[i]
            v = v_values[i]

            
            condidance_value = tensor[v, u]  

            
            norm_value = np.clip((condidance_value - tensor.min()) / (tensor.max() - tensor.min()) * 255, 0, 255).astype(np.uint8)
            norm_value = 255 - norm_value
            
            color = cv2.applyColorMap(np.array([[norm_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
            img[v, u] = color  

        ax3.imshow(img)
        ax3.set_title("project Image")
        ax3.axis('off')
        
        
        ax4.imshow(rgb_img)
        ax4.set_title("RGB Image")
        ax4.axis('off')

        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

        
    def semantic_query(self, t_emb, points, colors, cmap=None):
        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7) # (N,C)
        buf_indices = self.active_buf_indices()
        buf_indices = torch.utils.dlpack.from_dlpack(
            buf_indices.to_dlpack()
        )
        buf_indices = buf_indices.to(self.emb_keys.device)
        embed_keys = self.emb_keys[buf_indices]
        
        none_zeros_mask = torch.all(embed_keys == 0, dim=1)
        zeros_mask = ~none_zeros_mask
        valid_embed_keys = embed_keys[zeros_mask]
        valid_buf_indices = buf_indices[zeros_mask]
        valid_buf_indices = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(valid_buf_indices)).cuda()
        
        buf_coords = self.buf_coords(valid_buf_indices)
        image_features = self.embedding_book[valid_embed_keys]
        text_features = t_emb
        bests, bestm, score_map = self.compute_similarity_and_entropy(image_features,text_features)
        score_map = score_map
        buf_coords_np = buf_coords.cpu().numpy().reshape(-1,3)
        value_distribution = torch.softmax(score_map, dim=0).cpu().numpy()
        self.visualize_points_with_values(buf_coords_np, value_distribution, colormap='viridis', sample_ratio=1.0, save_path=None)
        tree = KDTree(buf_coords.cpu().numpy().reshape(-1,3), leaf_size=10)
        dist, ind = tree.query(points, k=3)
        return points, colors
    
    def get_uninspected_points(self,query_embedding, room_embedding, query_probability):
        buf_indices = self.active_buf_indices()
        buf_indices = torch.utils.dlpack.from_dlpack(
            buf_indices.to_dlpack()
        )
        buf_indices = buf_indices.to(self.emb_keys.device)
        embed_keys = self.emb_keys[buf_indices] 
        none_zeros_mask = torch.all(embed_keys == 0, dim=1)
        zeros_mask = ~none_zeros_mask
        valid_embed_keys = embed_keys[zeros_mask]
        valid_buf_indices = buf_indices[zeros_mask]
        valid_buf_indices = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(valid_buf_indices)).cuda()
        buf_coords = self.buf_coords(valid_buf_indices)
        buf_coords = torch.utils.dlpack.from_dlpack(
            buf_coords.to_dlpack()
        )
        active_voxel_coords =  torch.floor(buf_coords / self.voxel_size).to(torch.int32) 
        if active_voxel_coords.shape[0] > 0:
            active_voxel_coords = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(active_voxel_coords))
            found_indices, found_mask = self.conf_hash_map.find(active_voxel_coords)
            found_indices = found_indices[found_mask]
            value_tensors = self.conf_hash_map.value_tensors()
            if value_tensors and found_indices.shape[0] > 0:
                active_confidence_value = value_tensors[0][found_indices]  # 只取第一个 value tensor
            else:
                active_confidence_value = None
            
            if active_confidence_value.shape[0] != active_voxel_coords.shape[0]:
                print("error: can not find all the active confidence in conf_hash_map")
                print("active_confidence_value: ",active_confidence_value.shape)
                print("unique_voxel_coords_obs: ",active_voxel_coords.shape)
                return torch.tensor([0,3]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), [],torch.empty((0, 3)),torch.tensor([0])
                # return None
                
        active_voxel_coords = torch.utils.dlpack.from_dlpack(active_voxel_coords.to_dlpack())
        active_coords = active_voxel_coords * self.voxel_size
        active_confidence_value =  torch.utils.dlpack.from_dlpack(active_confidence_value.to_dlpack())
        image_features = self.embedding_book[valid_embed_keys]
        text_features = query_embedding
        bests, bestm, score_map = self.compute_sum_similarity(image_features,text_features,query_probability)
        room_category = self.compute_map_catagories(image_features,room_embedding)
        max_room_category = self.most_frequent_indices(room_category)
        score_max = score_map.max()**2
        score_min = score_map.min()**2
        if score_min!= score_max:
            score_map = (score_map**2 - score_min) / (score_max - score_min)
        active_uncertanty_value = 1 - active_confidence_value.squeeze()
        semantic_gain = active_uncertanty_value * score_map
        semantic_gain_mask = (semantic_gain > 0.01) & (active_coords[:, 2] > 0.1) & (active_coords[:, 2] < 2.5)
        semantic_gain_filter = semantic_gain[semantic_gain_mask]
        active_coords_filter = active_coords[semantic_gain_mask]
        if semantic_gain_filter.shape[0] > 30:
            top_semantic, top_indices = torch.topk(semantic_gain_filter, 30)
        else:
            top_semantic, top_indices = torch.topk(semantic_gain_filter,semantic_gain_filter.shape[0])
        top_active_coords = active_coords[top_indices,:]
        db = DBSCAN(eps=0.5, min_samples=5)
        labels = db.fit_predict(top_active_coords.cpu().numpy())  # -1 表示噪声
        unique_labels = set(labels)
        cluster_points_list = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = top_active_coords[labels == label]
            cluster_points_list.append(cluster_points)
        mask = (buf_coords[:, 2] < (2 + self.agent_height)) & (buf_coords[:, 2] > (0.2+ self.agent_height))
        mask_room = (buf_coords[:, 2] < (2 + self.agent_height)) & (buf_coords[:, 2] > (0.44 + self.agent_height))
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return buf_coords[mask], score_map[mask], semantic_gain[mask], active_uncertanty_value[mask], cluster_points_list,buf_coords[mask_room],max_room_category[mask_room]
    
    def get_room_feature(self,room_embedding):
        buf_indices = self.active_buf_indices()
        buf_indices = torch.utils.dlpack.from_dlpack(
            buf_indices.to_dlpack()
        )
        buf_indices = buf_indices.to(self.emb_keys.device)
        embed_keys = self.emb_keys[buf_indices] 
        none_zeros_mask = torch.all(embed_keys == 0, dim=1)
        zeros_mask = ~none_zeros_mask
        valid_embed_keys = embed_keys[zeros_mask]
        valid_buf_indices = buf_indices[zeros_mask]
        valid_buf_indices = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(valid_buf_indices)).cuda()
        buf_coords = self.buf_coords(valid_buf_indices)
        buf_coords = torch.utils.dlpack.from_dlpack(
            buf_coords.to_dlpack()
        )
        active_voxel_coords =  torch.floor(buf_coords / self.voxel_size).to(torch.int32) 
        if active_voxel_coords.shape[0] > 0:
            active_voxel_coords = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(active_voxel_coords))
            found_indices, found_mask = self.conf_hash_map.find(active_voxel_coords)
            found_indices = found_indices[found_mask]
            value_tensors = self.conf_hash_map.value_tensors()
            if value_tensors and found_indices.shape[0] > 0:
                active_confidence_value = value_tensors[0][found_indices]  # 只取第一个 value tensor
            else:
                active_confidence_value = None
            
            if active_confidence_value.shape[0] != active_voxel_coords.shape[0]:
                print("error: can not find all the active confidence in conf_hash_map")
                print("active_confidence_value: ",active_confidence_value.shape)
                print("unique_voxel_coords_obs: ",active_voxel_coords.shape)
                return torch.tensor([0,3]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), [],torch.empty((0, 3)),torch.tensor([0])
                
        active_voxel_coords = torch.utils.dlpack.from_dlpack(active_voxel_coords.to_dlpack())
        active_coords = active_voxel_coords * self.voxel_size
        active_confidence_value =  torch.utils.dlpack.from_dlpack(active_confidence_value.to_dlpack())
        image_features = self.embedding_book[valid_embed_keys]
        room_category = self.compute_map_catagories(image_features,room_embedding)
        max_room_category = self.most_frequent_indices(room_category)
        mask_room = buf_coords[:, 2] < 2
        return buf_coords[mask_room],max_room_category[mask_room]
    def get_observed_area(self):
        buf_indices = self.active_buf_indices()
        buf_indices = torch.utils.dlpack.from_dlpack(
            buf_indices.to_dlpack()
        )
        buf_indices = buf_indices.to(self.emb_keys.device)
        embed_keys = self.emb_keys[buf_indices]
        none_zeros_mask = torch.all(embed_keys == 0, dim=1)
        zeros_mask = ~none_zeros_mask
        valid_embed_keys = embed_keys[zeros_mask]
        valid_buf_indices = buf_indices[zeros_mask]
        valid_buf_indices = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(valid_buf_indices)).cuda()
        buf_coords = self.buf_coords(valid_buf_indices)
        buf_coords = torch.utils.dlpack.from_dlpack(
            buf_coords.to_dlpack()
        )
        active_voxel_coords =  torch.floor(buf_coords / self.voxel_size).to(torch.int32) 
        if active_voxel_coords.shape[0] > 0:
            active_voxel_coords = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(active_voxel_coords))
            found_indices, found_mask = self.conf_hash_map.find(active_voxel_coords)
            found_indices = found_indices[found_mask]
            value_tensors = self.conf_hash_map.value_tensors()
            if value_tensors and found_indices.shape[0] > 0:
                active_confidence_value = value_tensors[0][found_indices]  
            else:
                active_confidence_value = None

            if active_confidence_value.shape[0] != active_voxel_coords.shape[0]:
                print("error: can not find all the active confidence in conf_hash_map")
                print("active_confidence_value: ",active_confidence_value.shape)
                print("unique_voxel_coords_obs: ",active_voxel_coords.shape)
        active_confidence_value =  torch.utils.dlpack.from_dlpack(active_confidence_value.to_dlpack())
        mask = (buf_coords[:,2] > 0.05) & (buf_coords[:,2] < 3)
        return buf_coords[mask],  active_confidence_value[mask]
    
        
    
    def compute_info_gain(self,query_embedding, query_probability, extrinsic, observation_range = [1.0,1.5]):
        self.compute_count +=1
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        start_time = time.time()
        extrinsic = o3c.Tensor.from_numpy(extrinsic)
        width, height = self.img_size
        buf_indices = self.active_buf_indices()
        buf_indices = torch.utils.dlpack.from_dlpack(
            buf_indices.to_dlpack()
        )
        buf_indices = buf_indices.to(self.emb_keys.device)
        embed_keys = self.emb_keys[buf_indices]
        
        none_zeros_mask = torch.all(embed_keys == 0, dim=1)
        zeros_mask = ~none_zeros_mask
        valid_embed_keys = embed_keys[zeros_mask]
        valid_buf_indices_tensor = buf_indices[zeros_mask].cuda()
        valid_buf_indices = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(valid_buf_indices_tensor)).cuda()
        
        buf_coords = self.buf_coords(valid_buf_indices).cuda()
        image_features = self.embedding_book[valid_embed_keys].cuda()
        v_proj, u_proj, d_proj, mask_proj = self.get_points_in_fov(
            buf_coords, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
        )
        if v_proj.shape[0] > 0:
            buf_coords = torch.utils.dlpack.from_dlpack(
                buf_coords.to_dlpack()
            )
            buf_coords_origin = buf_coords.clone()
            mask_proj = mask_proj.to(image_features.device)
            image_features = image_features[mask_proj]
            buf_coords = buf_coords[mask_proj]
            valid_buf_indices_tensor = valid_buf_indices_tensor[mask_proj]
            height_mask = (buf_coords[:, 2] < (1.8 + self.agent_height)) & (buf_coords[:, 2] > (0.2 + self.agent_height)) 
            valid_buf_indices_tensor = valid_buf_indices_tensor[height_mask]
            buf_coords = buf_coords[height_mask]
            image_features = image_features[height_mask]
            v_proj = v_proj[height_mask]
            u_proj = u_proj[height_mask]
            d_proj = d_proj[height_mask]
            buf_coords = buf_coords.clone()
            buf_coords_o3d = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(buf_coords_origin))
            buf_coords_o3d = (buf_coords_o3d / (self.voxel_size * self.block_resolution)).to(o3d.core.Dtype.Int32).cuda()
            ray_cast_result =self.world.ray_cast(block_coords =buf_coords_o3d,
            intrinsic = self.intrinsic, extrinsic = extrinsic, 
            width =  width, height = height, render_attributes = ['depth','color'], 
            depth_scale = self.depth_scale, depth_min = self.depth_min, 
            depth_max = self.depth_max, weight_threshold = 1.0, 
            trunc_voxel_multiplier = float(self.block_resolution),
            range_map_down_factor = 8)
            try:    
                torch.cuda.synchronize()  
                o3d.core.cuda.synchronize()
            except RuntimeError as e:
                print(f"CUDA error occurred: {e}")
            render_depth_map = ray_cast_result['depth']
            render_depth_map =  torch.utils.dlpack.from_dlpack(
            render_depth_map.to_dlpack()).squeeze()
            render_depth_map_np = render_depth_map.cpu().numpy()
            depth_normalized = cv2.normalize(render_depth_map_np, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)
            
            observed_in_fov_mask = render_depth_map_np != 0
            observed_in_fov_mask_int = observed_in_fov_mask.astype(np.uint8)
            mask_normalized = cv2.normalize(observed_in_fov_mask_int, None, 0, 255, cv2.NORM_MINMAX)
            
        
            render_RGB_map = ray_cast_result['color']
            render_RGB_map =  torch.utils.dlpack.from_dlpack(
            render_RGB_map.to_dlpack()).squeeze()
            render_RGB_map = render_RGB_map.cpu().numpy()
            if render_RGB_map.max() <= 1.0:
                render_RGB_map = (render_RGB_map * 255).astype(np.uint8)
            else:
                render_RGB_map = render_RGB_map.astype(np.uint8)

            render_RGB_map = cv2.cvtColor(render_RGB_map, cv2.COLOR_RGB2BGR)
            
            
            
            render_depth = render_depth_map[v_proj, u_proj]
            origin_depth = d_proj
            diff_depth = origin_depth - render_depth
            render_mask = (diff_depth) < (self.voxel_size * self.block_resolution) * 1.3
            valid_buf_indices_tensor = valid_buf_indices_tensor[render_mask]
            buf_coords = buf_coords[render_mask]
            v_depth_filter = v_proj[render_mask]
            u_depth_filter = u_proj[render_mask]
            d_depth_filter = d_proj[render_mask]
            active_voxel_coords =  torch.floor(buf_coords / self.voxel_size).to(torch.int32) 
            valid_buf_indices = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(valid_buf_indices_tensor)).cuda()
            tsdf_value =  torch.utils.dlpack.from_dlpack(
                self.world.attribute("tsdf")[valid_buf_indices].to_dlpack()
            ).reshape(-1)
            tsdf_mask = tsdf_value >= 0
            buf_coords = buf_coords[tsdf_mask]
            active_voxel_coords = active_voxel_coords[tsdf_mask]
            v_depth_filter = v_depth_filter[tsdf_mask]
            u_depth_filter = u_depth_filter[tsdf_mask]
            d_depth_filter = d_depth_filter[tsdf_mask]
            if active_voxel_coords.shape[0] > 0:
                image_features = image_features[render_mask]
                image_features = image_features[tsdf_mask]
                text_features = query_embedding
                bests, bestm, score_map = self.compute_sum_similarity(image_features,text_features,query_probability)
                score_max = score_map.max()**2
                score_min = score_map.min()**2
                if score_min!= score_max:
                    score_map = (score_map**2 - score_min) / (score_max - score_min)
                    
                active_voxel_coords = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(active_voxel_coords))
                found_indices, found_mask = self.conf_hash_map.find(active_voxel_coords)
                masked_found_indices = found_indices[found_mask]

                value_tensors = self.conf_hash_map.value_tensors()
                if value_tensors and masked_found_indices.shape[0] > 0:
                    active_confidence_value = value_tensors[0][masked_found_indices]  
                    
                else:
                    active_confidence_value = None
                
                if active_confidence_value is None:
                    print("error: can not find all the active confidence in conf_hash_map with None")
                    return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.zeros((height,width)), torch.zeros([0,0]), torch.zeros([0,2])
                
                if active_confidence_value.shape[0] != active_voxel_coords.shape[0]:
                    print("error: can not find all the active confidence in conf_hash_map") 
                    active_voxel_coords = active_voxel_coords[found_mask,:]
                    buf_coords = buf_coords.clone()
                    found_mask_tensor = torch.from_dlpack(found_mask.to(o3d.core.Dtype.Int32).to_dlpack()).to(torch.bool)
                    buf_coords = buf_coords[found_mask_tensor,:]
                    score_map = score_map[found_mask_tensor]
                    v_depth_filter = v_depth_filter[found_mask_tensor]
                    u_depth_filter = u_depth_filter[found_mask_tensor]
                    d_depth_filter = d_depth_filter[found_mask_tensor]
                
                active_confidence_value = torch.utils.dlpack.from_dlpack(
                    active_confidence_value.to_dlpack()
                )
                active_uncertainty_value = 1 - active_confidence_value
                points_uv = torch.stack([u_depth_filter, v_depth_filter], dim=1)
                render_confidence_depth_map = torch.torch.ones_like(render_depth_map) * self.depth_max
                render_confidence_depth_map[v_depth_filter, u_depth_filter] = d_depth_filter
                render_confidence_map = self.compute_confidence_from_depth_torch(
                render_confidence_depth_map,
                fx = self.intrinsic_np[0,0], fy = self.intrinsic_np[1,1], cx = self.intrinsic_np[0,2], cy = self.intrinsic_np[1,2],
                best_distance = observation_range,
                min_distance= 0.0,
                max_distance = self.depth_max * 0.5,
                alpha = 0.25,
                gamma = 0.05,
                w_d = 0.8,
                w_yaw = 0.1,
                w_pitch = 0.10,
                with_yaw_pitch = False
                )
                render_confidence = render_confidence_map[v_depth_filter, u_depth_filter]
                def gray_to_3channel(gray):
                    return cv2.merge([gray, gray, gray])  # 复制三个通道
                points_in_fov = self.circles_image(u_proj, v_proj, image_size= (height,width),background = render_RGB_map, filename="tensor_circles_output.png")
                observed_in_fov_mask = torch.tensor(observed_in_fov_mask)
                semantic_information_gain = torch.sum(active_uncertainty_value.squeeze() * score_map * render_confidence) * self.semantic_gain_weight 
            else:
                semantic_information_gain = torch.tensor(0)
                observed_in_fov_mask = torch.zeros([height,width])
                points_in_fov = torch.zeros([0,0])
                buf_coords = torch.zeros([0,3])
        else:
            semantic_information_gain = torch.tensor(0)
            observed_in_fov_mask = torch.zeros([height,width])
            points_in_fov = torch.zeros([0,0])  
            buf_coords = torch.zeros([0,3])          
        if not observed_in_fov_mask.all():
            max_depth_image = np.ones_like(observed_in_fov_mask) * self.depth_max * 0.7
            unobserved_coords, mask = self.depth_to_point_cloud(
                    max_depth_image, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max, self.depth_scale, observed_mask = observed_in_fov_mask
                )
            unique_voxel_coords_obs = self.voxels_in_fov(unobserved_coords, extrinsic)
            
            value_tensors = self.conf_hash_map.value_tensors()
            unique_voxel_coords_obs = unique_voxel_coords_obs.to(torch.int32)
            fov_voxel_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(unique_voxel_coords_obs))
            found_indices, found_mask = self.conf_hash_map.find(fov_voxel_keys)
            found_indices = found_indices[found_mask]
            if value_tensors and found_indices.shape[0] > 0:
                existing_values = value_tensors[0][found_indices]  
                
            else:
                existing_values = None
            observed_voxel = fov_voxel_keys[found_mask] 
            observed_value = existing_values
            unfound_mask = found_mask.logical_not()
            unobserved_voxel = fov_voxel_keys[unfound_mask]
            
            unobserved_voxel = torch.utils.dlpack.from_dlpack(
                unobserved_voxel.to_dlpack()
            )
            explore_gain = unobserved_voxel.shape[0] * self.explore_gain_weight
        else:
            explore_gain = 0
        information_gain = explore_gain + semantic_information_gain
        if not isinstance(information_gain, torch.Tensor):
            information_gain = torch.tensor(information_gain, device='cuda:0')
        end_time = time.time()
        execution_time = end_time - start_time
        torch.cuda.empty_cache()
        return information_gain, explore_gain, semantic_information_gain, observed_in_fov_mask, points_in_fov, buf_coords[:,:2]
        
    def get_score_map(self,query_embedding):
        buf_indices = self.active_buf_indices()
        buf_indices = torch.utils.dlpack.from_dlpack(
            buf_indices.to_dlpack()
        )
        buf_indices = buf_indices.to(self.emb_keys.device)
        embed_keys = self.emb_keys[buf_indices]
        
        none_zeros_mask = torch.all(embed_keys == 0, dim=1)
        zeros_mask = ~none_zeros_mask
        valid_embed_keys = embed_keys[zeros_mask]
        valid_buf_indices = buf_indices[zeros_mask]
        valid_buf_indices = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(valid_buf_indices))
        
        buf_coords = self.buf_coords(valid_buf_indices)
        image_features = self.embedding_book[valid_embed_keys]
        text_features = query_embedding
        bests, bestm, score_map = self.compute_sum_similarity(image_features,text_features)
        buf_coords = torch.utils.dlpack.from_dlpack(
            buf_coords.to_dlpack()
        )
        mask = (buf_coords[:, 2] < (3+ self.agent_height)) & (buf_coords[:, 2] > (0.05+ self.agent_height))

        buf_coords = buf_coords[mask]
        score_map = score_map[mask]
        return buf_coords, score_map, bests, bestm
        
        



    def filter_by_min_depth_lexsort(self, v_proj, u_proj, d_proj, image_width):
        pixel_idx = v_proj * image_width + u_proj  
        
        
        pixel_idx_np = pixel_idx.cpu().numpy()  # (M,)
        d_proj_np = d_proj.cpu().numpy()          # (M,)
        
        sorted_indices = np.lexsort((d_proj_np, pixel_idx_np))  
        
        
        sorted_pixel_idx = pixel_idx_np[sorted_indices]
        _, unique_first_idx = np.unique(sorted_pixel_idx, return_index=True)
        

        selected_sorted_indices = sorted_indices[unique_first_idx]
        

        new_mask_np = np.zeros_like(pixel_idx_np, dtype=bool)
        new_mask_np[selected_sorted_indices] = True
        new_mask = torch.from_numpy(new_mask_np)
        filter_v_proj = v_proj[new_mask]
        filter_u_proj = u_proj[new_mask]
        filter_d_proj = d_proj[new_mask]
        return filter_v_proj,filter_u_proj,filter_d_proj,new_mask
    
    
    
    def create_concave_mask(self, points_uv, image_width, image_height, alpha=0.5):
        if points_uv.shape[0] < 20:
            return np.zeros((image_height, image_width), dtype=bool), None
        try:
            concave_hull = alphashape.alphashape(points_uv, alpha)
        except Exception as e:
            print(f"Error while calculating concave hull: {e}")
            return np.zeros((image_height, image_width), dtype=bool), None
        
        
        if concave_hull is None:
            return np.zeros((image_height, image_width), dtype=bool), None
        
        if isinstance(concave_hull, GeometryCollection):
            concave_hull = max(concave_hull.geoms, key=lambda p: p.area)
        if not isinstance(concave_hull, Polygon):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(points_uv[:, 0], points_uv[:, 1], s=10)
            ax.set_title("2D Scatter Plot of Points")
            plt.savefig("concave_hulls_plot.png", dpi=300, bbox_inches='tight') 
            print("Warning: The resulting concave hull is not a Polygon.")
            print("TYPE concave_hull: ",type(concave_hull))
            return np.zeros((image_height, image_width), dtype=bool), None
        
        hull_coords = np.array(concave_hull.exterior.coords)  
        hull_coords = np.round(hull_coords).astype(np.int32)
        hull_coords[:, 0] = np.clip(hull_coords[:, 0], 0, image_width - 1)
        hull_coords[:, 1] = np.clip(hull_coords[:, 1], 0, image_height - 1)
        pts = hull_coords.reshape((-1, 1, 2))
        
        mask_uint8 = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask_uint8, [pts], color=1)
        mask = mask_uint8.astype(bool)
        
        return mask, concave_hull

    def cluster_and_generate_masks(self, points_uv, image_width, image_height, eps=30, min_samples=20, alpha=0.2):
        points_uv = points_uv.cpu().numpy()
        final_mask = np.zeros((image_height, image_width), dtype=bool)
        concave_mask, concave_hull = self.create_concave_mask(points_uv, image_width, image_height, alpha=alpha)
        final_mask = final_mask | concave_mask
        return final_mask
    
    

    def circles_image(self, u_values, v_values, image_size=(1000, 1000),background = None, filename="tensor_circles_output.png"):
        H, W = image_size  

       
        if background is None:
            img = np.ones((H, W, 3), dtype=np.uint8) * 255
        else:
            img = background.copy()

        
        u_values = np.clip(u_values.cpu().numpy(), 0, W - 1)
        v_values = np.clip(v_values.cpu().numpy(), 0, H - 1)

        
        for i in range(len(u_values)):
            u = u_values[i]
            v = v_values[i]

            
            radius = 4  
            cv2.circle(img, (u, v), radius, (0, 0, 0), -1)  

        
        return img
    
    def pooling_depth(self,u, v, d, img_height, img_width, kernel_size=10):
        print("u: ",u.shape)
        depth_tensor = torch.full((img_height, img_width), float('inf')).to(u.device)
        depth_tensor[v, u] = d
        depth_tensor_neg = -depth_tensor
        pooled_depth = -F.max_pool2d(depth_tensor_neg.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=kernel_size//2)
        pooled_depth = pooled_depth.squeeze(0).squeeze(0)
        pooled_depth = pooled_depth[v,u]
        print("pooled_depth: ",pooled_depth)
        pooling_mask = pooled_depth == d
        pooling_mask = pooling_mask.to(u.device)
        print("pooling_mask: ",pooling_mask.shape)
        pooled_v = v[pooling_mask]
        pooled_u = u[pooling_mask]
        pooled_d = d[pooling_mask]
        return pooled_v,pooled_u,pooled_d,pooling_mask
        
         
         