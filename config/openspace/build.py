import os.path as osp
# from beliefmap.mapping.oopenfusion.datasets import ICL, Replica, ScanNet, Kobuki, Live

BASE_PATH = osp.dirname(osp.dirname(osp.abspath(__file__)))

PARAMS = {
    # "kobuki": {
    #     "dataset": Kobuki,
    #     "path": "{}/sample/kobuki/{}",
    #     "depth_scale": 1000.0,
    #     "depth_max": 5.0,
    #     "voxel_size": 8.0 / 512,
    #     "block_resolution": 8,
    #     "block_count": 100000, # will be increased automatically if needed
    #     "img_size": (1280,720),
    #     "input_size": (640,360)
    # },
    # "icl": {
    #     "dataset": ICL,
    #     "path": "{}/sample/icl/living_room/{}",
    #     "depth_scale": 5000.0,
    #     "depth_max": 5.0,
    #     "voxel_size": 0.2,
    #     "block_resolution": 8,
    #     "block_count": 20000,
    #     "img_size": (640,480),
    #     "input_size": (640,480)
    # },
    # "replica": {
    #     "dataset": Replica,
    #     "path": "{}/sample/replica/{}",
    #     "depth_scale": 6553.5,
    #     "depth_max": 5.0,
    #     "voxel_size": 0.2,
    #     "block_resolution": 1,
    #     "block_count": 100000,
    #     "img_size": (1200,680),
    #     "input_size": (600,340)
    # },
    # "scannet": {
    #     "dataset": ScanNet,
    #     "path": "{}/sample/scannet/{}",
    #     "depth_scale": 1000.0,
    #     "depth_max": 5.0,
    #     "voxel_size": 10.0 / 512,
    #     "block_resolution": 8,
    #     "block_count": 20000,
    #     "img_size": (640,480), # Resize RGB to match depth
    #     "input_size": (320,240)
    # },
    # "live": {
    #     "dataset": Live,
    #     "path": "{}/sample/live/{}",
    #     "depth_scale": 1000.0,
    #     "depth_max": 5.0,
    #     "voxel_size": 5.0 / 512,
    #     "block_resolution": 8,
    #     "block_count": 100000, # will be increased automatically if needed
    #     "img_size": (640,360),
    #     "input_size": (640,360)
    # },
    "hm3d": {
    "depth_scale": 1.0,
    "depth_max": 10.0,
    "depth_min":0.02,
    "voxel_size": 0.25,
    "block_resolution": 1,
    "block_count": 45000, # will be increased automatically if needed
    "img_size": (640,360),
    "input_size": (640,360),
    "device":"cuda:0",
    "algo":"cfusion"
    },
}


def get_config(dataset):
    params = PARAMS[dataset].copy()
    return params