import sys
import os

# 指定要添加的路径
specified_path = "/media/magic-4090/DATA2/zzb/vlfm"
sys.path.append(specified_path)
print("sys.path: ",sys.path)
from vlfm.vlm.grounding_dino import GroundingDINO
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.
detector = GroundingDINO(config_path = GROUNDING_DINO_CONFIG,
        weights_path = GROUNDING_DINO_WEIGHTS,
        caption = CLASSES)
