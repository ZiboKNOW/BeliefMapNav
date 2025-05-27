# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import sys
from typing import List, Optional
import random
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from beliefmap.vlm.coco_classes import COCO_CLASSES
from beliefmap.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

sys.path.insert(0, "yolov7/")
from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import (  # noqa: E402
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from utils.torch_utils import TracedModel 
# try:
#     from models.experimental import attempt_load  # noqa: E402
#     from utils.datasets import letterbox  # noqa: E402
#     from utils.general import (  # noqa: E402
#         check_img_size,
#         non_max_suppression,
#         scale_coords,
#     )
#     from utils.torch_utils import TracedModel  # noqa: E402
# except Exception:
#     print("Could not import yolov7. This is OK if you are only using the client.")
sys.path.pop(0)


class YOLOv7:
    def __init__(self, weights: str, image_size: int = 640, half_precision: bool = True):
        """Loads the model and saves it to a field."""
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.half_precision = self.device.type != "cpu" and half_precision
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(image_size, s=stride)  # check img_size
        self.model = TracedModel(self.model, self.device, self.image_size)
        if self.half_precision:
            self.model.half()  # to FP16

        # Warm-up
        if self.device.type != "cpu":
            dummy_img = torch.rand(1, 3, int(self.image_size * 0.7), self.image_size).to(self.device)
            if self.half_precision:
                dummy_img = dummy_img.half()
            for i in range(3):
                self.model(dummy_img)

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.2,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic_nms: bool = False,
    ) -> ObjectDetections:
        """
        Outputs bounding box and class prediction data for the given image.

        Args:
            image (np.ndarray): An RGB image represented as a numpy array.
            conf_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IOU threshold for filtering detections.
            classes (list): List of classes to filter by.
            agnostic_nms (bool): Whether to use agnostic NMS.
        """
        orig_shape = image.shape

        # Preprocess image
        img = cv2.resize(
            image,
            (self.image_size, int(self.image_size * 0.7)),
            interpolation=cv2.INTER_AREA,
        )
        img = letterbox(img, new_shape=self.image_size)[0]
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half_precision else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.inference_mode():  # Calculating gradients causes a GPU memory leak
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
        )[0]
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orig_shape).round()
        pred[:, 0] /= orig_shape[1]
        pred[:, 1] /= orig_shape[0]
        pred[:, 2] /= orig_shape[1]
        pred[:, 3] /= orig_shape[0]
        boxes = pred[:, :4]
        logits = pred[:, 4]
        phrases = [COCO_CLASSES[int(i)] for i in pred[:, 5]]
        detections = ObjectDetections(boxes, logits, phrases, image_source=image, fmt="xyxy")

        return detections

class YOLOv12:
    def __init__(self, weights: str, image_size: int = 640, half_precision: bool = False):
        """Loads the model and saves it to a field."""
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.half_precision = self.device.type != "cpu" and half_precision
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.model = YOLO("/media/magic-4090/44ee543b-82db-4f62-aa8d-c1ad5dd806dc2/zzb/BliefmapNav/data/yolo12x.pt")
    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.2,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic_nms: bool = False,
    ) -> ObjectDetections:
        """
        Outputs bounding box and class prediction data for the given image.

        Args:
            image (np.ndarray): An RGB image represented as a numpy array.
            conf_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IOU threshold for filtering detections.
            classes (list): List of classes to filter by.
            agnostic_nms (bool): Whether to use agnostic NMS.
        """

        print("image: ",image.shape)
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.model.predict(img, conf=conf_thres)
        yolo_classes = list(self.model.names.values())
    # print("yolo_classes: ",yolo_classes)
        classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
        colors = [random.choices(range(256), k=3) for _ in classes_ids]
        boxes_list = []
        phrases_list = []
        logits_list = []
        for result in results:
            for box in result.boxes:
                boxes_list.append(box.xyxy[0])
                color_number = classes_ids.index(int(box.cls[0]))
                phrases_list.append(result.names[int(box.cls[0])])
                logits_list.append(box.conf[0])
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), colors[color_number], 5)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, colors[color_number], 2)
        if len(boxes_list) > 0:
            boxes = torch.stack(boxes_list)
            logits = torch.tensor(logits_list)
        else:
            boxes = torch.empty((0, 4))
            logits = torch.empty((0, 1))
        detections = ObjectDetections(boxes, logits, phrases_list, image_source=image, fmt="xyxy")
        cv2.imwrite("/media/magic-4090/44ee543b-82db-4f62-aa8d-c1ad5dd806dc2/zzb/base_line/yolo_test/test_bed_result.png", img)
        return detections


class YOLOv7Client:
    def __init__(self, port: int = 12184):
        self.url = f"http://localhost:{port}/yolov7"

    def predict(self, image_numpy: np.ndarray) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)
    args = parser.parse_args()

    print("Loading model...")

    class YOLOv7Server(ServerMixin, YOLOv7):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    yolov7 = YOLOv7Server("data/yolov7-e6e.pt")
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolov7, name="yolov7", port=args.port)

    # Instantiate model
    # model = YOLOv7(weights="data/yolov7-e6e.pt")
    # img_path = "data/horses.jpg"
    # img = cv2.imread(img_path)
    # # Convert to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # Predict
    # pred = model.predict(img)
    # print("Pred")
    # print(pred)
