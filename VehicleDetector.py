import os
import time
import numpy as np
import torch


def format_boxes(bboxes, img_h, img_w):
    """ Convert bounding boxes from (center_y, center_x, h, w) to normalized `(xmin, ymin, width, height) """
    for box in bboxes:
        y = box[1] / img_h
        x = box[0] / img_w
        height = box[3] / img_h
        width = box[2] / img_w
        box[0], box[1], box[2], box[3] = x - width / 2, y - height / 2, width, height
    return bboxes


class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
names_class = {'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7}


class Detection:
    def __init__(self, box, class_name):
        self.box = box
        self.class_name = class_name


class VehicleDetector:
    def __init__(self):
        global class_names
        self.inference_size = 320
        self.allowed_classes = ['car', 'motorbike', 'bus', 'truck']
        self.allowed_classes_indexes = [2, 3, 5, 7]
        self.yolo_model = None

    def prepare_input(self, image):
        image = np.uint8(image)
        return image

    def initialize(self):
        if not self.yolo_model:
            global class_names
            print('Initializing yolov5')
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
            self.yolo_model.conf = 0.10  # NMS confidence threshold
            self.yolo_model.iou = 0.35  # NMS IoU threshold
            self.yolo_model.agnostic = True  # NMS class-agnostic
            self.yolo_model.multi_label = False  # NMS multiple labels per box
            self.yolo_model.classes = self.allowed_classes_indexes  # (optional list) filter by class
            self.yolo_model.max_det = 30  # maximum number of detections per image
            self.yolo_model.amp = True  # Automatic Mixed Precision (AMP) inference
            print(class_names)
            print('Initialized!')

    def detect(self, img, frame_num):
        self.initialize()
        prepared_input = self.prepare_input(img)
        start = time.time()
        detections_object = self.yolo_model(prepared_input, size=self.inference_size)
        end = time.time()
        detections = detections_object.xywh[0].cpu().numpy()
        bboxes = detections[:, :4]
        # Remove detections that are touching the bottom border
        remove_indexes = []
        for idx, box in enumerate(bboxes):
            if box[1] + box[3] >= prepared_input.shape[0] - 1:
                remove_indexes.append(idx)
        bboxes = np.array(bboxes)
        detections = np.delete(detections, remove_indexes, axis=0)
        bboxes = np.delete(bboxes, remove_indexes, axis=0)
        bboxes = format_boxes(bboxes, *prepared_input.shape[:2])
        detections[:, :4] = bboxes
        return detections
