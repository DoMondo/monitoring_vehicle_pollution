import cv2
import numpy as np
from matplotlib import pyplot as plt

import VehicleDetector
from CameraCalibration import CameraCalibration

SCALE_FACTOR = 1
IMAGE_W = 1920 // SCALE_FACTOR
IMAGE_H = 1080 // SCALE_FACTOR


def paint_image_detector_tracker(frame, detections, trackings, camera_calibration: CameraCalibration):
    ratio_w = IMAGE_W / frame.shape[1]
    ratio_h = IMAGE_H / frame.shape[0]
    frame = cv2.resize(frame.copy(), (IMAGE_W, IMAGE_H), cv2.INTER_AREA)
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    for idx, detection in enumerate(detections[-1]):
        color = colors[int(idx) % len(colors)]
        color = [i * 255 for i in color]
        bbox = detection[:4]
        name = VehicleDetector.class_names[int(detection[5])]
        p0y = int(bbox[1] * IMAGE_H)
        p0x = int(bbox[0] * IMAGE_W)
        p1y = int(bbox[3] * IMAGE_H) + p0y
        p1x = int(bbox[2] * IMAGE_W) + p0x
        cv2.rectangle(frame, (p0x, p0y), (p1x, p1y), color, 6 // SCALE_FACTOR)
        # cv2.putText(frame, str(name) + "-" + str(idx), (int(p0x), int(p0y - 10)), 0, 0.5, (255, 255, 255), 2)
    # Draw detection center and code time with color
    cmap = plt.get_cmap('jet')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, len(detections))]
    for idx, frame_detections in enumerate(detections):
        color = np.array(colors[idx])
        color *= 255
        for detection in frame_detections:
            screen_pos = camera_calibration.get_centroid_position(detection)
            screen_pos = np.uint16(screen_pos * (ratio_w, ratio_h))
            cv2.circle(frame, screen_pos, 9 // SCALE_FACTOR, color, cv2.FILLED, cv2.LINE_AA)
    # Draw trackings
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    for idx, track in enumerate(trackings):
        color = colors[int(idx) % len(colors)]
        color = [i * 255 for i in color]
        for pos_idx in range(len(track) - 1):
            p0 = track[pos_idx][:4]
            p1 = track[pos_idx + 1][:4]
            p0 = camera_calibration.get_centroid_position(p0)
            p1 = camera_calibration.get_centroid_position(p1)
            p0 = np.uint16(p0 * (ratio_w, ratio_h))
            p1 = np.uint16(p1 * (ratio_w, ratio_h))
            # if pos_idx == 0 or pos_idx == len(track) - 1:
            #     cv2.line(frame, p0, p1, [255, 255, 255], 9 // SCALE_FACTOR, lineType=cv2.LINE_AA)
            # else:
            cv2.line(frame, p0, p1, color, 9 // SCALE_FACTOR, lineType=cv2.LINE_AA)

    # Draw first and last points of each tracking
    for idx, track in enumerate(trackings):
        p0 = track[0][:4]
        p1 = track[-1][:4]
        p0 = camera_calibration.get_centroid_position(p0)
        p1 = camera_calibration.get_centroid_position(p1)
        p0 = np.uint16(p0 * (ratio_w, ratio_h))
        p1 = np.uint16(p1 * (ratio_w, ratio_h))
        cv2.circle(frame, p0, 9 // SCALE_FACTOR, [255, 255, 255], 3, lineType=cv2.LINE_AA)
        cv2.circle(frame, p1, 9 // SCALE_FACTOR, [255, 255, 255], 3, lineType=cv2.LINE_AA)

    # Draw goal line
    p0, p1 = camera_calibration.get_goal_line()
    p0[0] *= ratio_w
    p0[1] *= ratio_h
    p1[0] *= ratio_w
    p1[1] *= ratio_h
    p0 = np.uint16(p0)
    p1 = np.uint16(p1)
    cv2.line(frame, p0, p1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = np.uint8(frame)
    return frame


def paint_detector_tracker_top_view(input_image, detections, trackings, camera_calibration: CameraCalibration):
    # Generate top view
    frame = camera_calibration.generate_top_view(input_image, IMAGE_H, IMAGE_W)
    ratio_w = frame.shape[0] / input_image.shape[1]
    ratio_h = frame.shape[1] / input_image.shape[0]
    cmap = plt.get_cmap('jet')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, len(detections))]
    # Draw Goal Line
    p0, p1 = camera_calibration.get_goal_line()
    p0x, p0y = camera_calibration.get_xyz_top_view(p0[0], p0[1])
    p1x, p1y = camera_calibration.get_xyz_top_view(p1[0], p1[1])
    p0y = np.uint16(p0y)
    p1y = np.uint16(p1y)
    p0x = np.uint16(p0x)
    p1x = np.uint16(p1x)
    cv2.line(frame, (p0x, p0y), (p1x, p1y), (255, 0, 0), 3 // SCALE_FACTOR, lineType=cv2.LINE_AA)
    # Draw detection center and code time with color
    for idx, frame_detections in enumerate(detections):
        color = np.array(colors[idx])
        color *= 255
        for detection in frame_detections:
            screen_pos = camera_calibration.get_centroid_position(detection)
            screen_pos = np.uint16(screen_pos)
            xyz = camera_calibration.get_xyz_top_view(screen_pos[0], screen_pos[1])
            x = np.int16(xyz[0])
            y = np.int16(xyz[1])
            cv2.circle(frame, (x, y), 9 // SCALE_FACTOR, color, cv2.FILLED, cv2.LINE_AA)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for idx, track in enumerate(trackings):
        color = colors[int(idx) % len(colors)]
        color = [i * 255 for i in color]
        for pos_idx in range(len(track) - 1):
            p0 = track[pos_idx][:4]
            p1 = track[pos_idx + 1][:4]
            p0 = camera_calibration.get_centroid_position(p0)
            p1 = camera_calibration.get_centroid_position(p1)
            p0 = np.uint16(p0 * (ratio_w, ratio_h))
            p1 = np.uint16(p1 * (ratio_w, ratio_h))

            xyz = camera_calibration.get_xyz_top_view(p0[0], p0[1])
            p0[0] = np.int16(xyz[0])
            p0[1] = np.int16(xyz[1])
            xyz = camera_calibration.get_xyz_top_view(p1[0], p1[1])
            p1[0] = np.int16(xyz[0])
            p1[1] = np.int16(xyz[1])
            # if pos_idx == 0 or pos_idx == len(track) - 1:
            #     cv2.line(frame, p0, p1, [255, 255, 255], 9 // SCALE_FACTOR, lineType=cv2.LINE_AA)
            # else:
            cv2.line(frame, p0, p1, color, 9 // SCALE_FACTOR, lineType=cv2.LINE_AA)

    # Draw first and last points of each tracking
    for idx, track in enumerate(trackings):
        p0 = track[0][:4]
        p1 = track[-1][:4]
        p0 = camera_calibration.get_centroid_position(p0)
        p1 = camera_calibration.get_centroid_position(p1)
        p0 = np.uint16(p0 * (ratio_w, ratio_h))
        p1 = np.uint16(p1 * (ratio_w, ratio_h))
        cv2.circle(frame, p0, 9 // SCALE_FACTOR, [255, 255, 255], 3, lineType=cv2.LINE_AA)
        cv2.circle(frame, p1, 9 // SCALE_FACTOR, [255, 255, 255], 3, lineType=cv2.LINE_AA)
    return frame
