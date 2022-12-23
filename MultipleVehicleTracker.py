import math
import time
from CameraCalibration import CameraCalibration


class MultipleVehicleTracker:

    def __init__(self, fps):
        print('Initializing MVT')
        # Numer of frames that guarantee that a vehicle is not visible anymore
        self.__frames_to_mutis = int(fps)
        print('frames to mutis ', self.__frames_to_mutis)
        self.__max_meters_travelled_in_a_frame = 55 / fps
        self.__max_meters_travelled_in_a_frame_horizontal = 1

    def analyze(self, detections_sequence, frame_num, camera_calibration: CameraCalibration):
        # Convert all coordinates to top-down view
        converted_detections_sequence = []
        for frame in detections_sequence:
            converted_frame = []
            for detection in frame:
                screen_pos = camera_calibration.get_road_position(detection)
                converted_frame.append(camera_calibration.get_xyz(*screen_pos))
            converted_detections_sequence.append(converted_frame)
        # For each detection, check if it has reached the goal line
        goal_line_y = camera_calibration.get_xyz(*camera_calibration.get_goal_line()[0])[1]
        trails_idx = []
        trails_positions = []
        while True:
            clearing = False
            frame_idx = 0
            while not clearing and frame_idx < len(converted_detections_sequence):
                frame_id_in_video = frame_num - frame_idx
                frame = converted_detections_sequence[frame_idx]
                detection_idx = 0
                while not clearing and detection_idx < len(frame):
                    detection = frame[detection_idx]
                    if detection[1] >= goal_line_y - 8.0 and frame_num - frame_id_in_video >= self.__frames_to_mutis:
                        # Go up
                        trail_idx_up = self.__find_trail(frame_idx, frame_num, detection,
                                                         converted_detections_sequence, True)
                        # Go down
                        trail_idx_down = self.__find_trail(frame_idx, frame_num, detection,
                                                           converted_detections_sequence, False)
                        trail_idx = trail_idx_down[::-1] + trail_idx_up
                        trail_idx = trail_idx[::-1]
                        if len(trail_idx) != 0:
                            trails_idx.append(trail_idx)
                        # Delete and break the loop
                        trail = []
                        for t in trail_idx:
                            trail.append(detections_sequence[frame_num - t[0]][t[1]])
                            del converted_detections_sequence[frame_num - t[0]][t[1]]
                        if len(trail_idx) != 0:
                            trails_positions.append(trail)
                        clearing = True
                    detection_idx += 1
                frame_idx += 1
            break
        return trails_idx, trails_positions

    def __find_trail(self, frame_idx, frame_num, position, converted_detections_per_frame, direction):
        max_skips = 5
        max_distance = self.__max_meters_travelled_in_a_frame
        current_position = position
        current_frame = frame_idx
        trail = []
        n_skips = max_skips
        while n_skips > 0:
            # Find a candidate in the next frame
            if direction:
                current_frame -= 1
            else:
                current_frame += 1
            if current_frame < 0 or current_frame >= len(converted_detections_per_frame):
                break
            min_distance_idx = -1
            min_distance = math.inf
            for detection_idx, detection in enumerate(converted_detections_per_frame[current_frame]):
                distance = abs(detection[0] - current_position[0]) + abs(detection[1] - current_position[1])
                distance_x = abs(detection[0] - current_position[0])
                # Look for the smallest distance that is going up
                if distance < min_distance and (direction and detection[1] > current_position[1]
                                                or not direction and detection[1] < current_position[1]) and \
                        distance_x < self.__max_meters_travelled_in_a_frame_horizontal:
                    min_distance_idx = detection_idx
                    min_distance = distance
            if min_distance_idx != -1 and min_distance < max_distance:
                frame_id_in_video = frame_num - current_frame
                n_skips = max_skips
                max_distance = self.__max_meters_travelled_in_a_frame
                current_position = converted_detections_per_frame[current_frame][min_distance_idx]
                trail.append([frame_id_in_video, min_distance_idx])
            else:
                n_skips -= 1
                max_distance *= 2
        return trail
