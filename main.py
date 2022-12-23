import math
import os
import shutil
import time
import cv2
import numpy as np
import BrnoCompSpeedGtReader
import VehicleDetector
import OutputImagePainter
import MultipleVehicleTracker
from CameraCalibration import CameraCalibration
from emissions import movestar
import argparse


def run_session(dataset_session_path: str, interactive=False):
    video_name = dataset_session_path.split('/')[-1]
    print(f'Processing {video_name}')
    gt_data_path = dataset_session_path + '/gt_data.pkl'
    video_path = dataset_session_path + '/video.avi'
    mask_path = dataset_session_path + '/video_mask.png'
    frame_path = dataset_session_path + '/screen.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    frame_example = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    output_path = './outputs/images/' + video_name
    detector = VehicleDetector.VehicleDetector()
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    dataset_gt = BrnoCompSpeedGtReader.BrnoCompSpeedGtReader(gt_data_path, video_name)
    n_lanes = dataset_gt.get_lane_count()
    screen_coordinates, world_coordinates = dataset_gt.world_and_screen_coordinates()
    camera_calibration = CameraCalibration(screen_coordinates, world_coordinates, dataset_gt.goal_line(),
                                           frame_example.shape[1], frame_example.shape[0])
    fps = dataset_gt.fps()
    tracker = MultipleVehicleTracker.MultipleVehicleTracker(fps)
    detection_sequence = []
    min_time = dataset_gt.get_min_time()
    max_time = dataset_gt.get_max_time()
    min_frame = int(min_time * fps)
    max_frame = int(max_time * fps)

    # Advance the video to the first important frame
    frame_num = min_frame
    vid.set(cv2.CAP_PROP_POS_FRAMES, min_frame * 2)

    csv_array = []
    detector.initialize()
    last_time = time.time()
    while frame_num < max_frame:
        if frame_num % 1000 == 0:
            current_time = time.time()
            print('******* Frame #', frame_num, ' fps = ', round(1 / ((current_time - last_time) / 1000), 3))
            last_time = time.time()
        return_value, frame = vid.read()
        if not return_value:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_detections = detector.detect(frame, frame_num)

        # Delete detections that are outside the mask
        filtered_detections = []
        for detection in current_detections:
            h, w = mask.shape
            screen_pos = np.uint16(camera_calibration.get_road_position(detection))
            screen_pos[0] = min(w - 1, screen_pos[0])
            screen_pos[1] = min(h - 1, screen_pos[1])
            if mask[screen_pos[1], screen_pos[0]] > 0:
                filtered_detections.append(detection)
        detection_sequence.append(filtered_detections)
        trackings_idxs, trackings_data = tracker.analyze(detection_sequence[int(-fps * 3):][::-1].copy(), frame_num,
                                                         camera_calibration)

        # Delete trackings from detections
        for track in trackings_idxs:
            for element in track:
                del detection_sequence[element[0] - min_frame][element[1]]

        image_detector_tracker = OutputImagePainter.paint_image_detector_tracker(frame.copy(),
                                                                                 detection_sequence[int(-fps * 3):],
                                                                                 trackings_data,
                                                                                 camera_calibration)
        # Show last two seconds of detections
        image_classifier_top_view = OutputImagePainter.paint_detector_tracker_top_view(frame.copy(),
                                                                                       detection_sequence[
                                                                                       int(-fps * 3):],
                                                                                       trackings_data,
                                                                                       camera_calibration)

        # Find the lane to which the tracking belongs to
        trackings_lanes = []
        (g0j, g0i), (g1j, g1i) = camera_calibration.get_goal_line()
        g0x, g0y = camera_calibration.get_xyz(g0j, g0i)
        g1x, g1y = camera_calibration.get_xyz(g1j, g1i)
        invert = False
        if g1x < g0x:
            invert = True
            aux = g1x
            g1x = g0x
            g0x = aux
        for track in trackings_data:
            # Use the last element
            j, i = camera_calibration.get_centroid_position(track[0])
            x, y = camera_calibration.get_xyz(j, i)
            percentage = (x - g0x) / g1x
            lane_for_track = int(percentage * n_lanes)
            if invert:
                lane_for_track = n_lanes - 1 - lane_for_track
            trackings_lanes.append(lane_for_track)

        # Compute speed
        trackings_speeds = []
        for track, indices in zip(trackings_data, trackings_idxs):
            j0, i0 = camera_calibration.get_road_position(track[0])
            x0, y0 = camera_calibration.get_xyz(j0, i0)
            j1, i1 = camera_calibration.get_road_position(track[-1])
            x1, y1 = camera_calibration.get_xyz(j1, i1)
            distance = math.dist((x0, y0), (x1, y1))
            n_frames = indices[0][0] - indices[-1][0]
            if n_frames > 5:
                seconds = n_frames / fps
                speed_ms = distance / seconds
                speed_kmph = speed_ms * 3.6
                trackings_speeds.append(speed_kmph)
            else:
                trackings_speeds.append(-1)

        # For each tracking, find the closest frame to the goal line
        tracking_goal_frame = []
        for track_idx, track in enumerate(trackings_data):
            min_distance = math.inf
            min_idx = -1
            signed_distance = math.inf
            for idx, track_pos in enumerate(track):
                j, i = camera_calibration.get_road_position(track_pos)
                x, y = camera_calibration.get_xyz(j, i)
                distance = abs(y - g0y)
                if distance < min_distance:
                    signed_distance = y - g0y
                    min_distance = distance
                    min_idx = idx

            # Given the speed, and the distance between the closest point and the goal line, extrapolate the frame in 
            # which the vehicle should've crossed the goal line
            closest_frame = trackings_idxs[track_idx][min_idx][0]
            interpolated_frame = closest_frame - signed_distance * trackings_speeds[track_idx] / 3.6
            tracking_goal_frame.append(interpolated_frame)

        # For each tracking get the class
        trackings_classes = []
        for tracking_data in trackings_data:
            class_values = np.array(tracking_data)[:, 5]
            # Most repeated one
            class_idx = np.bincount(np.uint8(class_values)).argmax()
            trackings_classes.append(VehicleDetector.class_names[class_idx])

        # Save data to csv array: goal_frame, lane, number_of_detections, speed, class, CO, HC, NOx, PM2.5 Elem, PM2.5 Org, Energy, CO2, Fuel
        for idx in range(len(trackings_speeds)):
            if len(trackings_data[idx]) > 12:
                movestar_class = 1
                if trackings_classes[idx] == 'bus' or trackings_classes[idx] == 'truck':
                    movestar_class = 2
                emissions = movestar.movestar(movestar_class, [trackings_speeds[idx]])['Emission Rate'][0]
                data = [tracking_goal_frame[idx], trackings_lanes[idx], len(trackings_data[idx]),
                        trackings_speeds[idx], trackings_classes[idx],
                        *emissions[:8]]
                print('Time = ', tracking_goal_frame[idx] / 50)
                print('New data = ', data)
                csv_array.append(data)

        if interactive:
            cv2.imshow("imshow_python_classifier", image_detector_tracker)
            cv2.imshow("imshow_python_classifier_top", image_classifier_top_view)
            cv2.waitKey(0)

        # Update output every 10 frames
        if frame_num % 10 == 0:
            os.makedirs('outputs/' + video_name + '/', exist_ok=True)
            save_results_csv(csv_array, 'outputs/' + video_name + '/partial_measured_data.csv')

        cv2.imwrite(f'{output_path}/detector_' + str(frame_num) + '.jpg', image_detector_tracker)
        cv2.imwrite(f'{output_path}/classifier_' + str(frame_num) + '.jpg', image_classifier_top_view)
        frame_num += 1
    if interactive:
        cv2.destroyAllWindows()
    os.makedirs('outputs/' + video_name + '/', exist_ok=True)
    save_results_csv(csv_array, 'outputs/' + video_name + '/measured_data.csv')


def save_results_csv(data, path):
    with open(path, 'w') as f:
        f.write('goal_frame,lane,number_of_detections,speed,class,CO,HC,NOx,PM2.5 Elem,PM2.5 Org,Energy,CO2,Fuel\n')
        for goal_time, lane, number_of_detections, speed, class_name, CO, HC, NOx, PM25Elem, PM25Org, Energy, CO2, Fuel in data:
            f.write(str(goal_time) + ',')
            f.write(str(lane) + ',')
            f.write(str(number_of_detections) + ',')
            f.write(str(speed) + ',')
            f.write(str(class_name) + ',')
            f.write(str(CO) + ',')
            f.write(str(HC) + ',')
            f.write(str(NOx) + ',')
            f.write(str(PM25Elem) + ',')
            f.write(str(PM25Org) + ',')
            f.write(str(Energy) + ',')
            f.write(str(CO2) + ',')
            f.write(str(Fuel) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python main.py',
        description='Monitoring vehicle pollution and fuel consumption based on AI camera system and gas emission estimator model')

    parser.add_argument('dataset_dir')
    parser.add_argument('-i', '--interactive', action='store_true')
    args = parser.parse_args()

    run_session(args.dataset_dir, args.interactive)
