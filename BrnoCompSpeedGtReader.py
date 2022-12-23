import math
import pickle

import numpy as np

dataset_names = [
    'session0_center',
    'session0_left',
    'session0_right',
    'session1_center',
    'session1_left',
    'session1_right',
    'session2_center',
    'session2_left',
    'session2_right',
    'session3_center',
    'session3_left',
    'session3_right',
    'session4_center',
    'session4_left',
    'session4_right',
    'session5_center',
    'session5_left',
    'session5_right',
    'session6_center',
    'session6_left',
    'session6_right'
]

info_from_markers = {
    'session0_center':
        {
            'width': [4, 5, 6],
            'height': [0, 1],
            'plane_corners': [
                [0, 'p1'],
                [7, 'p1'],
                [6, 'p2'],
                [4, 'p1'],
            ],
            'goal_line': [[4, 'p1'], [6, 'p2']]
        },
    'session0_left':
        {
            'width': [8, 9, 10],
            'height': [7, 6, 5, 4],
            'plane_corners': [
                [0, 'p1'],
                [11, 'p1'],
                [10, 'p2'],
                [8, 'p1'],
            ],
            'goal_line': [[8, 'p1'], [10, 'p2']]
        },
    'session0_right':
        {
            'width': [4, 5, 6],
            'height': [3, 2],
            'plane_corners': [
                [0, 'p1'],
                [7, 'p1'],
                [6, 'p2'],
                [4, 'p1'],
            ],
            'goal_line': [[4, 'p1'], [6, 'p2']]
        },
    'session1_center':
        {
            'width': [3],
            'height': [1],
            'plane_corners': [
                [3, 'p2'],
                [3, 'p1'],
                [1, 'p1'],
                [2, 'p2'],
            ],
            'goal_line': [[2, 'p2'], [2, 'p1']]
        },
    'session1_left':
        {
            'width': [3],
            'height': [1],
            'plane_corners': [
                [3, 'p2'],
                [3, 'p1'],
                [2, 'p1'],
                [2, 'p2'],
            ],
            'goal_line': [[2, 'p2'], [2, 'p1']]
        },
    'session1_right':
        {
            'width': [3],
            'height': [1],
            'plane_corners': [
                [3, 'p2'],
                [3, 'p1'],
                [2, 'p1'],
                [2, 'p2'],
            ],
            'goal_line': [[2, 'p2'], [2, 'p1']]
        },
    'session2_center':
        {
            'width': [6],
            'height': [4],
            'plane_corners': [
                [5, 'p2'],
                [5, 'p1'],
                [6, 'p1'],
                [6, 'p2'],
            ],
            'goal_line': [[6, 'p2'], [6, 'p1']]
        },
    'session2_left':
        {
            'width': [6],
            'height': [4],
            'plane_corners': [
                [5, 'p2'],
                [5, 'p1'],
                [6, 'p1'],
                [6, 'p2'],
            ],
            'goal_line': [[6, 'p2'], [6, 'p1']]
        },
    'session2_right':
        {
            'width': [6],
            'height': [4],
            'plane_corners': [
                [5, 'p2'],
                [5, 'p1'],
                [6, 'p1'],
                [6, 'p2'],
            ],
            'goal_line': [[6, 'p2'], [6, 'p1']]
        },
    'session3_center':
        {
            'width': [9],
            'height': [7, 6, 5, 4],
            'plane_corners': [
                [8, 'p2'],
                [8, 'p1'],
                [9, 'p1'],
                [9, 'p2'],
            ],
            'goal_line': [[9, 'p2'], [9, 'p1']]
        },
    'session3_left':
        {
            'width': [9],
            'height': [7, 6, 5, 4],
            'plane_corners': [
                [8, 'p2'],
                [8, 'p1'],
                [9, 'p1'],
                [9, 'p2'],
            ],
            'goal_line': [[9, 'p2'], [9, 'p1']]
        },
    'session3_right':
        {
            'width': [9],
            'height': [7, 6, 5, 4],
            'plane_corners': [
                [8, 'p2'],
                [8, 'p1'],
                [9, 'p1'],
                [9, 'p2'],
            ],
            'goal_line': [[9, 'p2'], [9, 'p1']]
        },
    'session4_center':
        {
            'width': [8],
            'height': [6, 5, 4, 3],
            'plane_corners': [
                [7, 'p2'],
                [7, 'p1'],
                [8, 'p1'],
                [8, 'p2'],
            ],
            'goal_line': [[8, 'p2'], [8, 'p1']]
        },
    'session4_left':
        {
            'width': [8],
            'height': [6, 5, 4, 3],
            'plane_corners': [
                [7, 'p2'],
                [7, 'p1'],
                [8, 'p1'],
                [8, 'p2'],
            ],
            'goal_line': [[8, 'p2'], [8, 'p1']]
        },
    'session4_right':
        {
            'width': [8],
            'height': [6, 5, 4, 3],
            'plane_corners': [
                [7, 'p2'],
                [7, 'p1'],
                [8, 'p1'],
                [8, 'p2'],
            ],
            'goal_line': [[8, 'p2'], [8, 'p1']]
        },
    'session5_center':
        {
            'width': [7],
            'height': [5],
            'plane_corners': [
                [6, 'p2'],
                [6, 'p1'],
                [7, 'p1'],
                [7, 'p2'],
            ],
            'goal_line': [[7, 'p2'], [7, 'p1']]
        },
    'session5_left':
        {
            'width': [7],
            'height': [5],
            'plane_corners': [
                [6, 'p2'],
                [6, 'p1'],
                [7, 'p1'],
                [7, 'p2'],
            ],
            'goal_line': [[7, 'p2'], [7, 'p1']]
        },

    'session5_right':
        {
            'width': [7],
            'height': [5],
            'plane_corners': [
                [6, 'p2'],
                [6, 'p1'],
                [7, 'p1'],
                [7, 'p2'],
            ],
            'goal_line': [[7, 'p2'], [7, 'p1']]
        },
    'session6_center':
        {
            'width': [7],
            'height': [5],
            'plane_corners': [
                [6, 'p2'],
                [6, 'p1'],
                [7, 'p1'],
                [7, 'p2'],
            ],
            'goal_line': [[7, 'p1'], [7, 'p2']]
        },
    'session6_left':
        {
            'width': [7],
            'height': [5],
            'plane_corners': [
                [6, 'p2'],
                [6, 'p1'],
                [7, 'p1'],
                [7, 'p2'],
            ],
            'goal_line': [[7, 'p1'], [7, 'p2']]
        },
    'session6_right':
        {
            'width': [7],
            'height': [5],
            'plane_corners': [
                [6, 'p2'],
                [6, 'p1'],
                [7, 'p1'],
                [7, 'p2'],
            ],
            'goal_line': [[7, 'p1'], [7, 'p2']]
        },
}


class BrnoCompSpeedGtReader:
    __gt_data = None

    def __init__(self, parameters_path: str, session_name: str):
        # Load parameters
        f = open(parameters_path, 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        self.__gt_data = u.load()
        self.__session_name = session_name

    def world_and_screen_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        markers = self.__gt_data['distanceMeasurement']
        print('# markers = ', len(markers))
        screen_coordinates = []
        world_coordinates = []
        info = info_from_markers[self.__session_name]
        width = 0
        for idx in info['width']:
            width += markers[idx]['distance']
        height = 0
        for idx in info['height']:
            height += markers[idx]['distance']
        plane_corners = info['plane_corners']
        for element in plane_corners:
            screen_coordinates.append(markers[element[0]][element[1]])
        world_coordinates.append((0, 0, 0))
        world_coordinates.append((width, 0, 0))
        world_coordinates.append((width, height, 0))
        world_coordinates.append((0, height, 0))
        screen_coordinates = np.array(screen_coordinates)
        screen_coordinates = screen_coordinates[:, :2]
        world_coordinates = np.array(world_coordinates)
        return screen_coordinates, world_coordinates

    def get_markers(self):
        return self.__gt_data['distanceMeasurement']

    def fps(self) -> float:
        return self.__gt_data['fps']

    def get_car_measured_in_this_frame(self, frame_num, lane_idx):
        time_position = frame_num / self.fps()
        # max_error = (1 / self.fps()) * 0.5  # Half a frame in seconds
        min_error = math.inf
        car_idx = -1
        for idx, car in enumerate(self.__gt_data['cars']):
            if lane_idx in car['laneIndex']:
                for intersection in car['intersections']:
                    if intersection['measurementLineId'] == 0:
                        error = abs(time_position - intersection['videoTime'])
                        if error < min_error:
                            min_error = error
                            car_idx = idx
        print('lane = ', lane_idx)
        print('time = ', time_position)
        print('Car found (', car_idx, ')')
        print(self.__gt_data['cars'][car_idx])

    def goal_line(self):
        markers = self.__gt_data['distanceMeasurement']
        info = info_from_markers[self.__session_name]
        p0_idx, p0_extremum = info['goal_line'][0]
        p1_idx, p1_extremum = info['goal_line'][1]
        p0 = markers[p0_idx][p0_extremum][:2]
        p1 = markers[p1_idx][p1_extremum][:2]
        return p0, p1

    def get_lane_count(self):
        return len(self.__gt_data['laneDivLines']) - 1

    def save_cars_to_csv(self, path):
        with open(path, 'w') as f:
            f.write('id,goal_time,lane,speed\n')
            for car in self.__gt_data['cars']:
                f.write(str(car['carId']) + ',')
                for intersection in car['intersections']:
                    if intersection['measurementLineId'] == 0:
                        f.write(str(intersection['videoTime']) + ',')
                f.write(str(car['laneIndex']) + ',')
                f.write(str(car['speed']) + '\n')

    def get_min_time(self):
        """ Two second buffer """
        return max(float(self.__gt_data['cars'][0]['intersections'][0]['videoTime']) - 2, 0.0)

    def get_max_time(self):
        """ Two second buffer """
        return float(self.__gt_data['cars'][-1]['intersections'][-1]['videoTime'] + 2)
