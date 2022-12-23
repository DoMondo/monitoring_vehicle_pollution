import itertools
import cv2
import imageio as imageio
import numpy as np
import BrnoCompSpeedGtReader


class CameraCalibration:
    screen_coordinates = []
    __world_coordinates = []
    __screen_to_world_matrix = None
    __world_to_screen_matrix = None
    __img = None
    __top_view_matrix = None
    __goal_line_screen_coordinates = None
    __lanes = None
    __screen_w = None
    __screen_h = None

    def __init__(self, screen_coordinates, world_coordinates, goal_line_screen_coordinates, screen_w, screen_h):
        self.screen_coordinates = screen_coordinates
        self.__world_coordinates = world_coordinates
        self.__screen_to_world_matrix, _ = cv2.findHomography(self.screen_coordinates, self.__world_coordinates)
        self.__world_to_screen_matrix, _ = cv2.findHomography(self.__world_coordinates, self.screen_coordinates)
        self.__goal_line_screen_coordinates = goal_line_screen_coordinates
        self.__screen_w = screen_w
        self.__screen_h = screen_h

    def __find_vantage(self, image, point_left, point_right, positive: bool):
        """ Iteratively finds a point that is close to the border of the image """
        current_point_left = point_left.copy()
        current_point_right = point_right.copy()
        destination = 0
        threshold = 50
        delta = 1
        if not positive:
            destination = image.shape[:2]
        while np.abs(current_point_left[1] - image.shape[0]) > threshold and \
                np.abs(current_point_right[1] - image.shape[0]) > threshold and \
                np.abs(current_point_left[0] - image.shape[1]) > threshold and \
                np.abs(current_point_left[0] - image.shape[1]) > threshold and \
                current_point_left[1] > threshold and \
                current_point_right[1] > threshold and \
                current_point_left[0] > threshold and \
                current_point_left[0] > threshold:
            current_point_left_world = self.get_xyz(current_point_left[0], current_point_left[1])
            if positive:
                current_point_left_world[1] += delta
            else:
                current_point_left_world[1] -= delta
            current_point_left = self.get_ji(current_point_left_world[0], current_point_left_world[1])
            current_point_right_world = self.get_xyz(current_point_right[0], current_point_right[1])
            if positive:
                current_point_right_world[1] += delta
            else:
                current_point_right_world[1] -= delta
            current_point_right = self.get_ji(current_point_right_world[0], current_point_right_world[1])
        return current_point_left, current_point_right

    def generate_top_view(self, image, width, height):
        if self.__top_view_matrix is None:
            top_left = self.screen_coordinates[0]
            top_right = self.screen_coordinates[1]
            bottom_right = self.screen_coordinates[2]
            bottom_left = self.screen_coordinates[3]
            top_left, top_right = self.__find_vantage(image, top_left, top_right, False)
            bottom_left, bottom_right = self.__find_vantage(image, bottom_left, bottom_right, True)
            new_top_left = (0, 0)
            new_top_right = (width, 0)
            new_bottom_left = (0, height)
            new_bottom_right = (width, height)
            image = image.copy()
            from_pts = np.array([top_left, top_right, bottom_left, bottom_right])
            to_pts = np.array([new_top_left, new_top_right, new_bottom_left, new_bottom_right])
            self.__top_view_matrix, _ = cv2.findHomography(from_pts, to_pts)
        draw = cv2.warpPerspective(image, self.__top_view_matrix, (width, height))
        return draw

    def get_xyz_top_view(self, j, i):
        result = np.matmul(self.__top_view_matrix, (j, i, 1))
        return result[:2] / result[2]

    def get_xyz(self, j, i):
        result = np.matmul(self.__screen_to_world_matrix, (j, i, 1))
        return result[:2] / result[2]

    def get_ji(self, x, y):
        result = np.matmul(self.__world_to_screen_matrix, (x, y, 1))
        return np.uint16(result[:2] / result[2])

    def get_goal_line(self):
        return self.__goal_line_screen_coordinates[0].copy(), self.__goal_line_screen_coordinates[1].copy()

    def get_centroid_position(self, bbox):
        return np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]) * (self.__screen_w, self.__screen_h)

    def get_road_position(self, bbox):
        return np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3]]) * (self.__screen_w, self.__screen_h)