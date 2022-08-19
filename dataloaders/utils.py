from functools import cached_property

import numpy as np


def rotate_bbox(center_x, center_y, center_z, length, width, height, yaw, pitch, roll):
    xyz = np.array([center_x, center_y, center_z]).reshape(-1, 1)

    r_y = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    r_b = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    r_a = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    l, w, h = length / 2, width / 2, height / 2

    box_up = np.array([[l, l, -l, -l],
                       [-w, w, w, -w],
                       [h, h, h, h]])
    box_down = np.array([[l, l, -l, -l],
                         [-w, w, w, -w],
                         [-h, -h, -h, -h]])
    box_up = (r_y @ r_b @ r_a) @ box_up
    box_up = xyz + np.append(box_up, box_up[:, 0].reshape(-1, 1), axis=1)
    box_down = (r_y @ r_b @ r_a) @ box_down
    box_down = xyz + np.append(box_down, box_down[:, 0].reshape(-1, 1), axis=1)

    box = np.hstack([
        box_down,
        box_up[:, :2], box_down[:, 1].reshape(-1, 1),
        box_up[:, 1:3], box_down[:, 2].reshape(-1, 1),
        box_up[:, 2:4], box_down[:, 3].reshape(-1, 1),
        box_up[:, 3:]
    ])
    return box
