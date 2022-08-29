import numpy as np


def rotate_bbox(center_x, center_y, center_z, length, width, height, yaw, pitch, roll) -> np.ndarray:
    xyz = np.array([center_x, center_y, center_z])

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

    box_up = np.array([[ l, -w, h],
                       [ l,  w, h],
                       [-l,  w, h],
                       [-l, -w, h]])
    box_down = np.array([[ l, -w, -h],
                         [ l,  w, -h],
                         [-l,  w, -h],
                         [-l, -w, -h]])
    box_up = np.transpose((r_y @ r_b @ r_a) @ np.transpose(box_up))
    box_up = xyz + np.append(box_up, box_up[0, :].reshape(1, -1), axis=0)
    box_down = np.transpose((r_y @ r_b @ r_a) @ np.transpose(box_down))
    box_down = xyz + np.append(box_down, box_down[0, :].reshape(1, -1), axis=0)

    box = np.vstack([
        box_down,
        box_up[:2, :], box_down[1, :].reshape(1, -1),
        box_up[1:3, :], box_down[2, :].reshape(1, -1),
        box_up[2:4, :], box_down[3, :].reshape(1, -1),
        box_up[3:, :]
    ])
    return box


def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    '''
    :param points: [points, dimensions]
    :param transform_matrix: [n, n], n >= 2
    :return: [points, dimensions]
    '''
    if transform_matrix.shape[0] == transform_matrix.shape[1] >= 2:
        if transform_matrix.shape[1] - 1 == points.shape[1]:
            points = np.append(points, np.ones(shape=(points.shape[0], 1)), axis=1)
        if transform_matrix.shape[1] == points.shape[1]:
            return np.transpose(transform_matrix @ np.transpose(points))
        else:
            raise ValueError("Wrong points dimension size")
    else:
        raise ValueError("Wrong transform_matrix size")
