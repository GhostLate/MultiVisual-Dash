import numpy as np


def get_bbox(center_x, center_y, center_z, length, width, height, yaw, pitch, roll) -> np.ndarray:
    xyz = np.array([center_x, center_y, center_z])

    rot_matrix = get_rot_matrix(yaw, pitch, roll)

    l, w, h = length / 2, width / 2, height / 2

    box_up = np.array([[l, -w, h],
                       [l, w, h],
                       [-l, w, h],
                       [-l, -w, h]])
    box_down = np.array([[l, -w, -h],
                         [l, w, -h],
                         [-l, w, -h],
                         [-l, -w, -h]])
    bbox = xyz + np.transpose(rot_matrix @ np.transpose(np.vstack([box_down, box_up])))
    return bbox


def get_rot_matrix(yaw, pitch, roll):
    r_y = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0,            0,           1]])
    r_b = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                    [ 0,             1,             0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    r_a = np.array([[1,            0,             0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll),  np.cos(roll)]])
    return r_y @ r_b @ r_a


def bbox2plotly_drawing(bbox: np.ndarray) -> np.ndarray:
    bbox = np.vstack([
        bbox[:4], bbox[0, :].reshape(1, -1),
        bbox[4:6, :], bbox[1, :].reshape(1, -1),
        bbox[5:7, :], bbox[2, :].reshape(1, -1),
        bbox[6:, :], bbox[3, :].reshape(1, -1),
        bbox[7:, :], bbox[4, :].reshape(1, -1),
        np.array([np.nan, np.nan, np.nan]).reshape(1, -1)
    ])
    return bbox


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
