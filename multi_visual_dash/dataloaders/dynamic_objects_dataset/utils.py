from typing import Union

import numpy as np
import open3d as o3d

from multi_visual_dash.dataloaders.utils import get_bbox, get_rot_matrix


def get_oriented_bbox_and_points(bbox_label):
    bbox = get_bbox(bbox_label['center'][0], bbox_label['center'][1], bbox_label['center'][2],
                    bbox_label['extent'][0], bbox_label['extent'][1], bbox_label['extent'][2],
                    bbox_label['heading'], 0, 0)
    # return o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(bbox)), bbox

    rot_matrix = get_rot_matrix(bbox_label['heading'], 0, 0)
    return o3d.geometry.OrientedBoundingBox(bbox_label['center'], rot_matrix, bbox_label['extent']), bbox


def get_dynamic_points(data: dict, label_ids: Union[list, np.ndarray] = None):
    if label_ids is None:
        label_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13]

    points_all = data['points_all']
    point_labels_all = data['point_labels_all']

    point_types = np.unique(point_labels_all[:, 1])
    points = []
    for point_type in point_types:
        if point_type in label_ids:
            ids = np.where(point_labels_all == point_type)[0]
            points.append({
                'label_id': point_type,
                'points': points_all[ids, :]}
            )
    return points


def center_points(points: np.ndarray, bbox: dict):
    points -= bbox['center'].reshape(1, -1)
    return np.transpose(np.linalg.inv(bbox['rot_matrix']) @ np.transpose(points))
