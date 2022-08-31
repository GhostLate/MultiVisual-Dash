import os
import numpy as np
import open3d as o3d
from multi_visual_dash.dataloaders.utils import transform_points, get_bbox, get_rot_matrix


def get_oriented_bboxes(data, transform_matrix: np.ndarray = None):
    oriented_bboxes = []
    for bbox_label in data['bboxes']:
        bbox = get_bbox(bbox_label['center'][0], bbox_label['center'][1], bbox_label['center'][2],
                        bbox_label['extent'][0], bbox_label['extent'][1], bbox_label['extent'][2],
                        bbox_label['heading'], 0, 0)

        if transform_matrix is not None:
            bbox = transform_points(bbox, transform_matrix)

        # oriented_bbox = o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(bbox))

        rot_matrix = get_rot_matrix(bbox_label['heading'], 0, 0)
        oriented_bbox = o3d.geometry.OrientedBoundingBox(bbox_label['center'], rot_matrix, bbox_label['extent'])

        oriented_bboxes.append({
            'agent_name': bbox_label['name'],
            'agent_type': bbox_label['type'],
            'oriented_bbox': oriented_bbox
        })
    return oriented_bboxes


def get_dynamic_points(data, transform_matrix: np.ndarray = None):
    points_all = data['points_all']
    point_labels_all = data['point_labels_all']

    point_types = np.unique(point_labels_all[:, 1])
    if transform_matrix is not None:
        points_all = transform_points(points_all, transform_matrix)
    points = []
    for point_type in point_types:
        if point_type in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13]:
            ids = np.where(point_labels_all == point_type)[0]
            points.append({
                'type': point_type,
                'points': points_all[ids, :]}
            )
    return points
