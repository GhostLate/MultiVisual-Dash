import os
import pickle

import numpy as np
import open3d as o3d

from multi_visual_dash.dataloaders.dynamic_objects_dataset.utils import get_dynamic_points, get_oriented_bboxes


class DynamicObjectsDataLoader:
    post_data: dict

    def __init__(self, dataset_dir: str):
        self.data_dir = dataset_dir

    def get_filtered_data(self, min_pc_points: int, max_dist2bbox: int):
        self.post_data = dict()
        file_names = os.listdir(self.data_dir)
        for file_name in file_names:
            with open(os.path.join(self.data_dir, file_name), 'rb') as file:
                data = pickle.load(file)
                for id in data:
                    self.filter_data(data[id], min_pc_points, max_dist2bbox)
        return self.post_data

    def filter_data(self, data, min_pc_points: int, max_dist2bbox: int):
        dynamic_points = get_dynamic_points(data)
        oriented_bboxes = get_oriented_bboxes(data)
        self.post_data.setdefault(data['name'], {})
        scene_data = self.post_data[data['name']]
        for dynamic_point in dynamic_points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dynamic_point['points'])
            for bbox_data in oriented_bboxes:
                if np.linalg.norm(bbox_data['oriented_bbox'].center) < max_dist2bbox:
                    inliers_indices = bbox_data['oriented_bbox'].get_point_indices_within_bounding_box(pcd.points)
                    inliers_points = np.asarray(pcd.select_by_index(inliers_indices, invert=False).points)
                    if inliers_points.shape[0] > min_pc_points:
                        scene_data.setdefault(bbox_data['agent_name'], {})
                        agent_data = scene_data[bbox_data['agent_name']]
                        agent_data.setdefault('points', {})
                        agent_data['points'].setdefault(data['timestamp_micros'], [])
                        agent_data['points'][data['timestamp_micros']].append({
                            'points': inliers_points,
                            'type': dynamic_point['type']
                        })
                        agent_data['bbox_type'] = bbox_data['agent_type']
        return data
