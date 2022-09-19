import copy
import os
import pickle

import blosc
import numpy as np

from multi_visual_dash.dataloaders.utils import transform_points
from multi_visual_dash.dataloaders.dynamic_objects_dataset.utils import center_points


class DynamicObjectsDataLoader:
    def __init__(self, dataset_dir: str):
        self.dynamic_objects_dir = os.path.join(dataset_dir, 'dynamic_objects/')
        self.scenes_dir = os.path.join(dataset_dir, 'scenes/')

    def __call__(self):
        file_names = os.listdir(self.dynamic_objects_dir)
        for file_name in file_names:
            with open(os.path.join(self.dynamic_objects_dir, file_name), 'rb') as file:
                scenes = pickle.load(file)
                for scene_name, scene_data in scenes.items():
                    yield {scene_name: scene_data}

    def transform_scenes(self, scenes: dict):
        t_scenes = dict()
        for scene_name, scene_data in scenes.items():
            t_scenes[scene_name] = self.transform_scene_samples(scene_data)
        return t_scenes

    @staticmethod
    def transform_scene_samples(scene_data: dict):
        t_scene_data = copy.deepcopy(scene_data)
        for agent_id, sample_data in t_scene_data['samples'].items():
            for timestamp, ts_data in sample_data['timestamps'].items():
                ts_data['inv_transform_matrix'] = np.linalg.inv(ts_data['transform_matrix'])
                for points_cloud in ts_data['points_clouds']:
                    points_cloud['points'] = transform_points(points_cloud['points'], ts_data['transform_matrix'])
                ts_data['bbox']['original_points'] = transform_points(
                    ts_data['bbox']['original_points'], ts_data['transform_matrix'])
                ts_data['bbox']['points'] = transform_points(ts_data['bbox']['points'], ts_data['transform_matrix'])
                transform_matrix = np.eye(4)
                transform_matrix[:3, 3] = ts_data['bbox']['center']
                transform_matrix[:3, :3] = ts_data['bbox']['rot_matrix']
                transform_matrix = ts_data['transform_matrix'] @ transform_matrix
                ts_data['bbox']['center'] = transform_matrix[:3, 3]
                ts_data['bbox']['rot_matrix'] = transform_matrix[:3, :3]
        return t_scene_data

    def center_scenes(self, scenes: dict):
        c_scenes = dict()
        for scene_name, scene_data in scenes.items():
            c_scenes[scene_name] = self.center_scene_samples(scene_data)
        return c_scenes

    @staticmethod
    def center_scene_samples(scene_data: dict):
        c_scene_data = copy.deepcopy(scene_data)
        for agent_id, sample_data in c_scene_data['samples'].items():
            for timestamp, ts_data in sample_data['timestamps'].items():
                for points_cloud in ts_data['points_clouds']:
                    points_cloud['points'] = center_points(points_cloud['points'], ts_data['bbox'])
                ts_data['bbox']['original_points'] = center_points(ts_data['bbox']['original_points'], ts_data['bbox'])
                ts_data['bbox']['points'] = center_points(ts_data['bbox']['points'], ts_data['bbox'])
                ts_data['bbox']['center'] = np.zeros(3)
                ts_data['bbox']['rot_matrix'] = np.eye(3)
        return c_scene_data

    def filter_scenes(self, scenes: dict, min_pc_points: int, max_dist2bbox: int):
        f_scenes = dict()
        for scene_name, scene_data in scenes.items():
            f_scene_data = self.filter_scene_samples(scene_data, min_pc_points, max_dist2bbox)
            if len(f_scene_data['samples'].keys()) > 0:
                f_scenes[scene_name] = f_scene_data
        return f_scenes

    @staticmethod
    def filter_scene_samples(scene_data: dict, min_pc_points: int, max_dist2bbox: int):
        f_scene_data = {'samples': {}}
        for agent_id, sample_data in scene_data['samples'].items():
            f_sample_data = dict(timestamps=dict())
            for timestamp, ts_data in sample_data['timestamps'].items():
                if np.linalg.norm(ts_data['bbox']['center']) < max_dist2bbox:
                    f_timestamp_data = dict(
                        bbox=ts_data['bbox'],
                        points_clouds=[pc for pc in ts_data['points_clouds'] if pc['points'].shape[0] > min_pc_points],
                        transform_matrix=ts_data['transform_matrix']
                    )
                    if len(f_timestamp_data['points_clouds']) > 0:
                        f_sample_data['timestamps'][timestamp] = f_timestamp_data
            if len(f_sample_data['timestamps'].keys()) > 0:
                f_scene_data['samples'][agent_id] = f_sample_data
        return f_scene_data

    def get_full_scenes_data(self):
        file_names = os.listdir(self.scenes_dir)
        for file_name in file_names:
            with open(os.path.join(self.scenes_dir, file_name), "rb") as f:
                compressed_pickle = f.read()
                depressed_pickle = blosc.decompress(compressed_pickle)
                scenes_data = pickle.loads(depressed_pickle)
                yield scenes_data
