import copy

import numpy as np

from multi_visual_dash.dataloaders.dynamic_objects_dataset.dataloader import DynamicObjectsDataLoader


class PairsDataLoader:
    def __init__(self, dataset_dir: str, min_pc_points: int, max_dist2bbox: int, visualize_bboxes: bool = False):
        self.data_dir = dataset_dir
        self.dataloader = DynamicObjectsDataLoader(dataset_dir)
        self.min_pc_points = min_pc_points
        self.max_dist2bbox = max_dist2bbox
        self.visualize_bboxes = visualize_bboxes

    def __call__(self):
        for scenes in self.dataloader():
            f_scenes = self.dataloader.filter_scenes(scenes, self.min_pc_points, self.max_dist2bbox)
            t_scenes = self.dataloader.transform_scenes(f_scenes)
            c_scenes = self.dataloader.center_scenes(t_scenes)
            for scene_name, scene_data in c_scenes.items():
                for sample_id, sample_data in scene_data['samples'].items():
                    object_data_base = None
                    for timestamp, ts_data in sample_data['timestamps'].items():
                        ts_data['all_points'] = np.vstack([pc['points'] for pc in ts_data['points_clouds']])

                        meta_data = {'scene_name': scene_name, 'sample_id': sample_id, 'timestamp': timestamp}
                        object_data = {
                            'point_cloud': ts_data['all_points'],
                            'bbox': ts_data['bbox'],
                            'meta_data': meta_data
                        }

                        if object_data_base is None:
                            object_data_base = object_data
                        else:
                            yield copy.deepcopy(object_data_base), copy.deepcopy(object_data)
                            object_data_base = object_data
