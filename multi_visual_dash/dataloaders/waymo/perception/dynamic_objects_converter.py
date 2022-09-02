import os
import pickle

import blosc
import numpy as np
import open3d as o3d
from tqdm import tqdm

from multi_visual_dash.dataloaders.dynamic_objects_dataset.utils import get_dynamic_points, get_oriented_bbox_and_points

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from multi_visual_dash.dataloaders.waymo.perception.dataloader import parse_raw_data


class DynamicObjetsConverter:
    def __init__(self, tfrecord_dir: str, save_dir: str):
        self.tfrecord_dir = tfrecord_dir
        self.data_dir = save_dir
        self.convert_dataset(tfrecord_dir, save_dir)

    def convert_dataset(self, tfrecord_dir: str, save_dir: str):
        file_names = os.listdir(tfrecord_dir)
        print(f"Start converting Waymo Perception Dataset... \nTotal files: {len(file_names)}")
        file_names_tqdm = tqdm(file_names, desc='Parsed: 0')
        for file_name in file_names_tqdm:
            dataset = tf.data.TFRecordDataset(os.path.join(tfrecord_dir, file_name), compression_type='')
            scene_samples = {}
            for scene_id, raw_data in enumerate(dataset):
                if (parsed_data := parse_raw_data(raw_data)) is not None:
                    scene_samples[scene_id] = parsed_data
                file_names_tqdm.set_description(f'Parsed: {scene_id}')
            processed_data = self.process_data(scene_samples)

            save_path = os.path.join(save_dir, 'dynamic_objects', f'{os.path.splitext(file_name)[0]}.pickle')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as file:
                pickle.dump(processed_data, file, protocol=pickle.HIGHEST_PROTOCOL)

            save_path = os.path.join(save_dir, 'scenes', f'{os.path.splitext(file_name)[0]}.dat')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as file:
                pickled_data = pickle.dumps(scene_samples, protocol=pickle.HIGHEST_PROTOCOL)
                compressed_pickle = blosc.compress(pickled_data)
                file.write(compressed_pickle)

    @staticmethod
    def process_data(scene_samples: dict):
        scenes_data = {}
        for scene_id, scene_sample in scene_samples.items():
            dynamic_points = get_dynamic_points(scene_sample)
            scenes_data.setdefault(scene_sample['name'], {})
            scene_data = scenes_data[scene_sample['name']]
            scene_data.setdefault('samples', {})
            for dynamic_point in dynamic_points:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(dynamic_point['points'])
                for bbox_label in scene_sample['bboxes']:
                    oriented_bbox, bbox_points = get_oriented_bbox_and_points(bbox_label)
                    inliers_indices = oriented_bbox.get_point_indices_within_bounding_box(pcd.points)
                    inliers_points = np.asarray(pcd.select_by_index(inliers_indices, invert=False).points)
                    if inliers_points.shape[0] > 0:
                        scene_data['samples'].setdefault(bbox_label['name'], {})
                        agent_data = scene_data['samples'][bbox_label['name']]
                        agent_data.setdefault('timestamps', {})
                        agent_data['timestamps'].setdefault(scene_sample['timestamp_micros'], {})
                        timestamp_data = agent_data['timestamps'][scene_sample['timestamp_micros']]
                        timestamp_data.setdefault('points_clouds', [])
                        timestamp_data.setdefault('bbox', {})
                        timestamp_data.setdefault('transform_matrix', scene_sample['transform_matrix'])

                        timestamp_data['points_clouds'].append({
                            'points': inliers_points,
                            'label_id': dynamic_point['label_id']
                        })
                        bbox_data = timestamp_data['bbox']
                        bbox_data.setdefault('label_id', bbox_label['label_id'])
                        bbox_data.setdefault('points', np.asarray(oriented_bbox.get_box_points()))
                        bbox_data.setdefault('original_points', bbox_points)
                        bbox_data.setdefault('center', oriented_bbox.center)
                        bbox_data.setdefault('rot_matrix', oriented_bbox.R)
                        bbox_data.setdefault('extent', oriented_bbox.extent)
        return scenes_data
