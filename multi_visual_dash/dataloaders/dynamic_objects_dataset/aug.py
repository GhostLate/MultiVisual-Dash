import copy

import numpy as np
from scipy.spatial.transform import Rotation

from multi_visual_dash.dataloaders.utils import transform_points


def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> np.ndarray:
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform


def apply_transform(src_object_data: dict, transform_matrix, scale: float = 1):
    object_data = copy.deepcopy(src_object_data)
    object_data['point_cloud'] = transform_points(object_data['point_cloud'] * scale, transform_matrix)
    object_data['bbox']['points'] = transform_points(object_data['bbox']['points'] * scale, transform_matrix)
    object_data['bbox']['original_points'] = transform_points(object_data['bbox']['original_points'] * scale,
                                                              transform_matrix)
    curr_matrix = np.eye(4)
    curr_matrix[:3, 3] = object_data['bbox']['center'] * scale
    curr_matrix[:3, :3] = object_data['bbox']['rot_matrix']
    curr_matrix = transform_matrix @ curr_matrix
    object_data['bbox']['center'] = curr_matrix[:3, 3]
    object_data['bbox']['rot_matrix'] = curr_matrix[:3, :3]
    return object_data
