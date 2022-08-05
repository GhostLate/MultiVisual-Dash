import json
import time
import numpy as np
import os
from tqdm import tqdm

from feattures_description import generate_features_description

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def get_slices(samples_id) -> list:
    start_val = samples_id[0]
    start_id = 0
    vals = []
    for idx, val in enumerate(samples_id[1:]):
        if val != start_val:
            vals.append([start_val, start_id, idx + 1])
            start_val = val
            start_id = idx + 1
    vals.append([start_val, start_id, samples_id.shape[0]])
    return vals


def remove_unvalid_data(valid: np.array, data: np.array) -> np.array:
    return data[np.where(valid == 1)]


def filter_valid_data(valid: np.array, data: np.array) -> list:
    data = data.tolist()
    indexes = np.where(valid == 0)[0].tolist()
    for i in indexes:
        data[i] = None
    return data


def get_road_scatters(data: dict) -> list:
    roads_type = np.squeeze(data['roadgraph_samples/type'].numpy())
    roads_valid = np.squeeze(data['roadgraph_samples/valid'].numpy())
    roads_xyz = data['roadgraph_samples/xyz'].numpy()

    ids_slices = get_slices(np.squeeze(data['roadgraph_samples/id'].numpy()))
    scatters = []
    for ids_slice in ids_slices:
        if ids_slice[0] >= 0:
            x = filter_valid_data(roads_valid[ids_slice[1]: ids_slice[2]], roads_xyz[ids_slice[1]: ids_slice[2], 0])
            y = filter_valid_data(roads_valid[ids_slice[1]: ids_slice[2]], roads_xyz[ids_slice[1]: ids_slice[2], 1])
            scatter = {
                'name': f'road_line_{ids_slice[0]}',
                'mode': 'lines+markers',
                'x': x,
                'y': y,
                'type': roads_type[ids_slice[0]],
                'size': 2
            }
            scatters.append(scatter)
    return scatters


def get_light_scatters(data: dict) -> list:
    lights_state = np.vstack([data['traffic_light_state/current/state'].numpy(), data['traffic_light_state/past/state'].numpy()])
    lights_valid = np.squeeze(data['traffic_light_state/current/valid'].numpy())
    lights_x = np.squeeze(data['traffic_light_state/current/x'].numpy())
    lights_y = np.squeeze(data['traffic_light_state/current/y'].numpy())
    scatters = []
    for idx in np.where(lights_valid == 1)[0].tolist():
        scatter = {
            'name': f'lights_{idx}',
            'mode': 'markers',
            'x': [lights_x[idx]],
            'y': [lights_y[idx]],
            'type': 'light',
            'state': lights_state[idx],
            'size': 9
        }
        scatters.append(scatter)
    return scatters


class WaymoDataLoader:
    def __init__(self, tfrecord_path: str):
        self.dataset = tf.data.TFRecordDataset([tfrecord_path], num_parallel_reads=1)

    def __call__(self):
        for data_id, data in enumerate(tqdm(self.dataset.as_numpy_iterator())):
            plot_data = {
                'command_type': 'add2plot',
                'plot_name': data_id,
                'scatters': []
            }
            data = tf.io.parse_single_example(data, generate_features_description())
            road_scatters = get_road_scatters(data)
            plot_data['scatters'].extend(road_scatters)
            light_scatters = get_light_scatters(data)
            plot_data['scatters'].extend(light_scatters)
            yield plot_data
