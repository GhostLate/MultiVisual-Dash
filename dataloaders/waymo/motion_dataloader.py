import os

import numpy as np

from dataloaders.waymo.motion_utils import get_road_scatters, get_car_rect_scatters, get_trajectory_scatters, \
    get_light_scatters, get_pred_trajectory_scatters
from dataloaders.waymo.feattures_description import generate_features_description

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class WaymoMotionDataLoader:
    def __init__(self, tfrecord_path: str):
        self.dataset = tf.data.TFRecordDataset([tfrecord_path], num_parallel_reads=1)

    def __call__(self):
        for data_id, data in enumerate(self.dataset.as_numpy_iterator()):
            data = tf.io.parse_single_example(data, generate_features_description())
            yield self.data2plot_data(data)

    @staticmethod
    def data2plot_data(data, pred_traj_data: dict = None, save_dir: str = None):
        plot_data = {
            'command_type': 'add2plot',
            'plot_name': str(data['scenario/id'].numpy().astype(str)[0]),
            'scatters': [],
        }
        if save_dir is not None:
            plot_data['save_dir'] = save_dir

        road_scatters = get_road_scatters(data)
        plot_data['scatters'].extend(road_scatters)
        light_scatters = get_light_scatters(data)
        plot_data['scatters'].extend(light_scatters)
        car_rect_scatters = get_car_rect_scatters(data)
        plot_data['scatters'].extend(car_rect_scatters)

        if pred_traj_data is not None:
            trajectory_scatters = get_trajectory_scatters(data, [pred_traj_data['agent_id']])
            plot_data['scatters'].extend(trajectory_scatters)
            trajectory_scatters = get_pred_trajectory_scatters(
                data, pred_traj_data['coords'], pred_traj_data['probas'], pred_traj_data['agent_id'])
            plot_data['scatters'].extend(trajectory_scatters)
        else:
            trajectory_scatters = get_trajectory_scatters(data)
            plot_data['scatters'].extend(trajectory_scatters)

        return plot_data

    def get_data_by_scenario_id(self, scenario_id: str) -> dict:
        for data_id, data in enumerate(self.dataset.as_numpy_iterator()):
            data = tf.io.parse_single_example(data, generate_features_description())
            if str(data['scenario/id'].numpy().astype(str)[0]) == scenario_id:
                return data

    def get_plot_data_with_pred(self, scenario_id: str, coords: np.array, probas: np.array, agent_id: int, save_dir: str):
        """
        coords: [samples; points; x, y]
        probas: [samples]
        """
        data = self.get_data_by_scenario_id(scenario_id)
        pred_traj_data = {'coords': coords, 'probas': probas, 'agent_id': agent_id}
        plot_data = self.data2plot_data(data, pred_traj_data, save_dir)
        return plot_data
