import os

import numpy as np

from multi_visual_dash.dash_viz.data import DashMessage
from multi_visual_dash.dataloaders.waymo.motion_utils import get_road_scatters, get_car_rect_scatters, \
    get_trajectory_scatters, get_light_scatters, get_pred_trajectory_scatters
from multi_visual_dash.dataloaders.waymo.feattures_description import generate_features_description

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class WaymoMotionDataLoader:
    def __init__(self, tfrecord_path: str, save_dir: str = None):
        self.dataset = tf.data.TFRecordDataset([tfrecord_path], num_parallel_reads=1)
        self.save_dir = save_dir

    def __call__(self):
        for data_id, data in enumerate(self.dataset.as_numpy_iterator()):
            data = tf.io.parse_single_example(data, generate_features_description())
            yield self.data2plot_data(data, self.save_dir)

    @staticmethod
    def data2plot_data(data, pred_traj_data: dict = None, save_dir: str = None):
        viz_massage = DashMessage('new_plot', str(data['scenario/id'].numpy().astype(str)[0]))
        if save_dir is not None:
            viz_massage.save_dir = save_dir

        road_scatters = get_road_scatters(data)
        viz_massage.scatters.extend(road_scatters)
        light_scatters = get_light_scatters(data)
        viz_massage.scatters.extend(light_scatters)
        car_rect_scatters = get_car_rect_scatters(data)
        viz_massage.scatters.extend(car_rect_scatters)

        if pred_traj_data is not None:
            trajectory_scatters = get_trajectory_scatters(data, [pred_traj_data['agent_id']])
            viz_massage.scatters.extend(trajectory_scatters)
            trajectory_scatters = get_pred_trajectory_scatters(
                data, pred_traj_data['coords'], pred_traj_data['probas'], pred_traj_data['agent_id'])
            viz_massage.scatters.extend(trajectory_scatters)
        else:
            trajectory_scatters = get_trajectory_scatters(data)
            viz_massage.scatters.extend(trajectory_scatters)

        return viz_massage

    def get_data_by_scenario_id(self, scenario_id: str) -> dict:
        for data_id, data in enumerate(self.dataset.as_numpy_iterator()):
            data = tf.io.parse_single_example(data, generate_features_description())
            if str(data['scenario/id'].numpy().astype(str)[0]) == scenario_id:
                return data

    def get_plot_data_with_pred(
            self,
            scenario_id: str,
            coords: np.array,
            probas: np.array,
            agent_id: int,
            save_dir: str = None
    ) -> DashMessage:
        """
        :param coords: [samples; points; x, y]
        :param probas: [samples]
        :return: plot_data: DashMessage
        """
        data = self.get_data_by_scenario_id(scenario_id)
        pred_traj_data = {'coords': coords, 'probas': probas, 'agent_id': agent_id}
        return self.data2plot_data(data, pred_traj_data, save_dir)
