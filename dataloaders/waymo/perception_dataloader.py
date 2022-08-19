import os
import numpy as np

from dash_viz.data import DashMessage, SceneCamera
from utils import timing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from dataloaders.waymo.perception_utils import get_point_scatters, get_bbox_scatters, range_image2pc_labels


class WaymoPerceptionDataLoader:
    def __init__(self, tfrecord_path: str):
        self.dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

    def __call__(self):
        for data_id, data in enumerate(self.dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if frame.lasers[0].ri_return1.segmentation_label_compressed:
                (range_images, camera_projections, segmentation_labels,
                 range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

                points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                    frame, range_images, camera_projections, range_image_top_pose)
                point_labels = range_image2pc_labels(frame, range_images, segmentation_labels)

                # 3d points in vehicle frame.
                points_all = np.concatenate(points, axis=0)
                # point labels.
                point_labels_all = np.concatenate(point_labels, axis=0)
                # camera projection corresponding to each point.
                cp_points_all = np.concatenate(cp_points, axis=0)
                data = {'points_all': points_all, 'point_labels_all': point_labels_all, 'cp_points_all': cp_points_all,
                        #'name': f'{"_".join(frame.context.name.split("_")[3:])}_{data_id}',
                        'name': f'{"_".join(frame.context.name.split("_")[3:])}',
                        'bbox_labels': frame.laser_labels}
                yield self.frame2plot_data(data) #, './samples')

    @staticmethod
    def frame2plot_data(data, save_dir: str = None):
        scene_camera = SceneCamera()
        scene_camera.up.z = 1
        scene_camera.eye.x = -0.5
        scene_camera.eye.z = 0.1
        scene_camera.center.z = 0.05
        viz_massage = DashMessage('new_plot', data['name'])
        viz_massage.scene_camera = scene_camera
        if save_dir is not None:
            viz_massage.save_dir = save_dir

        point_scatters = get_point_scatters(data)
        viz_massage.scatters.extend(point_scatters)
        point_scatters = get_bbox_scatters(data)
        viz_massage.scatters.extend(point_scatters)
        return dict(viz_massage)

