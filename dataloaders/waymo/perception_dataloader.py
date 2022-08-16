import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


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
                point_labels = self.range_image2pc_labels(frame, range_images, segmentation_labels)

                # 3d points in vehicle frame.
                points_all = np.concatenate(points, axis=0)
                # point labels.
                point_labels_all = np.concatenate(point_labels, axis=0)
                # camera projection corresponding to each point.
                cp_points_all = np.concatenate(cp_points, axis=0)
                data = {'points_all': points_all, 'point_labels_all': point_labels_all, 'cp_points_all': cp_points_all,
                        'name': f'{"_".join(frame.context.name.split("_")[3:])}_{data_id}',
                        'bbox_labels': frame.laser_labels}
                yield self.frame2plot_data(data) #, './samples')

    @staticmethod
    def frame2plot_data(data, save_dir: str = None):
        plot_data = {
            'command_type': 'add2plot',
            'plot_name': data['name'],
            'scatters': [],
        }
        if save_dir is not None:
            plot_data['save_dir'] = save_dir

        point_scatters = get_point_scatters(data)
        plot_data['scatters'].extend(point_scatters)
        point_scatters = get_bbox_scatters(data)
        plot_data['scatters'].extend(point_scatters)
        return plot_data

    @staticmethod
    def range_image2pc_labels(frame, range_images, segmentation_labels, ri_index=0):
        """
        Convert segmentation labels from range images to point clouds.

        Args:
          frame: open dataset frame
          range_images: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
          segmentation_labels: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
          ri_index: 0 for the first return, 1 for the second return.

        Returns:
          point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
            points that are not labeled.
        """

        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        point_labels = []
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0

            if c.name in segmentation_labels:
                sl = segmentation_labels[c.name][ri_index]
                sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
                sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            else:
                num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
                sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

            point_labels.append(sl_points_tensor.numpy())
        return point_labels


def get_bbox_scatters(data):
    scatters = []
    for bbox_label in data['bbox_labels']:
        agent_rect_x = bbox_label.box.center_x
        agent_rect_y = bbox_label.box.center_y
        agent_rect_z = bbox_label.box.center_z
        agent_rect_height = bbox_label.box.height
        agent_rect_length = bbox_label.box.length
        agent_rect_width = bbox_label.box.width
        agent_name = bbox_label.id
        agent_type = bbox_label.type
        xyz = np.array([agent_rect_x, agent_rect_y, agent_rect_z]).reshape(-1, 1)
        yaw = 0
        pitch = 0
        roll = 0

        r_y = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        r_b = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        r_a = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        l, w, h = agent_rect_length / 2, agent_rect_width / 2, agent_rect_height / 2
        box_up = np.array([[l, l, -l, -l],
                           [-w, w, w, -w],
                           [h, h, h, h]])
        box_down = np.array([[l, l, -l, -l],
                             [-w, w, w, -w],
                             [-h, -h, -h, -h]])
        box_up = (r_y @ r_b @ r_a) @ box_up
        box_up = xyz + np.append(box_up, box_up[:, 0].reshape(-1, 1), axis=1)
        box_down = (r_y @ r_b @ r_a) @ box_down
        box_down = xyz + np.append(box_down, box_down[:, 0].reshape(-1, 1), axis=1)

        box = np.hstack([
            box_down,
            box_up[:, :2], box_down[:, 1].reshape(-1, 1),
            box_up[:, 1:3], box_down[:, 2].reshape(-1, 1),
            box_up[:, 2:4], box_down[:, 3].reshape(-1, 1),
            box_up[:, 3:]
        ])

        scatter = {
            'name': f'agent_{agent_name}',
            'mode': 'lines',
            'x': box[0, :],
            'y': box[1, :],
            'z': box[2, :],
            'line_size': 4,
            #'fill': True,
            'type': agent_type
        }
        scatters.append(scatter)
    return scatters


def get_point_scatters(data):
    points_all = data['points_all']
    point_labels_all = data['point_labels_all']

    types = np.unique(point_labels_all[:, 1])
    scatters = []
    for type in types:
        ids = np.where(point_labels_all == type)[0]
        x = points_all[ids, 0]
        y = points_all[ids, 1]
        z = points_all[ids, 2]
        scatter = {
            'name': f'points_{type}',
            'mode': 'markers',
            'x': x,
            'y': y,
            'z': z,
            'type': type,
            'marker_size': 2
        }
        scatters.append(scatter)
    return scatters
