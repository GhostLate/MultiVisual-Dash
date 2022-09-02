import os

import numpy as np
from tqdm import tqdm

from multi_visual_dash.dataloaders.waymo.perception.utils import range_image2pc_labels

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def parse_raw_data(raw_data) -> dict:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(raw_data.numpy()))
    if frame.lasers[0].ri_return1.segmentation_label_compressed:
        (r_imgs, cam_project, seg_labels, r_img_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        points, _ = frame_utils.convert_range_image_to_point_cloud(frame, r_imgs, cam_project, r_img_pose)
        point_labels = range_image2pc_labels(frame, r_imgs, seg_labels)

        bboxes = [{
            'center': np.array([bbox_label.box.center_x, bbox_label.box.center_y, bbox_label.box.center_z]),
            'extent': np.array([bbox_label.box.length, bbox_label.box.width, bbox_label.box.height]),
            'heading': bbox_label.box.heading,
            'name': bbox_label.id,
            'label_id': bbox_label.type
        } for bbox_label in frame.laser_labels]

        return {
            'points_all': np.concatenate(points, axis=0),
            'point_labels_all': np.concatenate(point_labels, axis=0),
            'transform_matrix': np.array(frame.pose.transform).reshape(4, 4),
            'name': f'{int("".join(frame.context.name.split("_"))):x}',
            'bboxes': bboxes,
            'timestamp_micros': frame.timestamp_micros,
        }


class WaymoPerceptionDataLoader:
    def __init__(self, tfrecord_dir: str):
        self.tfrecord_dir = tfrecord_dir

    def __call__(self):
        file_names = os.listdir(self.tfrecord_dir)
        for file_name in tqdm(file_names):
            dataset = tf.data.TFRecordDataset(os.path.join(self.tfrecord_dir, file_name), compression_type='')
            for data_id, raw_data in enumerate(dataset):
                if (parsed_data := parse_raw_data(raw_data)):
                    yield parsed_data