import os
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from multi_visual_dash.dataloaders.waymo.motion.feattures_description import generate_features_description


class WaymoMotionDataLoader:
    def __init__(self, tfrecord_dir: str):
        self.tfrecord_dir = tfrecord_dir

    def __call__(self):
        file_names = os.listdir(self.tfrecord_dir)
        for file_name in tqdm(file_names):
            dataset = tf.data.TFRecordDataset(os.path.join(self.tfrecord_dir, file_name), num_parallel_reads=1)
            for data_id, data in enumerate(dataset.as_numpy_iterator()):
                yield tf.io.parse_single_example(data, generate_features_description())

    def get_data_by_scenario_id(self, scenario_id: str, tfrecord_name: str) -> dict:
        dataset = tf.data.TFRecordDataset(os.path.join(self.tfrecord_dir, tfrecord_name), num_parallel_reads=1)
        for data_id, data in enumerate(dataset.as_numpy_iterator()):
            data = tf.io.parse_single_example(data, generate_features_description())
            if str(data['scenario/id'].numpy().astype(str)[0]) == scenario_id:
                return data
