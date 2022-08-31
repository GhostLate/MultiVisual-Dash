import os
import pickle

from tqdm import tqdm

from multi_visual_dash.dataloaders.waymo.perception.dataloader import parse_raw_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class DynamicObjetsConverter:
    def __init__(self, tfrecord_dir: str, dataset_dir: str):
        self.tfrecord_dir = tfrecord_dir
        self.data_dir = dataset_dir
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        self.preprocess_dataset(tfrecord_dir, dataset_dir)

    def preprocess_dataset(self, tfrecord_dir: str, dataset_dir: str):
        file_names = os.listdir(tfrecord_dir)
        for file_name in tqdm(file_names):
            dataset = tf.data.TFRecordDataset(os.path.join(tfrecord_dir, file_name), compression_type='')
            data = {}
            for data_id, raw_data in enumerate(dataset):
                data[data_id] = preprocess_data if (preprocess_data := parse_raw_data(raw_data)) else None
            with open(os.path.join(dataset_dir, f'{os.path.splitext(file_name)[0]}.pickle'), 'wb') as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
