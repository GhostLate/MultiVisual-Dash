import json
import pickle

import blosc2
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def compress_message(message):
    pickled_data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_pickle = blosc2.compress2(pickled_data)
    return compressed_pickle.decode('latin-1')


def decompress_message(compressed_message):
    compressed_pickle = compressed_message.encode('latin-1')
    pickled_data = blosc2.decompress2(compressed_pickle)
    return pickle.loads(pickled_data)
