import pickle
import json
import numpy as np


def open_pkl_file(base_path):
    with open(base_path, 'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
