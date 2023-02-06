import os
import pickle
import json
from os.path import isfile


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_obj(f_name, obj):
    with open(f_name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)


def read_json(f_name):
    with open(f_name, 'r') as f:
        return json.load(f)
