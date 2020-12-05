'''
Useful functions
'''
import copy
from importlib import import_module

import ujson
import json # we need this for dumping nans
import logging
import os
import pickle
import sys
import torch


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def exists_dir(d):
    return os.path.exists(d)

def dump_json_file(filename, contents):
    with open(filename, 'w') as f:
        try:
            ujson.dump(contents, f)
        except OverflowError:
            json.dump(contents, f)

def load_json_file(filename):
    with open(filename, 'r') as f:
        contents = ujson.load(f)
    return contents

def dump_pickle_file(filename, contents):
    with open(filename, 'wb') as f:
        pickle.dump(contents, f)

def load_pickle_file(filename):
    with open(filename, 'rb') as f:
        contents = pickle.load(f)
    return contents

def flatten(arr):
    return [item for sublist in arr for item in sublist]

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size