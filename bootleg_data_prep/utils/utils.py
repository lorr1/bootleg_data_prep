'''
Useful functions
'''
from itertools import islice, chain

import ujson
import json # we need this for dumping nans
import os
import pickle
import sys

from bootleg_data_prep.language import ENSURE_ASCII


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def exists_dir(d):
    return os.path.exists(d)

def dump_json_file(filename, contents):
    with open(filename, 'w', encoding='utf8') as f:
        try:
            ujson.dump(contents, f, ensure_ascii=ENSURE_ASCII)
        except OverflowError:
            json.dump(contents, f, ensure_ascii=ENSURE_ASCII)
        except TypeError:
            json.dump(contents, f, ensure_ascii=ENSURE_ASCII)

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


def chunks(iterable, n):
   """chunks(ABCDE,2) => AB CD E"""
   iterable = iter(iterable)
   while True:
       try:
           yield chain([next(iterable)], islice(iterable, n-1))
       except StopIteration:
           return None

def chunk_file(in_file, out_dir, num_lines, prefix="out_"):
    ensure_dir(out_dir)
    out_files = {}
    total_lines = 0
    ending = os.path.splitext(in_file)[1]
    with open(in_file) as bigfile:
        i = 0
        while True:
            try:
                lines = next(chunks(bigfile, num_lines))
            except StopIteration:
                break
            except RuntimeError:
                break
            file_split = os.path.join(out_dir, f"{prefix}{i}{ending}")
            total_file_lines = 0
            i += 1
            with open(file_split, 'w', encoding='utf8') as f:
                while True:
                    try:
                        line = next(lines)
                    except StopIteration:
                        break
                    total_lines += 1
                    total_file_lines += 1
                    f.write(line)
            out_files[file_split] = total_file_lines
    return total_lines, out_files
