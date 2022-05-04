import json
import numpy as np
import os

def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
            pass
    return paths

folder_paths = get_path('E:/labeled_data')
for path in folder_paths:
    score_json_paths = get_path(path)
    with open(score_json_paths[0], 'r') as score_json:
        frame_dict = json.load(score_json)
        print(np.array(frame_dict['frames'][0]['joint'], dtype=int).reshape((1,68)))