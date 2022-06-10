import os
import scene_utils
import time
from scene_utils import scene_classifier

def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
            pass
    return paths


vid_paths = get_path('E:/test_videos')

print(vid_paths)
