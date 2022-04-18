import preprocess
from scene_utils import scene_classifier
import video_download as vd
import os


def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
            pass
    return paths


vid_infos = vd.get_vid_paths('video.csv')
finish_download = vd.download_vid(vid_infos)
# finish_download = True

# if finish_download:
#     vid_paths = get_path('../inputs/full_game_1080p')
#     # vid_path = '../inputs/full_game_1080p/CTC_A_jump.mp4'
#     for vid_path in vid_paths:
#         print(vid_path)
#         success, vid_info = preprocess.video_preprocess(vid_path)
#         print(success, vid_info)