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

if finish_download:
    print('Start resolving...')
    vid_paths = get_path('../inputs/full_game_1080p')
    total = len(vid_paths)
    current = 1
    failed = 0
    for vid_path in vid_paths:
        print(f'Progress: {current}/{total}\nFailed: {failed}', '\n', vid_path)
        success, vid_info = preprocess.video_preprocess(vid_path)
        if success:
            current += 1
        else:
            failed += 1
        print(success, vid_info)
    print('Finish resolving!')