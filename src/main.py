import csv
import preprocess
from scene_utils import scene_classifier
import video_download as vd
import frame_process as fp
import video_process as vp

# vid_infos = vd.get_vid_paths('video.csv')
# finish_download = vd.download_vid(vid_infos)
finish_download = True

# 存成圖片版
# if finish_download:
#     print('Start resolving...')
#     vid_paths = preprocess.get_path('../inputs/full_game_1080p')
#     total = len(vid_paths)
#     current = 1
#     proccessed = 0
#     for vid_path in vid_paths:
#         print(f'Progress: {current}/{total}\nProccessed: {proccessed}\nCurrent path:  {vid_path}')
#         success, vid_info = preprocess.video_preprocess(vid_path)
#         if success:
#             current += 1
#             proccessed += 1
#         else:
#             current += 1
#         print('Processed: ', proccessed, '\n', 'Video info: ', vid_info)
#     print('Finish resolving!')
#     print('='*30)

# court_p_A = [[590, 434, 1], [1310, 434, 1], [476, 624, 1],[1427, 623, 1],[256, 1000, 1],[1660, 1002, 1]]
#
# vid_paths = preprocess.get_path('../test/T_data')
#
# for vid_path in vid_paths:
#     scores = preprocess.get_path(vid_path)
#     for score in scores:
#         success = fp.score_process(score, court_p_A)
#
# print('Success: ', success)

if finish_download:
    print(f"Start resolving...\n{'='*30}")
    vid_paths = preprocess.get_path('../inputs/full_game_1080p') # the dir where the videos are
    total_vids = len(vid_paths)
    processed = []
    with open('csv_records/processed.csv') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if len(row) != 0:
                processed.append(row[0])
            else:
                continue

    for i, path in enumerate(vid_paths):
        print(f'Progress: {i+1} / {total_vids}')
        vid_name = path.split('/')[-1].split('.')[0]
        if vid_name not in processed:
            vpr = vp.video_processor(path, output_base='..')    # output base is where "outputs" dir is
            vpr.process()
            with open('csv_records/processed.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([vid_name])
        else:
            print(f"{vid_name} already processed!\n{'='*30}")
    print('Finish resolving!')


