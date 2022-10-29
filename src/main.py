import os
from utility import get_path
from pipeline import video_resolver
from transformer_utils import coordinateEmbedding, PositionalEncoding, Optimus_Prime
from scene_utils import scene_classifier


def main():
    paths = get_path('E:/test_videos/inputs')
    vid_paths = []
    for path in paths:
        if path.split('/')[-1].split('.')[-1] == 'mp4':
            vid_paths.append(path)
    for vid_path in vid_paths:
        vid_name = vid_path.split('/')[-1].split('.')[0]
        isExit = os.path.exists(f'E:/test_videos/outputs/{vid_name}')
        vpr = video_resolver(vid_path, output_base='E:/test_videos', isExit=isExit)  # output base is where "outputs" dir is
        if not isExit:
            _ = vpr.resolve()

        blue_win_dict, blue_loss_dict, red_win_dict, red_loss_dict = vpr.get_player_strategy()
        print(blue_win_dict, blue_loss_dict, red_win_dict, red_loss_dict)
        # total_info = vpr.get_total_info()
        # scores_dict = vpr.get_respective_score_info()
        # print(total_info)
        # print(scores_dict['g1'])
        # print(len(scores_dict['g1']), len(scores_dict['g2']))

if __name__ == '__main__':
    main()

