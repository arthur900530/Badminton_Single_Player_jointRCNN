import os
import json
from pipeline import video_resolver


class resolve_interface:
    def __init__(self, vid_id, base='E:/test_videos'):
        vid_path = f'{base}/inputs/{vid_id}'
        isExit = os.path.exists(f'{base}/outputs/{vid_id}')

        # output base is where "outputs" dir is
        vpr = video_resolver(vid_path, output_base=base, isExit=isExit)
        if not isExit:
            _ = vpr.resolve()
        # boolean, boolean, [blue win key, blue loss key, red win key, red loss key]
        blue_highlight, red_highlight, keys = vpr.get_highlights_info()
        total_info = vpr.get_total_info()
        scores_dict = vpr.get_respective_score_info()

        return_info_dict = {
            'highlights info': [blue_highlight, red_highlight, keys],
            'total info': total_info,
            'respective scores': scores_dict
        }
        joint_save_path = f'{base}/outputs/{vid_id}/return_info.json'
        with open(joint_save_path, 'w') as f:
            json.dump(return_info_dict, f, indent=2)

