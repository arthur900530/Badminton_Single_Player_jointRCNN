import numpy as np
import json


def top_bottom(joint):
    if joint[0][16][1] > joint[1][16][1]:
        top = 1
        bottom = 0
    else:
        top = 0
        bottom = 1
    return top, bottom

def check_hit_frame(direction_list, joint_list, court_points):
    shot_list = []
    got_first = False
    last_d = 0
    for i in range(len(direction_list)):
        d = direction_list[i]
        if not got_first:
            if d == 1:
                top, bot = top_bottom(joint_list[i])
                first_coord = (joint_list[i][top][16][1] + joint_list[i][top][15][1]) / 2
                first_i = i
                got_first = True
                last_d = 1
                continue
            elif d == 2:
                top, bot = top_bottom(joint_list[i])
                first_coord = (joint_list[i][bot][16][1] + joint_list[i][bot][15][1]) / 2
                first_i = i
                got_first = True
                last_d = 2
                continue
        if d != last_d and d != 0:
            second_coord = (joint_list[i][bot][16][1] + joint_list[i][bot][15][1]) / 2
            second_i = i
            shot = shot_recog(first_coord, second_coord, d, court_points)
            shot_list.append((shot, first_i, second_i))
            last_d = d
            first_coord = second_coord


def shot_recog(first_coord, second_coord, d, court_points):
    pass

def get_time(start_frame):
    pass


def get_data(path):
    input = []
    joint_list = []
    frame_num = []
    with open(path, 'r') as mp_json:
        frame_dict = json.load(mp_json)

    for i in range(len(frame_dict['frames'])):
        temp_x = []
        if i == 0:
            temp_f = []
            former = np.array(frame_dict['frames'][i]['joint'])
            for p in range(2):
                temp_f.append(former[p][5:])
            temp_f = np.array(temp_f)
            frame_num.append(frame_dict['frames'][i]['frame'])
            joint_list.append(frame_dict['frames'][i]['joint'])
            continue
        frame_num.append(frame_dict['frames'][i]['frame'])
        joint_list.append(frame_dict['frames'][i]['joint'])
        joint = frame_dict['frames'][i]['joint']
        for p in range(2):
            temp_x.append(joint[p][5:])  # ignore head part
        temp_x = np.array(temp_x)
        dif_x = temp_f - temp_x
        temp_f = temp_x
        input.append(dif_x)

    input = np.array(input)
    frame_num = np.array(frame_num)
    joint_list = np.array(joint_list)
    return input, frame_num, joint_list

input, frame_num, s_joint_list = get_data('E:/test_videos/outputs/p_test/score_15/score_15.json')
print(len(input), len(frame_num), len(s_joint_list))
print(s_joint_list[10].shape)
with open('E:/test_videos/outputs/p_test/score_15/score_15.json', 'r') as mp_json:
    frame_dict = json.load(mp_json)

labels = []
for i in range(len(frame_dict['frames'])):
    labels.append(frame_dict['frames'][i]['label'])
print(labels, len(labels))