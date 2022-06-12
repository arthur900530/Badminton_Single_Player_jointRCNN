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
        if d != last_d and last_d == 1:
            second_coord = (joint_list[i][bot][16][1] + joint_list[i][bot][15][1]) / 2
            second_i = i
            shot = shot_recog(first_coord, second_coord, d, court_points)
            shot_list.append((shot, first_i, second_i))
            last_d = d
            first_coord = second_coord
        if d != last_d and last_d == 2:
            second_coord = (joint_list[i][top][16][1] + joint_list[i][top][15][1]) / 2
            second_i = i
            shot = shot_recog(first_coord, second_coord, d, court_points)
            shot_list.append((shot, first_i, second_i))
            last_d = d
            first_coord = second_coord

    return shot_list


# [[554, 513], [1366, 495], [462, 708], [1454, 704], [349, 1000], [1568, 999]]
def get_area_bound(court_points):
    top = round((court_points[0][1] + court_points[1][1]) / 2)
    mid = round((court_points[2][1] + court_points[3][1]) / 2)
    bot = round((court_points[4][1] + court_points[5][1]) / 2)
    top_sliced_area = (mid - top)/5
    bot_sliced_area = (bot - mid)/5
    top_back = (top, top + top_sliced_area)
    top_mid = (top + top_sliced_area, top + 3*top_sliced_area)
    top_front = (top + 3*top_sliced_area, mid)

    bot_back = (bot-bot_sliced_area, bot)
    bot_mid = (bot - 3*bot_sliced_area, bot-bot_sliced_area)
    bot_front = (mid, bot - 3*bot_sliced_area)

    bounds = [top_back, top_mid, top_front, bot_front, bot_mid, bot_back]
    return bounds

def check_pos(coord, bounds, pos):
    if pos == 'top':
        if coord < bounds[0][1]:
            return 'back'
        if coord > bounds[1][0] and coord < bounds[1][1]:
            return 'mid'
        if coord > bounds[2][0] and coord < bounds[2][1]:
            return 'front'
    if pos == 'bot':
        if coord > bounds[3][0] and coord < bounds[3][1]:
            return 'front'
        if coord > bounds[4][0] and coord < bounds[4][1]:
            return 'mid'
        if coord > bounds[5][0]:
            return 'back'
    return None


def shot_recog(first_coord, second_coord, d, court_points):
    bounds = get_area_bound(court_points)
    if d == 1:      # last d == 2
        pos_bot = check_pos(first_coord, bounds, 'bot')
        pos_top = check_pos(second_coord, bounds, 'top')
        serve = 'bot'
    if d == 2:      # last d == 1
        pos_top = check_pos(first_coord, bounds, 'top')
        pos_bot = check_pos(second_coord, bounds, 'bot')
        serve = 'top'
    shot = check_shot(pos_top, pos_bot, serve)
    return shot


def check_shot(pos_top, pos_bot, serve):
    if serve == 'top':
        if pos_top == 'front' and pos_bot == 'front':
            return '上至下小球'
        if pos_top == 'front' and pos_bot == 'mid':
            return '上至下平球'
        if pos_top == 'front' and pos_bot == 'back':
            return '上至下挑球'
        if pos_top == 'mid' and pos_bot == 'front':
            return '上至下小球'
        if pos_top == 'mid' and pos_bot == 'mid':
            return '上至下平球'
        if pos_top == 'mid' and pos_bot == 'back':
            return '上至下挑球'
        if pos_top == 'back' and pos_bot == 'front':
            return '上至下切球'
        if pos_top == 'back' and pos_bot == 'mid':
            return '上至下殺球'
        if pos_top == 'back' and pos_bot == 'back':
            return '上至下長球'
    if serve == 'bot':
        if pos_top == 'front' and pos_bot == 'front':
            return '下至上小球'
        if pos_top == 'front' and pos_bot == 'mid':
            return '下至上小球'
        if pos_top == 'front' and pos_bot == 'back':
            return '下至上切球'
        if pos_top == 'mid' and pos_bot == 'front':
            return '下至上平球'
        if pos_top == 'mid' and pos_bot == 'mid':
            return '下至上平球'
        if pos_top == 'mid' and pos_bot == 'back':
            return '下至上殺球'
        if pos_top == 'back' and pos_bot == 'front':
            return '下至上挑球'
        if pos_top == 'back' and pos_bot == 'mid':
            return '下至上挑球'
        if pos_top == 'back' and pos_bot == 'back':
            return '下至上長球'


def get_data(path):
    input = []
    joint_list = []

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
            joint_list.append(frame_dict['frames'][i]['joint'])
            continue
        joint_list.append(frame_dict['frames'][i]['joint'])
        joint = frame_dict['frames'][i]['joint']
        for p in range(2):
            temp_x.append(joint[p][5:])  # ignore head part
        temp_x = np.array(temp_x)
        dif_x = temp_f - temp_x
        temp_f = temp_x
        input.append(dif_x)

    input = np.array(input)
    joint_list = np.array(joint_list)
    return input, joint_list

input, frame_num, s_joint_list = get_data('E:/test_videos/outputs/p_test/score_15/score_15.json')
print(len(input), len(frame_num), len(s_joint_list))
print(s_joint_list[10].shape)
with open('E:/test_videos/outputs/p_test/score_15/score_15.json', 'r') as mp_json:
    frame_dict = json.load(mp_json)

labels = []
for i in range(len(frame_dict['frames'])):
    labels.append(frame_dict['frames'][i]['label'])
print(labels, len(labels))