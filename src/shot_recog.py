import numpy as np
import json
import cv2
from PIL import Image, ImageDraw, ImageFont

score_0_d_list = [2,2,2,2,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,2,
                  2,2,2,2,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,2,2,2,
                  2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

score_1_d_list = [1,1,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,
                  2,2,2,2,2,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]


def top_bottom(joint):
    if joint[0][16][1] > joint[1][16][1]:
        top = 1
        bottom = 0
    else:
        top = 0
        bottom = 1
    return top, bottom

# [top_back, top_mid, top_front, bot_front, bot_mid, bot_back]
def add_result(base, vid_path, shot_list, court_points):
    bounds = get_area_bound(court_points)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    FPS = cap.get(5)
    save_path = f"{base}{vid_path.split('/')[-1].split('.')[0]}_added.mp4"
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS,(frame_width, frame_height))
    count = 0
    i = 0
    imax = len(shot_list)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            for b in bounds:
                cv2.circle(frame, tuple((int(frame_width/2 - 2), int(b[0]))), 5, (255, 255, 0), 10)
                cv2.circle(frame, tuple((int(frame_width / 2 - 2), int(b[1]))), 5, (255, 255, 0), 10)
            bound = shot_list[i][2]
            if bound > count:
                text = shot_list[i][0]
                cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                draw = ImageDraw.Draw(pil_im)
                font = ImageFont.truetype("../font/msjh.ttc", 50, encoding="utf-8")
                draw.text((900, 50), text, (255, 255, 255), font=font)
                cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                # cv2.putText(frame, text, (700, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1,
                            # cv2.LINE_AA)
                out.write(cv2_text_im)
            count += 1
            if count > bound and i < imax-1:
                i += 1
        else:
            break
    return True

def get_pos_percentage(joint_list, bounds):
    top_front = 0
    top_mid = 0
    top_back = 0
    bot_front = 0
    bot_mid = 0
    bot_back = 0
    for i in range(len(joint_list)):
        top, bot = top_bottom(joint_list[i])
        t_coord = (joint_list[i][top][16][1] + joint_list[i][top][15][1]) / 2
        b_coord = (joint_list[i][bot][16][1] + joint_list[i][bot][15][1]) / 2
        t_pos = check_pos(t_coord, bounds, 'top')
        b_pos = check_pos(b_coord, bounds, 'bot')
        if t_pos == 'front':
            top_front += 1
        elif t_pos == 'mid':
            top_mid += 1
        elif t_pos == 'back':
            top_back += 1

        if b_pos == 'front':
            bot_front += 1
        elif b_pos == 'mid':
            bot_mid += 1
        elif b_pos == 'back':
            bot_back += 1
    all = len(joint_list)
    result = {
        'top': [top_front / all, top_mid / all, top_back / all],
        'bot': [bot_front / all, bot_mid / all, bot_back / all]
    }
    return result

def check_hit_frame(direction_list, joint_list, court_points):
    if direction_list == 0:
        direction_list = score_0_d_list
    elif direction_list == 1:
        direction_list = score_1_d_list
    bounds = get_area_bound(court_points)
    pos_percentage = get_pos_percentage(joint_list, bounds)
    first_coord = None
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
            elif d == 2:
                top, bot = top_bottom(joint_list[i])
                first_coord = (joint_list[i][bot][16][1] + joint_list[i][bot][15][1]) / 2
                first_i = i
                got_first = True
                last_d = 2
            continue
        if d != last_d and last_d == 1:
            if d == 0:
                d = 2
                change = True
            else:
                change = False
            top, bot = top_bottom(joint_list[i])
            second_coord = (joint_list[i][bot][16][1] + joint_list[i][bot][15][1]) / 2
            second_i = i
            shot = shot_recog(first_coord, second_coord, d, bounds)
            shot_list.append((shot, first_i, second_i))
            first_i = second_i
            last_d = d
            if change:
                last_d = 0
            first_coord = second_coord
        if d != last_d and last_d == 2:
            if d == 0:
                d = 1
                change = True
            else:
                change = False
            top, bot = top_bottom(joint_list[i])
            second_coord = (joint_list[i][top][16][1] + joint_list[i][top][15][1]) / 2
            second_i = i
            shot = shot_recog(first_coord, second_coord, d, court_points)
            shot_list.append((shot, first_i, second_i))
            first_i = second_i
            last_d = d
            if change:
                last_d = 0
            first_coord = second_coord
    return shot_list, pos_percentage


# [[554, 513], [1366, 495], [462, 708], [1454, 704], [349, 1000], [1568, 999]]
def get_area_bound(court_points):
    top = round((court_points[0][1] + court_points[1][1]) / 2)
    mid = round((court_points[2][1] + court_points[3][1]) / 2)
    bot = round((court_points[4][1] + court_points[5][1]) / 2)
    top_sliced_area = (mid - top)/10
    bot_sliced_area = (bot - mid)/10
    top_back = (top, top + 4*top_sliced_area)
    top_mid = (top + 4*top_sliced_area, top + 6*top_sliced_area)
    top_front = (top + 6*top_sliced_area, mid)

    bot_back = (bot - 4*bot_sliced_area, bot)
    bot_mid = (bot - 6*bot_sliced_area, bot - 4*bot_sliced_area)
    bot_front = (mid, bot - 6*bot_sliced_area)

    bounds = [top_back, top_mid, top_front, bot_front, bot_mid, bot_back]
    return bounds

def check_pos(coord, bounds, pos):
    if pos == 'top':
        if coord < bounds[0][1]:
            return 'back'
        if coord > bounds[1][0] and coord < bounds[1][1]:
            return 'mid'
        if coord > bounds[2][0]:
            return 'front'
    if pos == 'bot':
        if coord < bounds[3][1]:
            return 'front'
        if coord > bounds[4][0] and coord < bounds[4][1]:
            return 'mid'
        if coord > bounds[5][0]:
            return 'back'
    return None


def shot_recog(first_coord, second_coord, d, bounds):
    bounds = bounds
    if d == 1:      # last d == 2
        pos_bot = check_pos(first_coord, bounds, 'bot')
        pos_top = check_pos(second_coord, bounds, 'top')
        serve = 'bot'
    if d == 2:      # last d == 1
        pos_top = check_pos(first_coord, bounds, 'top')
        pos_bot = check_pos(second_coord, bounds, 'bot')
        serve = 'top'
    print(pos_top, pos_bot, serve)
    shot = check_shot(pos_top, pos_bot, serve)
    return shot


def check_shot(pos_top, pos_bot, serve):
    if serve == 'top':
        if pos_top == 'front' and pos_bot == 'front':
            return '↓ 小球'
        if pos_top == 'front' and pos_bot == 'mid':
            return '↓ 平球'
        if pos_top == 'front' and pos_bot == 'back':
            return '↓ 挑球'
        if pos_top == 'mid' and pos_bot == 'front':
            return '↓ 小球'
        if pos_top == 'mid' and pos_bot == 'mid':
            return '↓ 平球'
        if pos_top == 'mid' and pos_bot == 'back':
            return '↓ 挑球'
        if pos_top == 'back' and pos_bot == 'front':
            return '↓ 切球'
        if pos_top == 'back' and pos_bot == 'mid':
            return '↓ 殺球'
        if pos_top == 'back' and pos_bot == 'back':
            return '↓ 長球'
    if serve == 'bot':
        if pos_top == 'front' and pos_bot == 'front':
            return '↑ 小球'
        if pos_top == 'front' and pos_bot == 'mid':
            return '↑ 小球'
        if pos_top == 'front' and pos_bot == 'back':
            return '↑ 切球'
        if pos_top == 'mid' and pos_bot == 'front':
            return '↑ 平球'
        if pos_top == 'mid' and pos_bot == 'mid':
            return '↑ 平球'
        if pos_top == 'mid' and pos_bot == 'back':
            return '↑ 殺球'
        if pos_top == 'back' and pos_bot == 'front':
            return '↑ 挑球'
        if pos_top == 'back' and pos_bot == 'mid':
            return '↑ 挑球'
        if pos_top == 'back' and pos_bot == 'back':
            return '↑ 長球'


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

# input, s_joint_list = get_data('E:/test_videos/outputs/p_test/score_15/score_15.json')
# print(len(input), len(frame_num), len(s_joint_list))
# print(s_joint_list[10].shape)
# with open('E:/test_videos/outputs/p_test/score_15/score_15.json', 'r') as mp_json:
#     frame_dict = json.load(mp_json)
#
# labels = []
# for i in range(len(frame_dict['frames'])):
#     labels.append(frame_dict['frames'][i]['label'])
# print(labels, len(labels))