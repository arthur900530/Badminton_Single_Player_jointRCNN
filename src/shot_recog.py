import numpy as np
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
from utility import zone, cal_move_direction


def top_bottom(joint):
    a = joint[0][-1][1] + joint[0][-2][1]
    b = joint[1][-1][1] + joint[1][-2][1]
    if a > b:
        top = 1
        bottom = 0
    else:
        top = 0
        bottom = 1
    return top, bottom


def get_area(court_points, bounds):
    l_a = (court_points[0][1] - court_points[4][1]) / (
            court_points[0][0] - court_points[4][0])
    l_b = court_points[0][1] - l_a * court_points[0][0]
    r_a = (court_points[1][1] - court_points[5][1]) / (
            court_points[1][0] - court_points[5][0])
    r_b = court_points[1][1] - r_a * court_points[1][0]
    all = []

    tb0x = (min(court_points[0][1], court_points[1][1]) - l_b) / l_a
    tbox = (min(court_points[0][1], court_points[1][1]) - r_b) / r_a
    tb1x = (bounds[0][1] - l_b) / l_a
    tb2x = (bounds[0][1] - r_b) / r_a
    top_back_area = np.array([[tb0x, min(court_points[0][1], court_points[1][1])], [tbox, min(court_points[0][1], court_points[1][1])], [tb2x, bounds[0][1]], [tb1x, bounds[0][1]]], np.int32)
    top_b = top_back_area.reshape((-1, 1, 2))

    tm1x = (bounds[1][1] - l_b) / l_a
    tm2x = (bounds[1][1] - r_b) / r_a
    top_mid_area = np.array([[tb1x, bounds[0][1]], [tb2x, bounds[0][1]], [tm2x, bounds[1][1]], [tm1x, bounds[1][1]]], np.int32)
    top_m = top_mid_area.reshape((-1, 1, 2))

    tf1x = (bounds[2][1] - l_b) / l_a
    tf2x = (bounds[2][1] - r_b) / r_a
    top_front_area = np.array(
        [[tm1x, bounds[1][1]], [tm2x, bounds[1][1]], [tf2x, bounds[2][1]], [tf1x, bounds[2][1]]], np.int32)
    top_f = top_front_area.reshape((-1, 1, 2))

    bf1x = (bounds[3][1] - l_b) / l_a
    bf2x = (bounds[3][1] - r_b) / r_a
    bot_front_area = np.array(
        [[tf1x, bounds[2][1]+2], [tf2x, bounds[2][1]+2], [bf2x, bounds[3][1]], [bf1x, bounds[3][1]]], np.int32)
    bot_f = bot_front_area.reshape((-1, 1, 2))

    bm1x = (bounds[4][1] - l_b) / l_a
    bm2x = (bounds[4][1] - r_b) / r_a
    bot_mid_area = np.array(
        [[bf1x, bounds[3][1]], [bf2x, bounds[3][1]], [bm2x, bounds[4][1]], [bm1x, bounds[4][1]]], np.int32)
    bot_m = bot_mid_area.reshape((-1, 1, 2))

    bb1x = (bounds[5][1] - l_b) / l_a
    bb2x = (bounds[5][1] - r_b) / r_a
    bot_back_area = np.array(
        [[bm1x, bounds[4][1]], [bm2x, bounds[4][1]], [bb2x, bounds[5][1]], [bb1x, bounds[5][1]]], np.int32)
    bot_b = bot_back_area.reshape((-1, 1, 2))
    # (39, 171, 242)
    all.append((top_b, (255, 255, 0)))
    all.append((top_m, (255, 255, 0)))
    all.append((top_f, (255, 255, 0)))
    all.append((bot_f, (0, 255, 255)))
    all.append((bot_m, (0, 255, 255)))
    all.append((bot_b, (0, 255, 255)))
    return all


# [top_back, top_mid, top_front, bot_front, bot_mid, bot_back]
def add_result(base, vid_path, shot_list, move_dir_list, court_points):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    FPS = cap.get(5)
    save_path = f"{base}{vid_path.split('/')[-1].split('.')[0]}_added.mp4"
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS,(frame_width, frame_height))
    count = 1
    i = 0
    imax = len(shot_list)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            bound = shot_list[i][2]
            if bound >= count:
                text = shot_list[i][0] + ' ' + move_dir_list[i][0]
                cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                draw = ImageDraw.Draw(pil_im)
                font = ImageFont.truetype("../font/msjh.ttc", 50, encoding="utf-8")
                draw.text((900, 50), text, (255, 255, 255), font=font)
                cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                out.write(cv2_text_im)
                count += 1
            elif count > bound and i < imax - 1:
                i += 1
                text = shot_list[i][0]
                cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                draw = ImageDraw.Draw(pil_im)
                font = ImageFont.truetype("../font/msjh.ttc", 50, encoding="utf-8")
                draw.text((900, 50), text, (255, 255, 255), font=font)
                cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                out.write(cv2_text_im)
                count += 1
            else:
                out.write(frame)
                count += 1

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
        t_coord = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
        b_coord = (joint_list[i][bot][-1][1] + joint_list[i][bot][-2][1]) / 2
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


def check_hit_frame(direction_list, joint_list, court_points, multi_points):
    joint_list = joint_list.squeeze(0).cpu().numpy()  # seq len, 2, 12, 2
    multi_points = np.array(multi_points)
    multi_points = np.reshape(multi_points, (7, 5, 2))
    bounds = get_area_bound(court_points)
    shot_list = []
    move_dir_list = []
    got_first = False
    last_d = 0
    for i in range(len(direction_list)):
        d = direction_list[i]
        if not got_first:
            if d == 1:
                top, bot = top_bottom(joint_list[i])
                first_y = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
                first_x_bot = (joint_list[i][bot][-1][0] + joint_list[i][bot][-2][0]) / 2
                first_y_bot = (joint_list[i][bot][-1][1] + joint_list[i][bot][-2][1]) / 2
                first_coord_bot = np.array([first_x_bot, first_y_bot])

                first_zone = zone(first_coord_bot, multi_points)
                print(first_coord_bot, first_zone)
                first_i = i
                got_first = True
                last_d = 1
            elif d == 2:
                top, bot = top_bottom(joint_list[i])
                first_y = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
                first_x_top = (joint_list[i][top][-1][0] + joint_list[i][top][-2][0]) / 2
                first_y_top = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
                first_coord_top = np.array([first_x_top, first_y_top])
                first_zone = zone(first_coord_top, multi_points)
                print(first_coord_top, first_zone)

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
            second_y = (joint_list[i][bot][-1][1] + joint_list[i][bot][-2][1]) / 2
            second_x_top = (joint_list[i][top][-1][0] + joint_list[i][top][-2][0]) / 2
            second_y_top = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
            second_x_bot = (joint_list[i][bot][-1][0] + joint_list[i][bot][-2][0]) / 2
            second_y_bot = (joint_list[i][bot][-1][1] + joint_list[i][bot][-2][1]) / 2
            second_coord_fm = np.array([second_x_bot, second_y_bot])
            second_zone = zone(second_coord_fm, multi_points)
            print(second_coord_fm, second_zone)
            second_i = i

            shot, top_serve = shot_recog(first_y, second_y, d, bounds)
            move_dir = cal_move_direction(first_zone[0], second_zone[0])
            move_dir_list.append((move_dir, False))  # True for top
            first_coord_fm = np.array([second_x_top, second_y_top])
            first_zone = zone(first_coord_fm, multi_points)

            shot_list.append((shot, first_i, second_i, top_serve))

            first_i = second_i
            last_d = d
            if change:
                last_d = 0
            first_y = second_y
        if d != last_d and last_d == 2:
            if d == 0:
                d = 1
                change = True
            else:
                change = False
            top, bot = top_bottom(joint_list[i])
            second_y = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2

            second_x_top = (joint_list[i][top][-1][0] + joint_list[i][top][-2][0]) / 2
            second_y_top = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
            second_x_bot = (joint_list[i][bot][-1][0] + joint_list[i][bot][-2][0]) / 2
            second_y_bot = (joint_list[i][bot][-1][1] + joint_list[i][bot][-2][1]) / 2

            second_coord_fm = np.array([second_x_top, second_y_top])
            second_zone = zone(second_coord_fm, multi_points)
            print(second_coord_fm, second_zone)
            second_i = i

            shot, top_serve = shot_recog(first_y, second_y, d, bounds)
            move_dir = cal_move_direction(first_zone[0], second_zone[0])
            move_dir_list.append((move_dir, True))
            first_coord_fm = np.array([second_x_bot, second_y_bot])
            first_zone = zone(first_coord_fm, multi_points)

            shot_list.append((shot, first_i, second_i, top_serve))
            first_i = second_i
            last_d = d
            if change:
                last_d = 0
            first_y = second_y
    return shot_list, move_dir_list


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
    shot, top_serve = check_shot(pos_top, pos_bot, serve)
    return shot, top_serve


def check_shot(pos_top, pos_bot, serve):
    if serve == 'top':
        if pos_top == 'front' and pos_bot == 'front':
            return '↓ 小球', True                     #True stands for top player's
        if pos_top == 'front' and pos_bot == 'mid':
            return '↓ 撲球', True
        if pos_top == 'front' and pos_bot == 'back':
            return '↓ 挑球', True
        if pos_top == 'mid' and pos_bot == 'front':
            return '↓ 小球', True
        if pos_top == 'mid' and pos_bot == 'mid':
            return '↓ 平球', True
        if pos_top == 'mid' and pos_bot == 'back':
            return '↓ 挑球', True
        if pos_top == 'back' and pos_bot == 'front':
            return '↓ 切球', True
        if pos_top == 'back' and pos_bot == 'mid':
            return '↓ 殺球', True
        if pos_top == 'back' and pos_bot == 'back':
            return '↓ 長球', True
    if serve == 'bot':
        if pos_top == 'front' and pos_bot == 'front':
            return '↑ 小球', False
        if pos_top == 'front' and pos_bot == 'mid':
            return '↑ 小球', False
        if pos_top == 'front' and pos_bot == 'back':
            return '↑ 切球', False
        if pos_top == 'mid' and pos_bot == 'front':
            return '↑ 撲球', False
        if pos_top == 'mid' and pos_bot == 'mid':
            return '↑ 平球', False
        if pos_top == 'mid' and pos_bot == 'back':
            return '↑ 殺球', False
        if pos_top == 'back' and pos_bot == 'front':
            return '↑ 挑球', False
        if pos_top == 'back' and pos_bot == 'mid':
            return '↑ 挑球', False
        if pos_top == 'back' and pos_bot == 'back':
            return '↑ 長球', False


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