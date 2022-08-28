import os
import numpy as np

def cal_dis(p1, p2):
    return np.sqrt(np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]))

def check_dir(path):
    isExit = os.path.exists(path)
    if not isExit:
        os.mkdir(path)


def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
            pass
    return paths


def parse_time(FPS, frame_count):
    start_sec = int(frame_count / FPS)

    ssec = start_sec % 60
    smin = start_sec // 60
    if smin >= 60:
        smin = smin % 60
    shr = start_sec // 3600

    if ssec < 10:
        start_sec = '0' + str(start_sec)
    if smin < 10:
        smin = '0' + str(smin)
    if shr < 10:
        shr = '0' + str(shr)

    return f'{shr}-{smin}-{ssec}'


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


def top_bottom(joint):
    a = joint[0][15][1] + joint[0][16][1]
    b = joint[1][15][1] + joint[1][16][1]
    if a > b:
        top = 1
        bottom = 0
    else:
        top = 0
        bottom = 1
    return top, bottom