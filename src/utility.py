import os
import numpy as np


def correction(court_kp):
    ty = np.round((court_kp[0][1] + court_kp[1][1]) / 2)
    my = (court_kp[2][1] + court_kp[3][1]) / 2
    by = np.round((court_kp[4][1] + court_kp[5][1]) / 2)
    court_kp[0][1] = ty
    court_kp[1][1] = ty
    court_kp[2][1] = my
    court_kp[3][1] = my
    court_kp[4][1] = by
    court_kp[5][1] = by
    return court_kp


def extension(court_kp):
    tlspace = np.array(
        [np.round((court_kp[0][0] - court_kp[2][0]) / 3), np.round((court_kp[2][1] - court_kp[0][1]) / 3)], dtype=int)
    trspace = np.array(
        [np.round((court_kp[3][0] - court_kp[1][0]) / 3), np.round((court_kp[3][1] - court_kp[1][1]) / 3)], dtype=int)
    blspace = np.array(
        [np.round((court_kp[2][0] - court_kp[4][0]) / 3), np.round((court_kp[4][1] - court_kp[2][1]) / 3)], dtype=int)
    brspace = np.array(
        [np.round((court_kp[5][0] - court_kp[3][0]) / 3), np.round((court_kp[5][1] - court_kp[3][1]) / 3)], dtype=int)

    p2 = np.array([court_kp[0][0] - tlspace[0], court_kp[0][1] + tlspace[1]])
    p3 = np.array([court_kp[1][0] + trspace[0], court_kp[1][1] + trspace[1]])
    p4 = np.array([p2[0] - tlspace[0], p2[1] + tlspace[1]])
    p5 = np.array([p3[0] + trspace[0], p3[1] + trspace[1]])

    p8 = np.array([court_kp[2][0] - blspace[0], court_kp[2][1] + blspace[1]])
    p9 = np.array([court_kp[3][0] + brspace[0], court_kp[3][1] + brspace[1]])
    p10 = np.array([p8[0] - blspace[0], p8[1] + blspace[1]])
    p11 = np.array([p9[0] + brspace[0], p9[1] + brspace[1]])

    kp = np.array([court_kp[0], court_kp[1],
                   p2, p3, p4, p5,
                   court_kp[2], court_kp[3],
                   p8, p9, p10, p11,
                   court_kp[4], court_kp[5]], dtype=int)

    ukp = []

    for i in range(0, 13, 2):
        sub2 = np.round((kp[i] + kp[i + 1]) / 2)
        sub1 = np.round((kp[i] + sub2) / 2)
        sub3 = np.round((kp[i + 1] + sub2) / 2)
        ukp.append(kp[i])
        ukp.append(sub1)
        ukp.append(sub2)
        ukp.append(sub3)
        ukp.append(kp[i + 1])
    ukp = np.array(ukp, dtype=int)
    return ukp


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