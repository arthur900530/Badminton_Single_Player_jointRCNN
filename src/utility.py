import os

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

    return f'{shr}:{smin}:{ssec}'