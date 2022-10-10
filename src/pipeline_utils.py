def check_type(last_type, wait_list):
    sum = 0
    if last_type == 1:
        for pair in wait_list:
            sum += pair[0]
        if sum <= 3:
            return True
        else:
            return False
    else:
        for pair in wait_list:
            sum += pair[0]
        if sum >= 3:
            return True
        else:
            return False


# check if player is in court
def in_court(court_info, court_points, joint):
    l_a = court_info[0]
    l_b = court_info[1]
    r_a = court_info[2]
    r_b = court_info[3]
    ankle_x = (joint[15][0] + joint[16][0]) / 2
    ankle_y = (joint[15][1] + joint[16][1]) / 2
    top = ankle_y > court_points[0][1]
    bottom = ankle_y < court_points[5][1]
    lmp_x = (ankle_y - l_b) / l_a
    rmp_x = (ankle_y - r_b) / r_a
    left = ankle_x > lmp_x
    right = ankle_x < rmp_x

    if left and right and top and bottom:
        return True
    else:
        return False


# get the index of the in court players
def score_rank(court_info, court_points, joints):
    indexes = []
    for i in range(len(joints)):
        if in_court(court_info, court_points, joints[i]):
            indexes.append(i)
    if len(indexes) < 2:
        return False
    else:
        return indexes


# check if up court and bot court got player
def check_pos(court_mp, indexes, boxes):
    for i in range(len(indexes)):
        combination = 1
        if boxes[indexes[0]][1] < court_mp < boxes[indexes[combination]][3]:
            return True, [0, combination]
        elif boxes[indexes[0]][3] > court_mp > boxes[indexes[combination]][1]:
            return True, [0, combination]
        else:
            combination += 1
    return False, [0, 0]