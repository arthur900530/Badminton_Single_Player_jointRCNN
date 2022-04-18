import cv2
import matplotlib

edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),
         (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
         (12, 14), (14, 16), (5, 6)]


def checkPos(indexes, boxes, court_info):
    court_mp = court_info[4]
    for i in range(len(indexes)):
        combination = 1
        if boxes[indexes[0]][1] < court_mp and boxes[indexes[combination]][3] > court_mp:
            return True, [0, combination]
        elif boxes[indexes[0]][3] > court_mp and boxes[indexes[combination]][1] < court_mp:
            return True, [0, combination]
        else:
            combination += 1
    return False, [0, 0]


def in_court(joint, court_box, court_point):
    l_a = court_point[0]
    l_b = court_point[1]
    r_a = court_point[2]
    r_b = court_point[3]
    mp_y = court_point[4]
    ankle_x = (joint[15][0] + joint[16][0])/2
    ankle_y = (joint[15][1] + joint[16][1])/2
    top = ankle_y > court_box[1] - 80
    bottom = ankle_y < court_box[3]
    lmp_x = (ankle_y - l_b) / l_a
    rmp_x = (ankle_y - r_b) / r_a
    left = ankle_x > lmp_x
    right = ankle_x < rmp_x

    if left and right and top and bottom:
        return True
    else:
        return False


def score_rank(joints, court_box, court_info):
    indexes = []
    for i in range(len(joints)):
        if in_court(joints[i], court_box, court_info):
            indexes.append(i)
    if len(indexes) < 2:
        return False
    else:
        return indexes


def draw_keypoints(outputs, image, frame_count, court_box, court_point):
    playerJoints = []
    b = outputs[0]['boxes'].cpu().detach().numpy()
    # s = outputs[0]['scores'].cpu().detach().numpy()
    j = outputs[0]['keypoints'].cpu().detach().numpy()
    # if len(b) > 5:
    #     b = b[:5]
    #     # s = s[:5]
    #     j = j[:5]
    l_a = (court_point[0][1] - court_point[4][1]) / (court_point[0][0] - court_point[4][0])
    l_b = court_point[0][1] - l_a * court_point[0][0]
    r_a = (court_point[1][1] - court_point[5][1]) / (court_point[1][0] - court_point[5][0])
    r_b = court_point[1][1] - r_a * court_point[1][0]
    mp_y = (court_point[2][1] + court_point[3][1]) / 2
    court_info = [l_a, l_b, r_a, r_b, mp_y]
    topScores = score_rank(j, court_box, court_info)
    if topScores == False:
        cv2.putText(image, str(frame_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        return image, True
    fit, combination = checkPos(topScores, b, court_info)
    if fit:
        for c in combination:
            i = topScores[c]
            keypoints = j[i]
            keypoints = keypoints[:, :].reshape(-1, 3)
            playerJoints.append(j[i].tolist())
            for p in range(keypoints.shape[0]):
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(image, str(i), (int(keypoints[15, 0]), int(keypoints[15, 1])), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1,
                            cv2.LINE_AA)
            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                rgb = rgb * 255
                # join the keypoint pairs to draw the skeletal structure
                cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                         (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                         tuple(rgb), 2, lineType=cv2.LINE_AA)
        cv2.putText(image, str(frame_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        return image, playerJoints
    else:
        cv2.putText(image, str(frame_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        return image, True
