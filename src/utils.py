import cv2
import matplotlib
import numpy as np

edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),
         (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
         (12, 14), (14, 16), (5, 6)]


def checkPos(indexes, boxes):
    # 1080p: 423
    # 720p: 325
    for i in range(len(indexes)):
        combination = 1
        if boxes[indexes[0]][1] < 423 and boxes[indexes[combination]][3] > 423:
            return True, [0, combination]
        elif boxes[indexes[0]][3] > 423 and boxes[indexes[combination]][1] < 423:
            return True, [0, combination]
        else:
            combination += 1
    return False, [0, 0]


def scoreRank(joints, scores, topBound):
    indexes = []
    for i in range(len(scores)):
        if (joints[i][15][1] + joints[i][16][1])/2 > topBound:
            indexes.append(i)
    if len(indexes) < 2:
        return False
    else:
        return indexes


def findTopAnkle(joints):
    min = 10000
    for i in range(len(joints)):
        if (joints[i][15][1] + joints[i][16][1])/2 < min:
            min = (joints[i][15][1] + joints[i][16][1])/2
    return min


def draw_keypoints(outputs, image, width, frame_count):
    playerJoints = []
    # 1080p: int((width/2)-675),180
    # 720p: int((width/2)-400),120
    gapx = int((width / 2) - 675)
    gapy = int(180)
    b = outputs[0]['boxes'].cpu().detach().numpy()
    s = outputs[0]['scores'].cpu().detach().numpy()
    j = outputs[0]['keypoints'].cpu().detach().numpy()
    if len(s) > 5:
        b = b[:5]
        s = s[:5]
        j = j[:5]
    if len(s) > 2:
        topBound = findTopAnkle(j) + 50
        cv2.putText(image, "top bound", (950, int(topBound)+gapy), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        topBound = 0
    print(f'Frame: {frame_count}, Top bound: {topBound}')
    topScores = scoreRank(j, s, topBound)
    if topScores == False:
        cv2.putText(image, str(frame_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        return image, True
    legal, comb = checkPos(topScores, b)
    if legal:
        for k in comb:
            i = topScores[k]
            keypoints = j[i]
            keypoints = keypoints[:, :].reshape(-1, 3)
            playerJoints.append(j[i].tolist()[5:17])
            for p in range(keypoints.shape[0]):
                cv2.circle(image, (int(keypoints[p, 0]) + gapx, int(keypoints[p, 1]) + gapy), 3, (0, 0, 255),
                           thickness=-1, lineType=cv2.FILLED)
            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                rgb = rgb * 255
                # join the keypoint pairs to draw the skeletal structure
                cv2.line(image, (int(keypoints[e, 0][0]) + gapx, int(keypoints[e, 1][0]) + gapy),
                         (int(keypoints[e, 0][1]) + gapx, int(keypoints[e, 1][1]) + gapy),
                         tuple(rgb), 2, lineType=cv2.LINE_AA)
        cv2.putText(image, str(frame_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        return image, playerJoints
    else:
        cv2.putText(image, str(frame_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        return image, True
