import cv2
import json
from preprocess import get_path
import matplotlib
import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import transforms


def score_process(score_path, court_points):
    joint_list = []
    score_frame_count = 0
    score_frame_paths = get_path(score_path)
    total_count = len(score_frame_paths)

    vid_name = score_path.split('/')[-2]
    score_num = score_path.split('/')[-1]
    out = cv2.VideoWriter(f"../test/video/{vid_name}/{score_num}/{vid_name}-{score_num}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (1920, 1080))
    for path in score_frame_paths:
        output_image, player_joints = frame_joint_process(path, score_frame_count, court_points)
        out.write(output_image/255.)
        score_frame_count += 1
        print(f"{score_frame_count} / {total_count}")
        if player_joints != True:
            joint_list.append({
                'frame': score_frame_count,
                'joint': player_joints,
                'label': -1,
                'type': 'TYPE'
            })
        else:
            joint_list.append({
                'frame': score_frame_count,
                'joint': 0,
                'label': -1,
                'type': 'TYPE'
            })
            print("There's no players in the court")
    cv2.destroyAllWindows()
    framesDict = {'frames': joint_list}
    save_path = f"../test/joint_data/{vid_name}/{score_num}/{vid_name}-{score_num}.json"
    with open(save_path, 'w') as f:
        json.dump(framesDict, f, indent=2)
    return True


def frame_joint_process(frame_path, score_frame_count, court_points):
    vid_name = frame_path.split('/')[-3]
    score_num = frame_path.split('/')[-2]
    # save_path = f"../test/video/img/{vid_name}/{score_num}/{vid_name}-{score_frame_count}.jpg"
    pil_image = Image.open(frame_path).convert('RGB')
    orig_numpy = np.array(pil_image, dtype=np.float32)
    orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.

    transform = transforms.Compose([transforms.ToTensor()])
    # initialize the model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    with torch.no_grad():
        image = transform(pil_image)
        image = image.unsqueeze(0).to(device)
        outputs = model(image)
    output_image, playerJoints = draw_keypoints(outputs, orig_numpy, score_frame_count, court_points)

    # cv2.imwrite(save_path, output_image*255.)

    return output_image, playerJoints


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


def in_court(joint,  court_info, court_points):
    l_a = court_info[0]
    l_b = court_info[1]
    r_a = court_info[2]
    r_b = court_info[3]
    mp_y = court_info[4]
    ankle_x = (joint[15][0] + joint[16][0])/2
    ankle_y = (joint[15][1] + joint[16][1])/2
    top = ankle_y > court_points[0][1] - 80
    bottom = ankle_y < court_points[5][1] + 20
    lmp_x = (ankle_y - l_b) / l_a
    rmp_x = (ankle_y - r_b) / r_a
    left = ankle_x > lmp_x
    right = ankle_x < rmp_x

    if left and right and top and bottom:
        return True
    else:
        return False


def score_rank(joints, court_info, court_points):
    indexes = []
    for i in range(len(joints)):
        if in_court(joints[i], court_info, court_points):
            indexes.append(i)
    if len(indexes) < 2:
        return False
    else:
        return indexes


def draw_keypoints(outputs, image, frame_count, court_points):
    edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),(5, 7),
             (7, 9), (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16), (5, 6)]
    playerJoints = []
    b = outputs[0]['boxes'].cpu().detach().numpy()
    j = outputs[0]['keypoints'].cpu().detach().numpy()
    # court formula------------------------------------------------------------------------
    l_a = (court_points[0][1] - court_points[4][1]) / (court_points[0][0] - court_points[4][0])
    l_b = court_points[0][1] - l_a * court_points[0][0]
    r_a = (court_points[1][1] - court_points[5][1]) / (court_points[1][0] - court_points[5][0])
    r_b = court_points[1][1] - r_a * court_points[1][0]
    mp_y = (court_points[2][1] + court_points[3][1]) / 2
    court_info = [l_a, l_b, r_a, r_b, mp_y]
    # --------------------------------------------------------------------------------------
    topScores = score_rank(j, court_info, court_points)
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
