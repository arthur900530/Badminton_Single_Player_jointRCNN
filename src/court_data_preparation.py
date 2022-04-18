import copy
import cv2
from PIL import Image
import scene_utils
import torch
import json
import random
from scene_utils import scene_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'scene_classifier.pt'
sceneModel = scene_utils.build_model(model_path, device)

vid_path = '../inputs/full_game_1080p/CTC_C.mp4'
cap = cv2.VideoCapture(vid_path)
if not cap.isOpened():
    print('Error while trying to read video. Please check path again')
frame_count = 0
time_rate = 1.0
FPS = cap.get(5)
frame_rate = int(FPS) * time_rate
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
saved_count = 0

court_A_points = [[[590, 434, 1], [1310, 434, 1], [476, 624, 1],[1427, 623, 1],[256, 1000, 1],[1660, 1002, 1]]]
court_A_box = [[239, 433, 1673, 1007]]

court_B_points = [[[627, 539, 1], [1293, 538, 1], [543, 711, 1],[1378, 711, 1],[400, 1004, 1],[1523, 1007, 1]]]
court_B_box = [[390, 534, 1534, 1009]]

court_C_points = [[[625, 458, 1], [1300, 455, 1], [554, 665, 1],[1369, 664, 1],[442, 990, 1],[1482, 989, 1]]]
court_C_box = [[430, 454, 1489, 994]]

def randomCrop(frame, frame_width, frame_height, i_points, i_box):
    crop_x = random.randint(0, 1)
    crop_y = random.randint(0, 1)
    points = copy.deepcopy(i_points)
    box = copy.deepcopy(i_box)
    if crop_y:
        crop_size = random.randint(30, 180)
        box[0][1] -= crop_size
        box[0][3] -= crop_size
        for p in points[0]:
            p[1] -= crop_size
        frame = frame[crop_size:frame_height, 0:int(frame_width)]
        frame_height -= crop_size
    if crop_x:
        crop_size = random.randint(50, 320)
        box[0][0] -= int(crop_size / 2)
        box[0][2] -= int(crop_size / 2)
        for p in points[0]:
            p[0] -= int(crop_size / 2)
        croppedFrame = frame[0:frame_height, int(crop_size / 2):int(frame_width - crop_size / 2)]
        return croppedFrame, points, box
    else:
        return frame, points, box

while cap.isOpened():
    ret, frame = cap.read()
    if ret and saved_count < 400:
        if frame_count % frame_rate == 0:
            frame_count += 1
            sceneImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            sceneImg = scene_utils.preprocess(sceneImg, device)
            isCourt = scene_utils.predict(sceneModel, sceneImg)
            if isCourt == 1:
                saved_count += 1
                print(saved_count)
                croppedFrame, points, box = randomCrop(frame, frame_width, frame_height, court_C_points, court_C_box)
                # draw six points
                # for p in range(len(points[0])):
                #     print(p)
                #     cv2.circle(croppedFrame, (int(points[0][p][0]), int(points[0][p][1])), 3, (0, 0, 255),
                #            thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(croppedFrame, str(saved_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                pil_image = Image.fromarray(cv2.cvtColor(croppedFrame, cv2.COLOR_RGB2BGR))
                save_path = f"../outputs/court_data/{vid_path.split('/')[-1].split('.')[0]}/images/{vid_path.split('/')[-1].split('.')[0]}{str(saved_count)}.jpg"
                pil_image.save(save_path)
                # json
                json_path = f"../outputs/court_data/{vid_path.split('/')[-1].split('.')[0]}/annotations/{vid_path.split('/')[-1].split('.')[0]}{str(saved_count)}.json"
                annotations = {
                    "bboxes": box,
                    "keypoints": points,
                    "labels": [1],
                    "frame_num": saved_count
                }
                with open(json_path, 'w') as f:
                    json.dump(annotations, f)
        else:
            frame_count += 1
            continue
    else:
        break

cap.release()
print(f'Frame count:{frame_count}')
print(f'Saved count:{saved_count}')
