import json
import torch
import torchvision
import cv2
import utils
import time
from PIL import Image
from torchvision.transforms import transforms
import scene_utils
from scene_utils import scene_classifier

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
# initialize the model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                               num_keypoints=17)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()
# scene model
model_path = 'scene_classifier.pt'
sceneModel = scene_utils.build_model(model_path, device)

vid_path = '../inputs/full_game_1080p/CTC_A_Test_Trim.mp4'
cap = cv2.VideoCapture(vid_path)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the video frames' width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# set the save path
save_path = f"../outputs/{vid_path.split('/')[-1].split('.')[0]}.mp4"
# define codec and create VideoWriter object
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
frame_count = 0
time_rate = 0.2
FPS = cap.get(5)
frame_rate = int(FPS) * time_rate
total_fps = 0
jointList = []
# read until end of video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        orig_frame = frame
        if frame_count % frame_rate == 0:
            # 1080p:1350,180
            # 720p:800,120
            sceneImg = Image.fromarray(cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR))
            # sceneImg.thumbnail((384, 216))
            sceneImg = scene_utils.preprocess(sceneImg, device)
            p = scene_utils.predict(sceneModel, sceneImg)
            if p == 1:
                croppedFrame = frame[180:frame_height, int(frame_width / 2 - 675):int(frame_width / 2 + 675)]
                # pil_image = Image.fromarray(croppedFrame).convert('RGB')
                pil_image = Image.fromarray(cv2.cvtColor(croppedFrame, cv2.COLOR_RGB2BGR))
                # transform the image
                image = transform(pil_image)
                # add a batch dimension
                image = image.unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image)
                output_image, playerJoints = utils.draw_keypoints(outputs, orig_frame, frame_width, frame_count)
                if playerJoints != True:
                    for points in playerJoints:
                        for i, joints in enumerate(points):
                            points[i] = joints[0:2]
                    jointList.append({
                        'frame': frame_count,
                        'joint': playerJoints,
                        'label': -1,
                        'type': 'TYPE'
                    })
                frame_count += 1
                cv2.imshow('Pose detection frame', output_image)
                out.write(output_image)
            else:
                start_time = time.time()
                end_time = time.time()
                cv2.putText(orig_frame, str(frame_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1,
                           cv2.LINE_AA)
                frame_count += 1
                cv2.imshow('Pose detection frame', orig_frame)
                out.write(orig_frame)
        else:
            cv2.putText(orig_frame, str(frame_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)
            frame_count += 1
            cv2.imshow('Pose detection frame', orig_frame)
            out.write(orig_frame)
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
print(f'Frame count:{frame_count}')

framesDict = {'frames': jointList}

with open('../outputs/jointList.json', 'w') as f:
    json.dump(framesDict, f, indent=2)
