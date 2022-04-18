import os
import json
import torch
import torchvision
import cv2
import court_size_version_utils as c_utils
from PIL import Image
from torchvision.transforms import transforms
import scene_utils
from scene_utils import scene_classifier

def check_dir(path):
    isExit = os.path.exists(path)
    if not isExit:
        os.mkdir(path)

court_box_A = [239, 433, 1673, 1007]
court_mp_A = [[590, 434, 1], [1310, 434, 1], [476, 624, 1],[1427, 623, 1],[256, 1000, 1],[1660, 1002, 1]]

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

vid_path = '../inputs/full_game_1080p/CTC_A_jump.mp4'
paths = [f"../outputs/videos/{vid_path.split('/')[-1].split('.')[0]}",
         f"../outputs/scene_data/F_data/{vid_path.split('/')[-1].split('.')[0]}",
         f"../outputs/scene_data/T_data/{vid_path.split('/')[-1].split('.')[0]}",
         f"../outputs/joint_data/{vid_path.split('/')[-1].split('.')[0]}"]
for path in paths:
    check_dir(path)

cap = cv2.VideoCapture(vid_path)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the video frames' width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# set the save path
save_path = f"../outputs/videos/{vid_path.split('/')[-1].split('.')[0]}/{vid_path.split('/')[-1].split('.')[0]}.mp4"
# define codec and create VideoWriter object
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
frame_count = 0
save_count = 0
time_rate = 0.1
FPS = cap.get(5)
frame_rate = int(FPS) * time_rate
total_frame_count = int(cap.get(7))
total_save_count = int(total_frame_count / frame_rate)
jointList = []
# read until end of video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # orig_frame = frame
        if frame_count % frame_rate == 0:
            print(save_count,' / ',total_save_count)
            sceneImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            sceneImg = scene_utils.preprocess(sceneImg, device)
            p = scene_utils.predict(sceneModel, sceneImg)
            if p == 1:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                save_path = f"../outputs/scene_data/T_data/{vid_path.split('/')[-1].split('.')[0]}/{vid_path.split('/')[-1].split('.')[0]}{'_'+str(save_count)}.jpg"
                pil_image.save(save_path)
                # transform the image
                image = transform(pil_image)
                # add a batch dimension
                image = image.unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image)
                output_image, playerJoints = c_utils.draw_keypoints(outputs, frame, save_count, court_box_A, court_mp_A)
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
                save_count += 1
                # cv2.imshow('Pose detection frame', output_image)
                out.write(output_image)
            else:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                save_path = f"../outputs/scene_data/F_data/{vid_path.split('/')[-1].split('.')[0]}/{vid_path.split('/')[-1].split('.')[0]}{'_'+str(save_count)}.jpg"
                pil_image.save(save_path)
                cv2.putText(frame, str(save_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1,cv2.LINE_AA)
                frame_count += 1
                save_count += 1
                # cv2.imshow('Pose detection frame', frame)
                out.write(frame)
        else:
            # cv2.putText(orig_frame, str(frame_count), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1,
            #             cv2.LINE_AA)
            frame_count += 1
            # cv2.imshow('Pose detection frame', frame)
            # out.write(orig_frame)
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
print(f'Frame count:{frame_count}')
print(f'Save count:{save_count}')

framesDict = {'frames': jointList}
save_path = f"../outputs/joint_data/{vid_path.split('/')[-1].split('.')[0]}/{vid_path.split('/')[-1].split('.')[0]}.json"
with open(save_path, 'w') as f:
    json.dump(framesDict, f, indent=2)
