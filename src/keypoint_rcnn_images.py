import torch
import torchvision
import numpy as np
import cv2
import utils
from PIL import Image
from torchvision.transforms import transforms as transforms

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

image_path = '../inputs/check2.jpg'
image = Image.open(image_path).convert('RGB')
width, height = image.size
# NumPy copy of the image for OpenCV functions
orig_numpy = np.array(image, dtype=np.float32)
# convert the NumPy image to OpenCV BGR format
orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.
# transform the image
# image = image[180:height, int(width / 2 - 490):int(width / 2 + 490)]
image = image.crop((int(width / 2 - 675),180,int(width / 2 + 675),height))
image = transform(image)
# add a batch dimension
image = image.unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(image)
jointList = []
output_image, playerJoints = utils.draw_keypoints(outputs, orig_numpy, width, 0)
if(playerJoints != True):
    for points in playerJoints:
        for i, joints in enumerate(points):
            points[i] = joints[0:2]
    jointList.append({
                      'frame':0,
                      'joint':playerJoints,
                      'label':-1
                    })

# visualize the image
cv2.imshow('Keypoint image', output_image)
cv2.waitKey(0)

# set the save path
save_path = f"../outputs/{image_path.split('/')[-1].split('.')[0]}.jpg"
cv2.imwrite(save_path, output_image*255.)
