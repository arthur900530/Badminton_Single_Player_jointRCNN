import scene_utils
from scene_utils import scene_classifier
import torch
from PIL import Image

path1 = '../inputs/big.jpg'
path2 = '../inputs/calculate2.jpg'
model_path = 'scene_classifier.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = scene_utils.build_model(model_path, device)

img1 = scene_utils.preprocess(Image.open(path2), device)
p = scene_utils.predict(model, img1)
print(p)

