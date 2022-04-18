import cv2
from PIL import Image

vid_path = '../inputs/full_game/CTC_F.mp4'
cap = cv2.VideoCapture(vid_path)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
frame_count = 0
time_rate = 0.2
FPS = cap.get(5)
frame_rate = int(FPS)*time_rate
saved_count = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        if frame_count % frame_rate == 0:
            # pil_image = Image.fromarray(frame).convert('RGB')
            # pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            pil_img = pil_image.thumbnail((384, 216))
            save_path = f"../outputs/full_game/{vid_path.split('/')[-1].split('.')[0]}/{vid_path.split('/')[-1].split('.')[0]}{str(frame_count)}.jpg"
            pil_image.save(save_path)
            with open('../outputs/jointList.json', 'w') as f:
                json.dump(framesDict, f, indent=2)
            print(frame_count)
            saved_count += 1
        frame_count += 1
    else:
        break

cap.release()
print(f'Frame count:{frame_count}')
print(f'Saved count:{saved_count}')