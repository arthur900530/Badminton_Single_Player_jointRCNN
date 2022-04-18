import os
import torch
import cv2
from PIL import Image
import scene_utils
from scene_utils import scene_classifier

def check_dir(path):
    isExit = os.path.exists(path)
    if not isExit:
        os.mkdir(path)


def check_type(last_type, wait_list):
    if last_type == 0:
        for pair in wait_list:
            if pair[0] == 1:
                continue
            else:
                return False
    else:
        for pair in wait_list:
            if pair[0] == 0:
                continue
            else:
                return False
    return True

# first 1 image will be sent to get court keypoint
def video_preprocess(vid_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info_dict = {}
    wait_list = []
    # scene model
    model_path = 'scene_classifier.pt'
    sceneModel = scene_utils.build_model(model_path, device)

    # set up the paths
    # vid_path = '../inputs/full_game_1080p/CTC_A_jump.mp4'
    paths = [f"../outputs/videos",
             f"../outputs/scene_data",
             f"../outputs/scene_data/F_data",
             f"../outputs/scene_data/T_data",
             f"../outputs/joint_data",
             f"../outputs/videos/{vid_path.split('/')[-1].split('.')[0]}",
             f"../outputs/scene_data/F_data/{vid_path.split('/')[-1].split('.')[0]}",
             f"../outputs/scene_data/T_data/{vid_path.split('/')[-1].split('.')[0]}",
             f"../outputs/joint_data/{vid_path.split('/')[-1].split('.')[0]}"]
    for path in paths:
        check_dir(path)
    info_dict['video_name'] = vid_path.split('/')[-1].split('.')[0]
    info_dict['video_path'] = paths[0]
    info_dict['other_scene'] = paths[1]
    info_dict['court_scene'] = paths[2]
    info_dict['joint_path'] = paths[3]

    # setup video capturer
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
        return False, info_dict
    FPS = cap.get(5)
    frame_count = 0
    save_count = 0
    time_rate = 0.1
    frame_rate = int(FPS) * time_rate
    total_frame_count = int(cap.get(7))
    total_save_count = int(total_frame_count / frame_rate) + 1
    last_type = 0
    score_count = 0

    # video preprocess
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % frame_rate == 0:
                print(save_count,' / ',total_save_count, 'Score: ', score_count)
                frame_count += 1
                scene_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                scene_img = scene_utils.preprocess(scene_img, device)
                p = scene_utils.predict(sceneModel, scene_img)
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if len(wait_list) < 5:
                    wait_list.append((p, pil_image))
                else:
                    tup = wait_list.pop(0)
                    wait_list.append((p, pil_image))
                    if last_type != tup[0]:
                        if last_type == 0:
                            if check_type(last_type, wait_list):
                                last_type = 1
                                score_count += 1
                                score_path = f"../outputs/scene_data/T_data/{vid_path.split('/')[-1].split('.')[0]}/score_{score_count}"
                                check_dir(score_path)
                                save_path = score_path + f"/{vid_path.split('/')[-1].split('.')[0]}{'_' + str(save_count)}.jpg"
                                tup[1].save(save_path)
                                save_count += 1
                            else:
                                save_path = f"../outputs/scene_data/F_data/{vid_path.split('/')[-1].split('.')[0]}/{vid_path.split('/')[-1].split('.')[0]}{'_' + str(save_count)}.jpg"
                                tup[1].save(save_path)
                                save_count += 1
                        else:
                            if check_type(last_type, wait_list):
                                last_type = 0
                                save_path = f"../outputs/scene_data/F_data/{vid_path.split('/')[-1].split('.')[0]}/{vid_path.split('/')[-1].split('.')[0]}{'_' + str(save_count)}.jpg"
                                tup[1].save(save_path)
                                save_count += 1
                            else:
                                score_path = f"../outputs/scene_data/T_data/{vid_path.split('/')[-1].split('.')[0]}/score_{score_count}"
                                save_path = score_path + f"/{vid_path.split('/')[-1].split('.')[0]}{'_' + str(save_count)}.jpg"
                                tup[1].save(save_path)
                                save_count += 1
                    else:
                        if last_type == 1:
                            score_path = f"../outputs/scene_data/T_data/{vid_path.split('/')[-1].split('.')[0]}/score_{score_count}"
                            save_path = score_path + f"/{vid_path.split('/')[-1].split('.')[0]}{'_' + str(save_count)}.jpg"
                            tup[1].save(save_path)
                            save_count += 1
                        else:
                            save_path = f"../outputs/scene_data/F_data/{vid_path.split('/')[-1].split('.')[0]}/{vid_path.split('/')[-1].split('.')[0]}{'_' + str(save_count)}.jpg"
                            tup[1].save(save_path)
                            save_count += 1
            else:
                frame_count += 1
        else:
            break

    # release VideoCapture()
    cap.release()
    # clear wait list
    for i in range(len(wait_list)):
        if last_type == 0:
            save_path = f"../outputs/scene_data/F_data/{vid_path.split('/')[-1].split('.')[0]}/{vid_path.split('/')[-1].split('.')[0]}{'_' + str(save_count)}.jpg"
            wait_list[i][1].save(save_path)
            save_count += 1
        else:
            score_path = f"../outputs/scene_data/T_data/{vid_path.split('/')[-1].split('.')[0]}/score_{score_count}"
            save_path = score_path + f"/{vid_path.split('/')[-1].split('.')[0]}{'_' + str(save_count)}.jpg"
            wait_list[i][1].save(save_path)
            save_count += 1

    # calculate and print the average FPS
    print(f'Frame count:{frame_count} / {total_frame_count}')
    print(f'Save count:{save_count} / {total_save_count}')
    print(f'Score count:{score_count}')
    info_dict['frame_count'] = total_frame_count
    info_dict['save_count'] = total_save_count
    info_dict['score_count'] = score_count

    return True, info_dict
