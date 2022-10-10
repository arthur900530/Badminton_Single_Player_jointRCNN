import copy
import csv, cv2, json, time, numpy as np, matplotlib
from PIL import Image
import torch, torchvision
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import scene_utils, transformer_utils
from pipeline_utils import check_type, score_rank, check_pos
from shot_recognition import check_hit_frame, add_result
from utility import check_dir, get_path, parse_time, top_bottom, correction, extension, type_classify
from transformer_utils import coordinateEmbedding, PositionalEncoding, Optimus_Prime
from scene_utils import scene_classifier


class video_resolver:
    def __init__(self, vid_path, output_base='E:/test_videos'):
        self.start_time = time.time()
        self.base = output_base
        self.vid_path = vid_path
        self.vid_name = vid_path.split('/')[-1].split('.')[0]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
        self.model.to(self.device).eval()
        self.scene_model = scene_utils.build_model('model_weights/scene_classifier.pt', self.device)
        self.bsp_model = transformer_utils.build_model('model_weights/weights/clean_seq_labling_ultimate_2.pt')

        self.court_kp_model = torch.load('model_weights/court_kpRCNN.pth')
        self.court_kp_model.to(self.device).eval()

        self.court_kp_model_old = torch.load('model_weights/court_kpRCNN_old.pth')
        self.court_kp_model_old.to(self.device).eval()

        self.paths = [f"{self.base}/outputs",
                      f"{self.base}/outputs/{self.vid_name}"]

        for path in self.paths:
            check_dir(path)

        self.cap = cv2.VideoCapture(vid_path)
        self.start_recording = True
        self.start_frame = 0
        self.end_frame = 0

        if not self.cap.isOpened():
            print('Error while trying to read video. Please check path again')

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.frame_count = 1
        self.saved_count = 0
        self.time_rate = 0.1
        self.FPS = self.cap.get(5)
        self.frame_rate = round(int(self.FPS) * self.time_rate)
        self.total_frame_count = int(self.cap.get(7))
        self.total_saved_count = int(self.total_frame_count / self.frame_rate)

        self.court_points = None
        self.true_court_points = None
        self.multi_points = None
        self.court_info = None

        self.last_type = 0
        self.game = 1
        self.zero_count = 0
        self.score = 0
        self.last_score = 0
        self.top_bot_score = [0, 0]
        self.one_count = 0

    # get and set the court information
    def get_court_info(self, img):
        with torch.no_grad():
            img = F.to_tensor(img)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            output = self.court_kp_model(img)
            output_old = self.court_kp_model_old(img)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs],
                                            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])
        scores_old = output_old[0]['scores'].detach().cpu().numpy()
        high_scores_idxs_old = np.where(scores_old > 0.7)[0].tolist()
        post_nms_idxs_old = torchvision.ops.nms(output_old[0]['boxes'][high_scores_idxs_old],
                                            output_old[0]['scores'][high_scores_idxs_old], 0.3).cpu().numpy()
        keypoints_old = []
        for kps in output_old[0]['keypoints'][high_scores_idxs_old][post_nms_idxs_old].detach().cpu().numpy():
            keypoints_old.append([list(map(int, kp[:2])) for kp in kps])

        self.true_court_points = copy.deepcopy(keypoints[0])
        self.multi_points = extension(correction(np.array(keypoints[0]))).tolist()
        print(self.multi_points)
        keypoints_old[0][0][0] -= 80
        keypoints_old[0][0][1] -= 80
        keypoints_old[0][1][0] += 80
        keypoints_old[0][1][1] -= 80
        keypoints_old[0][2][0] -= 80
        keypoints_old[0][3][0] += 80
        keypoints_old[0][4][0] -= 80
        keypoints_old[0][4][1] = min(keypoints_old[0][4][1] + 80, self.frame_height - 40)
        keypoints_old[0][5][0] += 80
        keypoints_old[0][5][1] = min(keypoints_old[0][5][1] + 80, self.frame_height - 40)
        self.court_points = keypoints_old[0]

        l_a = (self.true_court_points[0][1] - self.true_court_points[4][1]) / (
                    self.true_court_points[0][0] - self.true_court_points[4][0])
        l_b = self.true_court_points[0][1] - l_a * self.true_court_points[0][0]
        r_a = (self.true_court_points[1][1] - self.true_court_points[5][1]) / (
                    self.true_court_points[1][0] - self.true_court_points[5][0])
        r_b = self.true_court_points[1][1] - r_a * self.true_court_points[1][0]
        mp_y = (self.true_court_points[2][1] + self.true_court_points[3][1]) / 2
        self.court_info = [l_a, l_b, r_a, r_b, mp_y]

        return True

    def draw_key_points(self, outputs, image):
        edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12), (5, 7),
                 (7, 9), (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16), (5, 6)]
        # bounds = get_area_bound(self.court_points)
        color1 = (217, 146, 65)
        color2 = (90, 31, 255)
        color3 = (90, 48, 0)
        color4 = (66, 6, 219)
        playerJoints = []
        b = outputs[0]['boxes'].cpu().detach().numpy()
        j = outputs[0]['keypoints'].cpu().detach().numpy()
        topScores = score_rank(self.court_info, self.court_points, j)
        if topScores == False:
            return image, True

        fit, combination = check_pos(self.court_info[4], topScores, b)

        top, bot = top_bottom([j[topScores[combination[0]]], j[topScores[combination[1]]]])

        if fit:
            for c in combination:
                if c == top:
                    color = color1
                    sub_color = color3
                else:
                    color = color2
                    sub_color = color4
                i = topScores[c]
                keypoints = j[i]
                box = b[i]
                # print(box, box[2])
                keypoints = keypoints[:, :].reshape(-1, 3)
                playerJoints.append(j[i].tolist())
                overlay = image.copy()

                # court bound point
                # for bound in bounds:
                #     cv2.circle(overlay, tuple((int(self.frame_width / 2 - 2), int(bound[0]))), 5, (255, 255, 0), 10)
                #     cv2.circle(overlay, tuple((int(self.frame_width / 2 - 2), int(bound[1]))), 5, (255, 255, 0), 10)

                c_edges = [[0, 1],[0, 5],[1, 2],[1, 6],[2, 3],[2, 7],[3, 4],[3, 8],[4, 9],
                           [5, 6],[5, 10],[6, 7],[6, 11],[7, 8],[7, 12],[8, 9],[8, 13],[9, 14],
                           [10, 11],[10, 15],[11, 12],[11, 16],[12, 13],[12, 17],[13, 14],[13, 18],
                           [14, 19],[15, 16],[15, 20],[16, 17],[16, 21],[17, 18],[17, 22],[18, 19],
                           [18, 23],[19, 24],[20, 21],[20, 25],[21, 22],[21, 26],[22, 23],[22, 27],
                           [23, 24],[23, 28],[24, 29],[25, 26],[25, 30],[26, 27],[26, 31],[27, 28],
                           [27, 32],[28, 29],[28, 33],[29, 34],[30,31],[31,32],[32,33],[33,34]]
                for e in c_edges:
                    cv2.line(overlay, (int(self.multi_points[e[0]][0]), int(self.multi_points[e[0]][1])),
                             (int(self.multi_points[e[1]][0]), int(self.multi_points[e[1]][1])),
                             (53, 195, 242), 2, lineType=cv2.LINE_AA)
                # for kps in [self.court_points]:
                for kps in [self.multi_points]:
                    for idx, kp in enumerate(kps):
                        cv2.circle(overlay, tuple(kp), 2, (5, 135, 242), 10)

                # cv2.ellipse(overlay, (int((box[2] + box[0]) / 2), int(box[3])),
                #             (int((box[2] - box[0]) / 1.8), int((box[3] - box[1]) / 10)),
                #             0, 0, 360, color, 15)
                # cv2.ellipse(overlay, (int((box[2] + box[0]) / 2), int(box[3])),
                #             (int((box[2] - box[0]) / 1.8), int((box[3] - box[1]) / 10)),
                #             0, int(((self.one_count - 1) * 10)),
                #             int((30 + (self.one_count - 1) * 10)), sub_color, 6)

                alpha = 0.4
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                for p in range(keypoints.shape[0]):
                    cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1,
                               lineType=cv2.FILLED)
                    # cv2.putText(image, str(i), (int(keypoints[15, 0]), int(keypoints[15, 1])), cv2.FONT_HERSHEY_DUPLEX,
                    #             1, (0, 255, 255), 1,
                    #             cv2.LINE_AA)
                for ie, e in enumerate(edges):
                    # get different colors for the edges
                    rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    rgb = rgb * 255
                    # join the keypoint pairs to draw the skeletal structure
                    cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                             (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                             tuple(rgb), 2, lineType=cv2.LINE_AA)
            return image, playerJoints
        else:
            return image, True

    def resolve(self):
        joint_list = []
        joint_img_list = []
        wait_list = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_count % self.frame_rate == 0:
                    sceneImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    sceneImg = scene_utils.preprocess(sceneImg, self.device)
                    # slice video into score videos
                    p = scene_utils.predict(self.scene_model, sceneImg)
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if len(wait_list) < 5:
                        wait_list.append((p, pil_image, frame))
                    else:
                        tup = wait_list.pop(0)
                        wait_list.append((p, pil_image, frame))
                        p = tup[0]
                        pil_image = tup[1]
                        frame = tup[2]
                        if p != self.last_type:
                            correct = check_type(self.last_type, wait_list)
                            if not correct:
                                if p == 1:
                                    p = 0
                                else:
                                    p = 1
                            else:
                                if p == 0:
                                    if len(joint_list) / self.one_count > 0.6 and self.one_count > 25:  # 25 is changable
                                        if not self.start_recording:
                                            self.start_recording = True
                                            self.end_frame = self.frame_count
                                        framesDict = {'frames': joint_list}
                                        store_path = f"{self.base}/outputs/{self.vid_name}/game_{self.game}_score_{self.score}"
                                        check_dir(store_path)
                                        start_time = parse_time(self.FPS, self.start_frame)
                                        end_time = parse_time(self.FPS, self.end_frame)
                                        out = cv2.VideoWriter(f"{store_path}/game_{self.game}_score_{self.score}.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                                              int(self.FPS / self.frame_rate), (self.frame_width, self.frame_height))
                                        for img in joint_img_list:
                                            out.write(img)
                                        joint_img_list = []
                                        out.release()

                                        save_path = f"{store_path}/score_{self.score}_joint.json"
                                        with open(save_path, 'w') as f:
                                            json.dump(framesDict, f, indent=2)

                                        joint_list = torch.tensor(np.array(transformer_utils.get_data(save_path)), dtype=torch.float32).to(self.device)
                                        orig_joint_list = np.squeeze(np.array(transformer_utils.get_original_data(save_path)), axis=0)

                                        shuttle_direction = transformer_utils.predict(self.bsp_model, joint_list).tolist()
                                        print(shuttle_direction)
                                        dz_count = 0
                                        for d in shuttle_direction:
                                            if d == 0:
                                                dz_count += 1
                                        if dz_count/len(shuttle_direction) < 0.9:
                                            # correct = transformer_utils.check_pos_and_score(shuttle_direction, orig_joint_list, self.multi_points, self.top_bot_score)
                                            # print('Score correct...') if correct else print('Wrong score...')
                                            shot_list, move_dir_list = check_hit_frame(shuttle_direction, orig_joint_list, self.true_court_points, self.multi_points)
                                            print(shot_list, move_dir_list)
                                            offensive, pos = type_classify(shot_list)
                                            success = add_result(f'{store_path}/', f"{store_path}/game_{self.game}_score_{self.score}.mp4", shot_list, move_dir_list, self.true_court_points)
                                            if offensive is None:
                                                top_type = None
                                                bot_type = None
                                            elif offensive:
                                                top_type = True if pos else False
                                                bot_type = True if not pos else False
                                            elif not offensive:
                                                top_type = False
                                                bot_type = False

                                            info_dict = {
                                                'id':None,
                                                'game':self.game,
                                                'score':self.score,
                                                'time':[start_time, end_time],
                                                'long rally':True if len(shot_list) > 15 else False,
                                                'shuttle direction':shuttle_direction,
                                                'shot list':shot_list,
                                                'move direction list':move_dir_list,
                                                'top player type':top_type,
                                                'bot player type': bot_type,
                                                'winner':None,
                                                'top bot score':self.top_bot_score
                                            }
                                            save_path = f"{store_path}/game_{self.game}_score_{self.score}_info.json"
                                            with open(save_path, 'w') as f:
                                                json.dump(info_dict, f, indent=2)
                                            if success:
                                                print(f'Finish score_{self.score}')

                                            if self.score != 0:
                                                with open(f"{self.base}/outputs/{self.vid_name}/game_{self.game}_score_{self.score-1}/game_{self.game}_score_{self.score-1}_info.json", 'r') as score_json:
                                                    dict = json.load(score_json)
                                                if 1 in shuttle_direction and 2 in shuttle_direction:
                                                    winner = True if shuttle_direction.index(1) < shuttle_direction.index(2) else False
                                                elif 1 in shuttle_direction and 2 not in shuttle_direction:
                                                    winner = False
                                                else:
                                                    winner = True

                                                dict['winner'] = winner
                                                if winner:
                                                    dict['top bot score'][0] += 1
                                                    self.top_bot_score[0] += 1
                                                else:
                                                    dict['top bot score'][1] += 1
                                                    self.top_bot_score[1] += 1
                                                with open(f"{self.base}/outputs/{self.vid_name}/game_{self.game}_score_{self.score - 1}/game_{self.game}_score_{self.score - 1}_info.json", 'w') as f:
                                                    json.dump(dict, f, indent=2)

                                            if self.score == 0 and self.game != 1:
                                                with open(f"{self.base}/outputs/{self.vid_name}/game_{self.game-1}_score_{self.last_score-1}/game_{self.game-1}_score_{self.last_score-1}_info.json", 'r') as score_json:
                                                    dict = json.load(score_json)
                                                winner = True if shuttle_direction.index(1) < shuttle_direction.index(2) else False
                                                dict['winner'] = winner
                                                if winner:
                                                    dict['top bot score'][0] += 1
                                                    self.top_bot_score[0] += 1
                                                else:
                                                    dict['top bot score'][1] += 1
                                                    self.top_bot_score[1] += 1
                                                with open(f"{self.base}/outputs/{self.vid_name}/game_{self.game-1}_score_{self.last_score - 1}/game_{self.game-1}_score_{self.last_score - 1}_info.json", 'w') as f:
                                                    json.dump(dict, f, indent=2)
                                            # if self.game == 3:
                                            #     half = False
                                            #     for sc in self.top_bot_score:
                                            #         if sc == 11:
                                            #             half = True
                                            #     if half:
                                            #         temp = self.top_bot_score

                                            joint_list = []
                                            self.score += 1
                                            self.one_count = 0
                                    else:
                                        joint_list = []
                                        self.one_count = 0
                                else:
                                    if self.court_points == None:
                                        _ = self.get_court_info(img=wait_list[2][2])
                                        print(self.true_court_points)
                                        print("Get!")
                                        self.court_kp_model = None
                                    # self.one_count += 1
                                self.last_type = p
                        if p == 1:
                            if self.start_recording:
                                self.start_frame = self.frame_count
                                self.start_recording = False
                            self.one_count += 1
                            image = self.transform(pil_image)
                            image = image.unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                outputs = self.model(image)
                            output_image, player_joints = self.draw_key_points(outputs, frame)

                            if player_joints != True:
                                for points in player_joints:
                                    for i, joints in enumerate(points):
                                        points[i] = joints[0:2]
                                joint_list.append({
                                    'joint': player_joints,
                                })
                            joint_img_list.append(output_image)
                            self.zero_count = 0
                        else:
                            self.zero_count += 1

                        if self.zero_count > 1100 and self.zero_count < 1500 and self.score != 0 and self.game < 3:
                            self.last_score = self.score
                            print(self.zero_count, '='*50)
                            self.zero_count = 0
                            self.top_bot_score = [0, 0]
                            self.game += 1
                            self.score = 0
                        elif self.zero_count > 1500:
                            self.zero_count = 0

                        self.saved_count += 1
                        print(self.saved_count, ' / ', self.total_saved_count)
                        self.frame_count += 1
                else:
                    self.frame_count += 1
            else:
                break
        g = self.game-1 if self.game <= 3 else 3
        with open(f"{self.base}/outputs/{self.vid_name}/game_{g}_score_{self.last_score-1}/game_{g}_score_{self.last_score-1}_info.json", 'r') as score_json:
            dict = json.load(score_json)
        dict['winner'] = True if self.top_bot_score[0] > self.top_bot_score[1] else False
        if dict['winner']:
            self.top_bot_score[0] += 1
            dict['top bot score'] = self.top_bot_score
        else:
            self.top_bot_score[1] += 1
            dict['top bot score'] = self.top_bot_score
        with open(f"{self.base}/outputs/{self.vid_name}/game_{g}_score_{self.last_score-1}/game_{g}_score_{self.last_score-1}_info.json",'w') as f:
            json.dump(dict, f, indent=2)

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Second cost: {round(time.time() - self.start_time, 1)}")
        print(f'Frame count:{self.frame_count}')
        print(f'Save count:{self.saved_count}')
        print(f'Score: {self.score}')

        with open('csv_records/pipeline_video_data.csv', 'a', newline='') as csvfile:
            fieldnames = ['vid_name', 'total_frame_count', 'total_saved_count', 'saved_count', 'score',
                          'execution_time(sec)']
            writer = csv.DictWriter(csvfile, fieldnames, delimiter=',', quotechar='"')
            writer.writerow({
                'vid_name': self.vid_name,
                'total_frame_count': self.total_frame_count,
                'total_saved_count': self.total_saved_count,
                'saved_count': self.saved_count,
                'score': self.score,
                'execution_time(sec)': round(time.time() - self.start_time, 1)
            })
        return True

paths = get_path('E:/test_videos')
vid_paths = []
for path in paths:
    if path.split('/')[-1].split('.')[-1] == 'mp4':
        vid_paths.append(path)
for vid_path in vid_paths:
    vpr = video_resolver(vid_path, output_base='E:/test_videos')    # output base is where "outputs" dir is
    vpr.resolve()