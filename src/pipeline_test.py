import copy
import os, csv, cv2, json, time, numpy as np, matplotlib, matplotlib.pyplot as plt
from PIL import Image
import torch, torchvision
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import scene_utils
import shot_recog
from scene_utils import scene_classifier


def check_dir(path):
    isExit = os.path.exists(path)
    if not isExit:
        os.mkdir(path)


def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
            pass
    return paths


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

        self.court_kp_model = torch.load('model_weights/court_kpRCNN.pth')
        self.court_kp_model.to(self.device).eval()

        self.paths = [f"{self.base}/outputs",
                      f"{self.base}/outputs/{self.vid_name}"]
        for path in self.paths:
            check_dir(path)

        self.cap = cv2.VideoCapture(vid_path)

        if not self.cap.isOpened():
            print('Error while trying to read video. Please check path again')

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.save_path = f"{self.base}/outputs/videos/{self.vid_name}/{self.vid_name}.mp4"
        self.frame_count = 1
        self.saved_count = 0
        self.time_rate = 0.1
        self.FPS = self.cap.get(5)
        self.frame_rate = round(int(self.FPS) * self.time_rate)
        self.total_frame_count = int(self.cap.get(7))
        self.total_saved_count = int(self.total_frame_count / self.frame_rate)
        # self.out = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), int(self.FPS/self.frame_rate),
        #                            (self.frame_width, self.frame_height))
        self.court_points = None
        self.true_court_points = None
        self.court_info = None
        self.joint_list = []
        self.wait_list = []
        self.last_type = 0
        self.checkpoint = False
        self.score = 0
        self.one_count = 0

    def get_court_info(self, img):
        with torch.no_grad():
            img = F.to_tensor(img)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            output = self.court_kp_model(img)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs],
                                            output[0]['scores'][high_scores_idxs],
                                            0.3).cpu().numpy()
        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])
        self.true_court_points = copy.deepcopy(keypoints[0])
        print(self.true_court_points)
        keypoints[0][0][0] -= 80
        keypoints[0][0][1] -= 80
        keypoints[0][1][0] += 80
        keypoints[0][1][1] -= 80
        keypoints[0][2][0] -= 80
        keypoints[0][3][0] += 80
        keypoints[0][4][0] -= 80
        keypoints[0][4][1] = min(keypoints[0][4][1] + 80, self.frame_height - 40)
        keypoints[0][5][0] += 80
        keypoints[0][5][1] = min(keypoints[0][5][1] + 80, self.frame_height - 40)
        self.court_points = keypoints[0]
        print(self.court_points)
        l_a = (self.court_points[0][1] - self.court_points[4][1]) / (
                    self.court_points[0][0] - self.court_points[4][0])
        l_b = self.court_points[0][1] - l_a * self.court_points[0][0]
        r_a = (self.court_points[1][1] - self.court_points[5][1]) / (
                    self.court_points[1][0] - self.court_points[5][0])
        r_b = self.court_points[1][1] - r_a * self.court_points[1][0]
        mp_y = (self.court_points[2][1] + self.court_points[3][1]) / 2
        self.court_info = [l_a, l_b, r_a, r_b, mp_y]
        return True

    def check_type(self, last_type):
        sum = 0
        if last_type == 1:
            for pair in self.wait_list:
                sum += pair[0]
            if sum <= 3:
                return True
            else:
                return False
        else:
            for pair in self.wait_list:
                sum += pair[0]
            if sum >= 3:
                return True
            else:
                return False

    def in_court(self, joint):
        l_a = self.court_info[0]
        l_b = self.court_info[1]
        r_a = self.court_info[2]
        r_b = self.court_info[3]
        mp_y = self.court_info[4]
        ankle_x = (joint[15][0] + joint[16][0]) / 2
        ankle_y = (joint[15][1] + joint[16][1]) / 2
        top = ankle_y > self.court_points[0][1]
        bottom = ankle_y < self.court_points[5][1]
        lmp_x = (ankle_y - l_b) / l_a
        rmp_x = (ankle_y - r_b) / r_a
        left = ankle_x > lmp_x
        right = ankle_x < rmp_x

        if left and right and top and bottom:
            return True
        else:
            return False

    def check_pos(self, indexes, boxes):
        court_mp = self.court_info[4]
        for i in range(len(indexes)):
            combination = 1
            if boxes[indexes[0]][1] < court_mp and boxes[indexes[combination]][3] > court_mp:
                return True, [0, combination]
            elif boxes[indexes[0]][3] > court_mp and boxes[indexes[combination]][1] < court_mp:
                return True, [0, combination]
            else:
                combination += 1
        return False, [0, 0]

    def score_rank(self, joints):
        indexes = []
        for i in range(len(joints)):
            if self.in_court(joints[i]):
                indexes.append(i)
        if len(indexes) < 2:
            return False
        else:
            return indexes

    def draw_keypoints(self, outputs, image):
        edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12), (5, 7),
                 (7, 9), (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16), (5, 6)]
        playerJoints = []
        b = outputs[0]['boxes'].cpu().detach().numpy()
        j = outputs[0]['keypoints'].cpu().detach().numpy()
        topScores = self.score_rank(j)
        if topScores == False:
            return image, True
        fit, combination = self.check_pos(topScores, b)
        if fit:
            for c in combination:
                i = topScores[c]
                keypoints = j[i]
                keypoints = keypoints[:, :].reshape(-1, 3)
                playerJoints.append(j[i].tolist())
                for p in range(keypoints.shape[0]):
                    cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1,
                               lineType=cv2.FILLED)
                    cv2.putText(image, str(i), (int(keypoints[15, 0]), int(keypoints[15, 1])), cv2.FONT_HERSHEY_DUPLEX,
                                1, (0, 255, 255), 1,
                                cv2.LINE_AA)
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
        joint_img_list = []
        temp_code = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_count % self.frame_rate == 0:
                    sceneImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    sceneImg = scene_utils.preprocess(sceneImg, self.device)
                    # slice video into score videos
                    p = scene_utils.predict(self.scene_model, sceneImg)
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if len(self.wait_list) < 5:
                        self.wait_list.append((p, pil_image, frame))
                    else:
                        tup = self.wait_list.pop(0)
                        self.wait_list.append((p, pil_image, frame))
                        p = tup[0]
                        pil_image = tup[1]
                        frame = tup[2]
                        if p != self.last_type:
                            correct = self.check_type(self.last_type)
                            if not correct:
                                if p == 1:
                                    p = 0
                                else:
                                    p = 1
                            else:
                                self.checkpoint = True
                                if p == 0:
                                    if len(self.joint_list) / self.one_count > 0.6 and self.one_count > 25:
                                        framesDict = {'frames': self.joint_list}
                                        sc_path = f"{self.base}/outputs/{self.vid_name}/score_{self.score}"
                                        check_dir(sc_path)
                                        b = f"{self.base}/outputs/{self.vid_name}/score_{self.score}/"
                                        out = cv2.VideoWriter(f"{b}score_{self.score}.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                                              int(self.FPS / self.frame_rate), (self.frame_width, self.frame_height))
                                        for img in joint_img_list:
                                            out.write(img)
                                        joint_img_list = []
                                        out.release()

                                        save_path = f"{b}score_{self.score}.json"
                                        with open(save_path, 'w') as f:
                                            json.dump(framesDict, f, indent=2)
                                        self.joint_list = []
                                        self.score += 1
                                        self.one_count = 0
                                        input, score_joint_list = shot_recog.get_data(save_path)

                                        shot_list, pos_percentage = shot_recog.check_hit_frame(temp_code, score_joint_list, self.true_court_points)
                                        print(shot_list, pos_percentage)
                                        success = shot_recog.add_result(b, f"{b}score_{self.score-1}.mp4", shot_list, self.true_court_points)
                                        if success:
                                            print(f'Finish score_{self.score}')
                                        # input 給 model 輸出 d
                                        # d =
                                    else:
                                        self.joint_list = []
                                        self.one_count = 0
                                else:
                                    if self.court_points == None:
                                        _ = self.get_court_info(img=self.wait_list[2][2])
                                        print("Get!")
                                    # self.one_count += 1
                                self.last_type = p
                        else:
                            self.checkpoint = False
                        if p == 1:
                            self.one_count += 1
                            image = self.transform(pil_image)
                            image = image.unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                outputs = self.model(image)
                            output_image, player_joints = self.draw_keypoints(outputs, frame)
                            if player_joints != True:
                                for points in player_joints:
                                    for i, joints in enumerate(points):
                                        points[i] = joints[0:2]
                                self.joint_list.append({
                                    'joint': player_joints,
                                })
                            # add features
                            for kps in [self.court_points]:
                                for idx, kp in enumerate(kps):
                                    cv2.circle(output_image, tuple(kp), 5, (255, 255, 0), 10)
                            # text = f"Frame count: {self.saved_count}, Court: True, Checkpoint: {self.checkpoint}, Score: {self.score}"
                            # cv2.putText(output_image, text, (700, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1,
                            #             cv2.LINE_AA)
                            joint_img_list.append(output_image)

                        self.saved_count += 1
                        print(self.saved_count, ' / ', self.total_saved_count)
                        self.frame_count += 1
                else:
                    self.frame_count += 1
            else:
                break

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

vpr = video_resolver('E:/test_videos/p_test.mp4', output_base='E:/test_videos')    # output base is where "outputs" dir is
vpr.resolve()
