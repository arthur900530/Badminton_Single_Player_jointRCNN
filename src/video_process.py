import cv2, json, numpy as np, matplotlib, matplotlib.pyplot as plt
from preprocess import get_path, check_dir
from PIL import Image
import torch, torchvision
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import scene_utils
from scene_utils import scene_classifier


class video_processor:
    def __init__(self, vid_path, output_base='..'):
        self.base = output_base
        self.vid_path = vid_path
        self.vid_name = vid_path.split('/')[-1].split('.')[0]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
        self.model.to(self.device).eval()
        self.scene_model = scene_utils.build_model('scene_classifier.pt', self.device)
        self.court_kp_model = torch.load('court_kpRCNN.pth')
        self.court_kp_model.to(self.device).eval()
        self.paths = [f"{self.base}/outputs",
                      f"{self.base}/outputs/videos",
                      f"{self.base}/outputs/joint_data",
                      f"{self.base}/outputs/videos/{self.vid_name}",
                      f"{self.base}/outputs/joint_data/{self.vid_name}"]
        for path in self.paths:
            check_dir(path)
        self.cap = cv2.VideoCapture(vid_path)
        if not self.cap.isOpened():
            print('Error while trying to read video. Please check path again')
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.save_path = f"{self.base}/outputs/videos/{self.vid_name}/{self.vid_name}.mp4"
        self.out = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (self.frame_width, self.frame_height))
        self.frame_count = 1
        self.save_count = 0
        self.time_rate = 0.1
        self.FPS = self.cap.get(5)
        self.frame_rate = int(int(self.FPS) * self.time_rate)
        self.total_frame_count = int(self.cap.get(7))
        self.total_save_count = int(self.total_frame_count / self.frame_rate)
        self.court_points = None
        self.court_info = None
        self.joint_list = []
        self.wait_list = []
        self.last_type = 0
        self.checkpoint = False
        self.score = 0
        self.one_count = 0

    # [[590, 434, 1], [1310, 434, 1],
    #  [476, 624, 1], [1427, 623, 1],
    #  [256, 1000, 1], [1660, 1002, 1]]

    def get_court_info(self, img):
        with torch.no_grad():
            img = F.to_tensor(img)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            output = self.court_kp_model(img)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])
        self.court_points = keypoints[0]
        l_a = (self.court_points[0][1] - self.court_points[4][1]) / (self.court_points[0][0] - self.court_points[4][0])
        l_b = self.court_points[0][1] - l_a * self.court_points[0][0]
        r_a = (self.court_points[1][1] - self.court_points[5][1]) / (self.court_points[1][0] - self.court_points[5][0])
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


    def process(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_count % self.frame_rate == 0:
                    sceneImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    sceneImg = scene_utils.preprocess(sceneImg, self.device)
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
                                        save_path = f"{self.base}/outputs/joint_data/{self.vid_name}/{self.vid_name}-score_{self.score}.json"
                                        with open(save_path, 'w') as f:
                                            json.dump(framesDict, f, indent=2)
                                        self.joint_list = []
                                        self.score += 1
                                        self.one_count = 0
                                    else:
                                        self.joint_list = []
                                        self.one_count = 0
                                else:
                                    if self.court_points == None:
                                        _ = self.get_court_info(img=frame)
                                        print("Get!")
                                    self.one_count += 1
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
                                    'frame': self.save_count,
                                    'joint': player_joints,
                                    'label': -1,
                                    'type': 'TYPE'
                                })
                            text = f"Frame count: {self.save_count}, Court: True, Checkpoint: {self.checkpoint}, Score: {self.score}"
                            cv2.putText(output_image, text, (700, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                            self.out.write(output_image)
                        else:
                            text = f"Frame count: {self.save_count}, Court: False, Checkpoint: {self.checkpoint}, Score: {self.score}"
                            cv2.putText(frame, text, (700, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                            self.out.write(frame)
                        self.save_count += 1
                        print(self.save_count, ' / ', self.total_save_count)
                        self.frame_count += 1
                else:
                    self.frame_count += 1
            else:
                break
        # clear wait list
        for i in range(len(self.wait_list)):
            frame = self.wait_list[i][2]
            if self.frame_count % self.frame_rate == 0:
                self.save_count += 1
                text = f"Frame count: {self.save_count}, Court: False, Checkpoint: {self.checkpoint}, Score: {self.score}"
                cv2.putText(frame, text, (700, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                self.out.write(frame)
                print(self.save_count, ' / ', self.total_save_count, self.frame_count)
                self.frame_count += 1
            else:
                self.frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()
        print(f'Frame count:{self.frame_count}')
        print(f'Save count:{self.save_count}')

        return True

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

    def in_court(self, joint):
        l_a = self.court_info[0]
        l_b = self.court_info[1]
        r_a = self.court_info[2]
        r_b = self.court_info[3]
        mp_y = self.court_info[4]
        ankle_x = (joint[15][0] + joint[16][0]) / 2
        ankle_y = (joint[15][1] + joint[16][1]) / 2
        top = ankle_y > self.court_points[0][1] - 80
        bottom = ankle_y < self.court_points[5][1] + 20
        lmp_x = (ankle_y - l_b) / l_a
        rmp_x = (ankle_y - r_b) / r_a
        left = ankle_x > lmp_x
        right = ankle_x < rmp_x

        if left and right and top and bottom:
            return True
        else:
            return False

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
