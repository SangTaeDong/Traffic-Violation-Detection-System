from collections import deque
import sys
import cv2
import os
import glob
import numpy as np
import copy
from python.lane_detection.module import NIA_SEGNet_module
from python.lanedet_220822 import _lanedetect
from python.stop_line import _stopline
from python.yoloVehicle_221013 import yoloVehicle
from python.yoloTrafficLight_220916 import yoloTrafficLight
from python.turn_signal import turn_signal
from python.FreeSpace import draw_linear_graph, intersection_point, bottom_min, left_right
from python.Slide_Window_Search import slide_window_search
from python.cover_not_interest import cover_not_interest
from python.lanetype import lane_state
import time


class carViolation:

    '''
    클래스 변수들 정리

    self.video_list         동영상 파일이름, 디렉토리 저장된 배열
    self.video_N            비디오 길이
    self.video_size         동영상의 h//2,w//2 저장(h,w)
    self.fps                동영상의 fps 정보

    self.queue_size         원형큐의 사이즈
    self.video_queue        원형큐

    self.video_cnt          영상 순서대로 불러오기위해 cnt 용도
    self.cap                self.cap.read() 사용 용도
    self.temp_dict          queue에 집어넣기 전에 임시저장용

    self.yolo_model         yolo_model
    self.yolo_class         yolo_model_class

    self.yolo_trafficlight_model 신호등용
    self.yolo_trafficlight_class

    self.lane_model         lane_model

    self.lower_white_hls    색 범위(lower)
    self.upper_white_hls    색 범위(upper)

    TODO
    self.lower_yellow_hls
    self.upper_yellow_hls

    self.pts1               원래 이미지에서 bird eyes view로 변환할 좌표
    self.pts2               bird eyes view 좌표

    self.frame_index        frame_index

    self.mode              모드설정
    self.lb                left base
    self.rb                right base  ## lane Area 

    self.car_counter       위법차량 중복검출 방지용
    '''
    mode = 'default'

    def __init__(self, video_path, yolo_weight_path, yolo_class_name_path, yolo_cfg_path, yolo_trafficlight_weight_path, yolo_trafficlight_class_name_path, yolo_trafficlight_cfg_path, ckpt_path, queue_size=100, cuda=False):        # TODO CUDA 사용?
        self.cuda = cuda

        # TODO yolo model 가져오기
        self.yolo_init(yolo_weight_path, yolo_class_name_path, yolo_cfg_path,
                       yolo_trafficlight_weight_path, yolo_trafficlight_class_name_path, yolo_trafficlight_cfg_path)

        # TODO Lane model 가져오기
        self.Lane_model_init(ckpt_path)

        # 영상 리스트 가져오기
        self.video_init(video_path)

        # 큐 초기화
        self.video_queue = deque()
        self.queue_size = queue_size

        # time
        self.prevTime = time.time()
        self.curTime = time.time()

        # Color Range Custome
        frame, detect = self.color_range_frame(frame_size=10)
        self.color_range_init(frame, detect)

        # 삭제요망
        self.lower_white_hls = np.array([0, 180, 0])
        # TODO Bird Eyes View Area Custome
        while True:
            if self.Bird_eyes_view_init(frame, detect) == True:
                break
            frame, detect = self.color_range_frame(frame_size=10)

        self.performance_evaluation()
        self.frame_load(isFirst=True)

        self.dataDuplication = deque([])

    def video_init(self, video_path):
        '''
        video_path를 입력받아 그 dir 에 있는 동영상 파일들을 골라내
        self.video_list에 넣어줌
        '''

        # file_list 에 video_path 내부의 파일 디렉토리를 다 넣어준다
        file_list = glob.glob(video_path+'/*')
        # mp4,avi 확장자를 가진 파일들만 가져와 self.video_list에 넣어준다
        self.video_list = [file for file in file_list if file.endswith(
            ".avi") or file.endswith(".mp4") or file.endswith(".AVI")]
        # 비디오 갯수
        self.video_N = len(self.video_list)
        # 순서대로 영상 불러오기 위해 sort 한번 해줌
        self.video_list.sort()
        # 비디오 cnt = 0 으로 0번쨰 인댁스 영상부터 틀기 위해 미리 초기화
        self.video_cnt = 0

        cap = cv2.VideoCapture(self.video_list[self.video_cnt])
        state, frame = cap.read()
        h, w = frame.shape[:2]
        self.video_size = (h//2, w//2)
        self.pts2 = np.float32([(0, 0), (350, 0), (350, 350), (0, 350)])
        self.car_counter = set()
        self.cnt = 10
        self.Lane_Change = False
        print(self.video_N, 'video load')

    def yolo_init(self, yolo_weight_path, yolo_class_name_path, yolo_cfg_path, yolo_trafficlight_weight_path, yolo_trafficlight_class_name_path, yolo_trafficlight_cfg_path):
        '''
        yolo class_name, weight, cfg path 입력받아서
        self.yolo_model 변수에 yolo model 넣어줌
        '''

        self.yolo_cnt = 0

        f = open(yolo_trafficlight_class_name_path, 'r')
        self.yolo_trafficlight_class = [line.strip() for line in f.readlines()]
        self.yolo_trafficlight_model = cv2.dnn.readNet(
            yolo_trafficlight_weight_path, yolo_trafficlight_cfg_path)
        print('yolo traffic model load success')

        f = open(yolo_class_name_path, 'r')
        self.yolo_class = [line.strip() for line in f.readlines()]
        self.yolo_model = cv2.dnn.readNet(yolo_weight_path, yolo_cfg_path)
        print('yolo model load success')

    def Lane_model_init(self, ckpt_path):
        '''
        Lane_model CKPT 가져와서 모델 Road
        이미지 한개씩 넣을것이기 때문에 batch_size = 1로 설정
        '''
        self.lane_model = NIA_SEGNet_module.load_from_checkpoint(
            checkpoint_path=ckpt_path)
        self.lane_model.batch_size = 1
        print('Lane model load success')

    def color_range_frame(self, frame_size=5):
        '''
        상황에 따라서 차선의 유무가 다르기 때문에
        영상을 랜덤으로 size(10)개만큼 뽑아와서
        model detect 후 return 해줌
        '''
        random_int = np.random.choice(
            self.video_N, frame_size, replace=False)  # replace => 중복 안나오게

        # 랜덤으로 뽑힌 인덱스들을 넣어주며 영상에 프레임 가져와 차선 검출 후
        # frame_arr, detect_arr에 각각 넣어준 후 return
        frame_arr = []
        detect_arr = []

        for idx in random_int:
            temp_cap = cv2.VideoCapture(self.video_list[idx])
            state, frame = temp_cap.read()
            w, h = frame.shape[:2]
            frame = cv2.resize(frame, dsize=(h//2, w//2))
            frame_arr.append(frame)
            detect_arr.append(self.lane_model.predict(frame))

        h, w = frame_arr[0].shape[:2]
        self.video_size = (h//2, w//2)
        return frame_arr, detect_arr

    def color_range_init(self, real_img, lane_model_img):
        '''
        원래 이미지와 레인 모델 이미지를 입력받아서
        차선 및 정지선의 색 범위를 구함
        '''
        # 랜덤으로 원래 이미지와 레인 모델 이미지 각각 10개 불러오기
        real_img, lane_model_img = self.color_range_frame()

        l_max_list = []
        list_len = len(real_img)
        for i in range(list_len):

            # 차선과 정지선 보이도록 threshold 해줌
            _, lane_model_IMG = cv2.threshold(
                lane_model_img[i], 2, 255, cv2.THRESH_TOZERO_INV)
            cv2.threshold(lane_model_IMG, 0, 255,
                          cv2.THRESH_BINARY, lane_model_IMG)

            # 잡음 제거
            mask = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]]).astype('uint8')
            lane_model_IMG = cv2.erode(lane_model_IMG, mask, iterations=3)


#             cv2.imwrite(str(i)+'real_img.png',real_img[i])
#             cv2.imwrite(str(i)+'lane_model_img.png',lane_model_IMG)

            # 모델에서 불러온 이미지와 원래 이미지를 bitwise and 해주고
            # 색 영역을 hls영역으로 변환해준 후 l의 히스토그램을 리스트로 받고
            # l의 최대를 구하여 l_max_list에 넣어준다.

            result = cv2.bitwise_and(real_img[i], lane_model_IMG)

            hls_img = cv2.cvtColor(result, cv2.COLOR_BGR2HLS)
            l_hist = cv2.calcHist([hls_img], [1], None, [255], [0, 255])

            l_max_list.append(np.argmax(l_hist[1:]))

        l_avg = sum(l_max_list)//list_len

        if l_avg > 180:
            l_tmp = abs(l_avg-180)//10
            self.lower_white_hls = np.array([0, 180+l_tmp, 0])
            self.upper_white_hls = np.array([180, 255, 255])

        else:
            self.lower_white_hls = np.array([0, 180, 0])
            self.upper_white_hls = np.array([180, 255, 255])

        print('Color Range Custom success')

    def frame_load(self, isFirst=False):
        '''
        프레임 불러오기를 위한함수
        초기화할떄는 isFirst = True 로 바꿔주고 사용해야함.

        '''

        if isFirst:
            # 처음 프레임 불러올경우 큐 크기만큼 가져옴
            self.cap = cv2.VideoCapture(self.video_list[self.video_cnt])
            self.frame_index = -1
            # self.video_size 저장해줌
            # 첫프레임 하나는 버리게됨
            # fps저장
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            state, frame = self.cap.read()
            h, w = frame.shape[:2]
            self.video_size = h//2, w//2

            for _ in range(self.queue_size):
                state, frame = self.cap.read()

                self.frame_index += 1
                # 큐에넣을 다양한 정보들(차량,차선,정지선,신호등 정보) 넣어줌
                if state == False:
                    self.state_false()
                    continue

                self.frame_dict_init(frame)

        else:
            # 프레임 불러오는게 처음이 아닐경우 큐에서 하나 제거하고 하나 추가해줌
            self.video_queue.popleft()
            state, frame = self.cap.read()
            self.frame_index += 1

            if state == False:
                self.state_false()
            else:
                self.frame_dict_init(frame)

    def state_false(self):
        '''
        cap.read()했을떄 State 가 False일 경우 사용하는 함수
        self.video_cnt 에 +1 을 해준다음
        cv2.VideoCapture을 사용해 다음 영상을 불러온뒤
        한 프레임만 큐에 넣어준다 <= frame_load 함수에서 state_false 사용시 continue 하게 되어있는데
        한 프레임이 저장이 안되었을것이기 때문에 한프레임 넣어줌

        만약 더이상 불러올 영상이 없을 경우
        '''
        self.video_cnt += 1

        # 더이상 영상이 없을 경우
        if self.video_cnt == self.video_N:
            print('모든 영상을 확인하였습니다.', '종료합니다', sep='\n')
            sys.exit()

            # 영상이 있을경우
        else:

            self.cap = cv2.VideoCapture(self.video_list[self.video_cnt])
            self.frame_index = 1
            state, frame = self.cap.read()
            self.frame_index += 1
            # 한프레임 넣어줌
            self.frame_dict_init(frame)

    def frame_dict_init(self, frame):
        '''
        frame을 받아와서
        딕셔너리에 넣을 정보들 (차량정보, 차선정보, 정지선정보, 신호등 정보) 딕셔너리에 넣어준뒤
        queue에 append 해줌
        '''
        (h, w) = self.video_size
        resize_frame = cv2.resize(frame, dsize=(w, h))
        self.temp_dict = {'frame': frame}

#         self.curTime = time.time()
#         fps = 1/(self.curTime - self.prevTime)
#         self.prevTime = self.curTime
#         print(fps)

        # 첫 프레임의 경우 초기화해주는 과정.
        if self.frame_index == 0:
            first_Check = True
            vehicle_dict = {}
        # 첫프레임이 아닐경우 이전의 프레임으로 초기화해주는 과정
        else:
            first_Check = False
            vehicle_dict = copy.deepcopy(self.video_queue[-1]['vehicle'])

        # 10프레임마다 한번씩 Yolo를 사용해 Object Detect 진행
        if self.cnt == 10:
            # Yolo Model
            cover_frame = cover_not_interest(
                resize_frame, first_Check, self.video_queue)
            self.temp_dict['vehicle'], hide_img, self.yolo_cnt = yoloVehicle(
                cover_frame, vehicle_dict, self.yolo_class, self.yolo_model, self.yolo_cnt, first_Check)
            self.temp_dict['traffic light'] = yoloTrafficLight(
                frame, self.yolo_trafficlight_class, self.yolo_trafficlight_model)
            self.multiTracker = cv2.legacy.MultiTracker_create()

            # 멀티 트레커 생성
            for key in self.temp_dict['vehicle'].keys():
                x, y, w, h = self.temp_dict['vehicle'][key]['pos']
                self.multiTracker.add(
                    cv2.legacy.TrackerMedianFlow_create(), resize_frame, (x, y, w, h))

            lane, stop_line = self.highAccCheck2(resize_frame)
            self.temp_dict['lane'] = copy.deepcopy(lane)
            self.temp_dict['stopline'] = copy.deepcopy(stop_line)
            self.cnt = 1
        else:
            self.cnt += 1

            # 멀티 트레커 업데이트
            success, boxes = self.multiTracker.update(resize_frame)
            vehicle_dict = copy.deepcopy(self.video_queue[-1]['vehicle'])
            hide_img = resize_frame.copy()

            # 좌표 매칭시켜주기
            for key, Pos in zip(vehicle_dict.keys(), boxes):
                vehicle_dict[key]['pos'] = list(map(int, Pos))
                x, y, w, h = vehicle_dict[key]['pos']
                cv2.rectangle(hide_img, (x, y), (x+w, x+y),
                              (0, 0, 0), cv2.FILLED)
            self.temp_dict['vehicle'] = copy.deepcopy(vehicle_dict)

            # 신호등 상태 가져오기
            self.temp_dict['traffic light'] = self.video_queue[-1]['traffic light']

            # 정지선 정보 가져오기'
            hide_img, stop_line = _stopline(
                hide_img, self.temp_dict['vehicle'], self.pts1, self.pts2, self.lower_white_hls, self.upper_white_hls, self.cuda, hide_img, self.video_queue)
            self.temp_dict['stopline'] = copy.deepcopy(stop_line)

            # 차선정보 가져오기
    #         lane_info,self.lb,self.rb = _lanedetect(hide_img,self.pts1,self.pts2,self.lower_white_hls,self.upper_white_hls,first_Check,self.video_queue,self.lb,self.rb,self.miss_point)
            lane = _lanedetect(hide_img, self.pts1, self.pts2,
                               self.lower_white_hls, self.upper_white_hls, self.video_queue)
            self.temp_dict['lane'] = copy.deepcopy(lane)
        # 차선이 올바르게 가져와지지 않았다면 모델을 이용해서 Detect 해준다.
#         if state == False:
#             lane, stop_line =self.highAccCheck2(resize_frame)
        # 차량 방향지시등 상태 가져오기

        turn_signal(resize_frame, self.temp_dict['vehicle'], self.video_queue)

        # 영상정보 저장
        self.temp_dict['Video_name'] = self.video_list[self.video_cnt]
        self.temp_dict['Video_frame'] = self.frame_index

        # 차량 상태 업데이트
        self.Car_State_Update(self.temp_dict)
        # 위반 사항 검출
        if self.viloation_check(self.temp_dict):
            cover_frame = cover_not_interest(
                resize_frame, first_Check, self.video_queue)
            self.temp_dict['vehicle'], hide_img, self.yolo_cnt = yoloVehicle(
                cover_frame, vehicle_dict, self.yolo_class, self.yolo_model, self.yolo_cnt, first_Check)
            self.temp_dict['traffic light'] = yoloTrafficLight(
                frame, self.yolo_trafficlight_class, self.yolo_trafficlight_model)
            turn_signal(
                resize_frame, self.temp_dict['vehicle'], self.video_queue)
            self.Last_Check(self.temp_dict['frame'], self.temp_dict)
            # 영상 저장
            save_video(self)

        # queue에 append
        self.video_queue.append(copy.deepcopy(self.temp_dict))

        '''
        이미지를 저장해 분석하려면 이 주석을 사용하세요
        
        if not os.path.exists('img/'+self.temp_dict['Video_name'].split('/')[-1]):
            os.mkdir('img/'+self.temp_dict['Video_name'].split('/')[-1])
        try:
            cv2.imwrite('img/'+self.temp_dict['Video_name'].split('/')[-1]+'/'+str(self.temp_dict['Video_frame'])+'.jpg',self.visyalize(-1),[cv2.IMWRITE_JPEG_QUALITY,30 ])
        except:
            pass
        
        '''

    def save_video(self):
        '''
        현재 큐 안에 들어있는 frame들을 영상으로 변환하여 저장해준다
        '''
        h, w = self.video_size
        out = cv2.VideoWriter('output'+self.temp_dict['Video_name'].split('/')[-1]+'/'+str(
            self.temp_dict['Video_frame'])+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (w*2, h*2))
        for video in self.video_queue:
            out.write(cv2.resize(video['frame'].astype(
                'uint8'), dsize=(w*2, h*2)))

        out.release()
        print('Video Save success')

    def Bird_eyes_view_init(self, frame, detect_arr):
        '''
        차선을 검출하기 위한 용도로 Bird Eyes View의 사각형 범위를 Custom 해준다

        '''
    # 직선 구해주기

        h, w = self.video_size
        w *= 2
        h *= 2
        bottom = 0
        top = 9999999
        top = []
        for img in detect_arr:
            _, lane_model_IMG = cv2.threshold(
                img, 2, 255, cv2.THRESH_TOZERO_INV)
            cv2.threshold(lane_model_IMG, 0, 255,
                          cv2.THRESH_BINARY, lane_model_IMG)

            dst = lane_model_IMG.copy()
            mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype('uint8')
            dst = cv2.erode(dst, mask, iterations=3)
            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            lines = cv2.HoughLinesP(
                gray, 1.0, np.pi / 180., 90, minLineLength=200//2, maxLineGap=1)

            est_line = []
            try:
                if len(lines) <= 1:
                    continue
                for line in lines:
                    x1, y1, x2, y2 = line[0].astype(np.int32)

                    if x1-x2 == 0 or ((abs((y1 - y2) / (x1 - x2)) < 0.2) | (abs((y1 - y2) / (x1 - x2)) > 20)):
                        continue
                    else:
                        cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        est_line.append([x1, y1, x2, y2])

                    # 직선의 방정식 구해주기
            except:
                continue
            straight = []
            for i in est_line:
                straight.append(draw_linear_graph(i[0], i[1], i[2], i[3]))

            N = len(straight)
            arr = np.random.choice(N, N, replace=False)

            # 이상한 좌표 뺴줘서 소실점 구해주기
            x_list = []
            y_list = []
            for idx in range(0, N-1, 2):
                (x, y) = intersection_point(
                    straight[arr[idx]], straight[arr[idx+1]])
                if 0 < x < w and 0 < y < h:
                    x_list.append(x)
                    y_list.append(y)

            if len(x_list) == 0 or len(y_list) == 0:
                continue

            miss_point = x_list[len(x_list)//2], y_list[len(y_list)//2]
            # free_space top
            top.append(int(miss_point[1]))
            # Free_sapce bottom
            if bottom < bottom_min(est_line):
                bottom = bottom_min(est_line)

        top.sort()
        top = top[len(top)//2]
        sub = bottom-top
        self.pts1 = np.float32(
            ([sub, top], [960-sub, top], [960, bottom], [0, bottom]))

        self.miss_point = [
            (self.pts1[0][0]+self.pts1[1][0])//2, self.pts1[0][1]]

        print('Bird Eyes View Custom success')
        return True

    def Car_State_Update(self, temp_dict):
        car_info = temp_dict['vehicle']
        left_lane_info = temp_dict['lane']['left']
        right_lane_info = temp_dict['lane']['right']
        stopline_info = temp_dict['stopline']
        traffic_light = temp_dict['traffic light'].split(' ')[0]

        for info in car_info.keys():

            x, y, w, h = car_info[info]['pos']
            bottom = (x, y+h, x+w, y+h)
            left = (x, y, x, y+h)
            right = (x+w, y, x+w, y+h)

            # 1
            if stopline_info != None and 280 < (stopline_info[1]+stopline_info[3])//2 and (stopline_info[1]+stopline_info[3])//2 + 20 > y+h:
                temp_dict['vehicle'][info]['state'] = 10
                continue

            if self.Lane_Change == True:
                temp_dict['vehicle'][info]['state'] = 0
                continue

            '''
            차량의 아랫부분 직선
            [x,y+h], [x+w,y+h]
            
            왼쪽부분
            [x,y], [x, y+h]
            
            오른쪽
            [x+w,y], [x+w, y+h]
            
            
            '''
            if len(left_lane_info['y']) >= 1:
                lm, ln = draw_linear_graph(
                    left_lane_info['x'][0], left_lane_info['y'][0], left_lane_info['x'][1], left_lane_info['y'][1])
                if lm == 0:
                    temp_dict['vehicle'][info]['state'] = 0
                    continue
            else:
                temp_dict['vehicle'][info]['state'] = 0
                continue

            if len(right_lane_info['y']) >= 1:
                rm, rn = draw_linear_graph(
                    right_lane_info['x'][0], right_lane_info['y'][0], right_lane_info['x'][1], right_lane_info['y'][1])
                if rm == 0:
                    temp_dict['vehicle'][info]['state'] = 0
                    continue
            else:
                temp_dict['vehicle'][info]['state'] = 0
                continue

            # 중심점의 x좌표 위치가 왼쪽 차선보다 왼쪽일경우
            if (y+h-ln)/lm > (x+w//2):
                temp_dict['vehicle'][info]['state'] = 20
                continue

            if (y+h-rn)/rm < (x+w//2):
                temp_dict['vehicle'][info]['state'] = 60
                continue

            temp_dict['vehicle'][info]['state'] = 40

        self.Lane_Change = False

    def viloation_check(self, temp_dict):

        if len(self.video_queue) < 10:
            return False

        viloation = {'lane': {'left': None, 'right': None},
                     'traffic light': None}

        left = {}
        right = {}
        trafficLight = {}

        for state in [self.video_queue[-1], self.video_queue[-2], self.video_queue[-3], self.video_queue[-4], self.video_queue[-5], self.video_queue[-6], self.video_queue[-7], self.video_queue[-8], self.video_queue[-9], self.video_queue[-10]]:

            if state['lane']['left']['state'] in left:
                left[state['lane']['left']['state']] += 1
            else:
                left[state['lane']['left']['state']] = 1

            if state['lane']['right']['state'] in right:
                right[state['lane']['right']['state']] += 1
            else:
                right[state['lane']['right']['state']] = 1

            if state['traffic light'] in trafficLight:
                trafficLight[state['traffic light']] += 1
            else:
                trafficLight[state['traffic light']] = 1
        left = list(left.items())
        left.sort(key=lambda x: -x[1])
        right = list(right.items())
        right.sort(key=lambda x: -x[1])
        trafficLight = list(trafficLight.items())
        trafficLight.sort(key=lambda x: -x[1])
        viloation['lane']['left'] = left[0][0]
        viloation['lane']['right'] = right[0][0]
        viloation['traffic light'] = trafficLight[0][0]

        stopline = temp_dict['stopline']
        vehicle = temp_dict['vehicle']
        traffic_light = temp_dict['traffic light'].split(' ')[0]
        viloationBefore = self.video_queue[-1]['vehicle']

        for vehicleInfo in temp_dict['vehicle'].items():

            # 위법 중복 검출 방지
            if vehicleInfo[0] in self.car_counter:
                continue
            violation = ''

            carNum = vehicleInfo[0]
            carPos = vehicleInfo[1]['pos']
            carType = vehicleInfo[1]['type']
            carState = vehicleInfo[1]['state']
            carSignal = vehicleInfo[1]['Turn_Signal']
            # 한프레임으로 검출가능한 것

            # 정지선위반
            if viloation['traffic light'] == 'TrafficLight_Red' and stopline:
                if (viloation['lane']['left'] == 'Yellow_Solid' or viloation['lane']['right'] == 'Yellow_Solid') and carState == 40:
                    pass
                else:
                    x, y, w, h = carPos
                    if stopline != None and traffic_light == 'TrafficLight_Red' and 250 < (stopline[1]+stopline[3])//2:
                        if stop_line_volation(*stopline, x, y, x+w, y+h, carType, carState):
                            violation += '1'
                # 신호위반

                # 이전프레임 봐야 검출 가능한 것

                # 이전에 차량이 있었을 경우
            if carNum in viloationBefore:
                # 진로변경위반
                if viloationBefore[carNum]['state'] == 20:
                    if carState == 40 and (viloation['lane']['left'] == 'White_Solid' or viloation['lane']['left'] == 'Yellow_Solid'):
                        violation += '4'

                if viloationBefore[carNum]['state'] == 40:
                    if carState == 20 and (viloation['lane']['left'] == 'White_Solid' or viloation['lane']['left'] == 'Yellow_Solid'):
                        violation += '4'
                    elif carState == 60 and (viloation['lane']['right'] == 'White_Solid'):
                        violation += '4'

                if viloationBefore[carNum]['state'] == 60:
                    if carState == 40 and (viloation['lane']['right'] == 'White_Solid'):
                        violation += '4'

                    # 제차조작신호불이행
#                 if viloationBefore[carNum]['state'] == 20 :
#                     if (carState == 40) and (carSignal== 1 or carSignal == 2):
#                         self.car_counter[carNum] = 200
#                         violation +='5'

#                 if viloationBefore[carNum]['state'] == 40:
#                     if (carState == 20) and (carSignal== 1 or carSignal == 3):
#                         self.car_counter[carNum] = 200
#                         violation +='5'
#                     if (carState == 60) and (carSignal== 1 or carSignal == 2):
#                         self.car_counter[carNum] = 200
#                         violation +='5'

#                 if viloationBefore[carNum]['state'] == 60:
#                     if (carState == 40) and (carSignal== 1 or carSignal == 3):
#                         self.car_counter[carNum] = 200
#                         violation +='5'

                    # 중앙선 침범
                if viloationBefore[carNum]['state'] == 20 and carState == 40 and (viloation['lane']['left'] == 'Yellow_Solid'):
                    violation += '3'
                elif viloationBefore[carNum]['state'] == 40 and carState == 20 and (viloation['lane']['left'] == 'Yellow_Solid'):
                    violation += '3'

                temp = []

                if violation != '':

                    return True
                else:
                    return False

    def visyalize(self, num):

        dst = self.video_queue[num]['frame'].copy()

        (w, h) = self.video_size
        dst = cv2.resize(dst, dsize=(h, w))

        # Bird Eyes View Area
        dst = cv2.line(dst, (int(self.pts1[3][0]), int(self.pts1[3][1])), (int(
            self.pts1[2][0]), int(self.pts1[2][1])), (0, 0, 255), 3, cv2.LINE_AA)
        dst = cv2.line(dst, (int(self.pts1[1][0]), int(self.pts1[1][1])), (int(
            self.pts1[0][0]), int(self.pts1[0][1])), (0, 0, 255), 3, cv2.LINE_AA)
        dst = cv2.line(dst, (int(self.pts1[3][0]), int(self.pts1[3][1])), (int(
            self.pts1[0][0]), int(self.pts1[0][1])), (0, 0, 255), 3, cv2.LINE_AA)
        dst = cv2.line(dst, (int(self.pts1[1][0]), int(self.pts1[1][1])), (int(
            self.pts1[2][0]), int(self.pts1[2][1])), (0, 0, 255), 3, cv2.LINE_AA)

        # lane

        if len(self.video_queue[num]['lane']['left']['x']) != 0:
            left_x_list = self.video_queue[num]['lane']['left']['x']
            left_y_list = self.video_queue[num]['lane']['left']['y']
            pts_left = np.array(
                [np.transpose(np.vstack([left_x_list, left_y_list]))])
            cv2.polylines(dst, [np.int32(pts_left)], False,
                          (0, 255, 0), 8, cv2.LINE_AA)

        if len(self.video_queue[num]['lane']['right']['x']) != 0:
            right_x_list = self.video_queue[num]['lane']['right']['x']
            right_y_list = self.video_queue[num]['lane']['right']['y']
            pts_right = np.array(
                [np.transpose(np.vstack([right_x_list, right_y_list]))])
            cv2.polylines(dst, [np.int32(pts_right)],
                          False, (0, 255, 0), 8, cv2.LINE_AA)

        # Vehicle

        for car_number in self.video_queue[num]['vehicle'].keys():
            x, y, w, h = self.video_queue[num]['vehicle'][car_number]['pos']
            cv2.rectangle(dst, (x, y, w, h), (255, 0, 255), 3)
            cv2.putText(dst, str(car_number)+' '+str(self.video_queue[num]['vehicle'][car_number]['state']), (
                x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # stop lane
        s_dict = self.video_queue[num]['stopline']
        if s_dict != None:
            x1 = int(s_dict[0])
            y1 = int(s_dict[1])
            x2 = int(s_dict[2])
            y2 = int(s_dict[3])

            cv2.line(dst, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)

        return dst

    def performance_evaluation(self):
        import csv
        self.mode = 'eval'
        if os.path.exists('result.csv'):
            print('이미 result.csv가 존재합니다.\n다시 실행하실려면 파일을 삭제하세요.')
            exit()

        csv_file = open('result.csv', 'w', newline='')
        self.wr = csv.writer(csv_file)
        self.wr.writerow(['video_name', 'frame', 'case', 'car_num'])

    def find_base(self, image):
        perspect_mat = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        dst = cv2.warpPerspective(
            image, perspect_mat, (350, 350), cv2.INTER_CUBIC)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        histogram = np.sum(dst[:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)

        self.lb = np.argmax(histogram[:midpoint])
        self.rb = np.argmax(histogram[midpoint:]) + midpoint

    def highAccCheck2(self, frame):
        high_dict = {'left': {'x': [], 'y': [], 'state': None},
                     'right': {'x': [], 'y': [], 'state': None}}
        '''
        Deep Learning Model 이용해
        도로정보를 확실하게 찾기
        
        STOP LANE 만 구현 해놨음 아직
        '''
        dsf = np.copy(frame)
        w, h = self.video_size
        resize_dst = cv2.resize(dsf, dsize=(h, w))
        lane = self.lane_model.predict(
            resize_dst[int(self.pts1[0][1])-30:int(self.pts1[2][1])+30, :])

        _, lane_model_IMG = cv2.threshold(lane, 1, 255, cv2.THRESH_TOZERO_INV)
        _, lane_th = cv2.threshold(lane_model_IMG, 0, 255, cv2.THRESH_BINARY)

        _, lane_model_IMG = cv2.threshold(lane, 2, 255, cv2.THRESH_TOZERO_INV)
        _, stop_th = cv2.threshold(lane_model_IMG, 1, 255, cv2.THRESH_BINARY)

        _, lane_model_IMG = cv2.threshold(lane, 3, 255, cv2.THRESH_TOZERO_INV)
        _, cross_th = cv2.threshold(lane_model_IMG, 2, 255, cv2.THRESH_BINARY)

        # 허프라인으로 차선, 정지선 찾아줘야함
        gray = cv2.cvtColor(lane_th, cv2.COLOR_BGR2GRAY)
        lane_lines = cv2.HoughLinesP(
            gray, 2.0, np.pi / 180., 90, minLineLength=20, maxLineGap=15)

        gray = cv2.cvtColor(stop_th, cv2.COLOR_BGR2GRAY)
        stop_lines = cv2.HoughLinesP(
            gray, 2.0, np.pi / 180., 90, minLineLength=30, maxLineGap=200)

        gray = cv2.cvtColor(cross_th, cv2.COLOR_BGR2GRAY)
        cross_lines = cv2.HoughLinesP(
            gray, 2.0, np.pi / 180., 90, minLineLength=30, maxLineGap=200)

        max_stop_length = 0
        max_cross_length = 0
        stop_line_index = -1
        cross_line_index = -1
        ilx1, ily1, ilx2, ily2 = 0, 0, 0, 0
        irx1, iry1, irx2, iry2 = 0, 0, 0, 0

        right_arr = []
        left_arr = []
        mid_arr = []
        rm, lm = 0, 0

        try:
            for idx, line in enumerate(stop_lines):
                x1, y1, x2, y2 = line[0]
                if abs(x1-x2) > max_stop_length:
                    stop_line_index = idx
                    max_stop_length = abs(x1-x2)
        except:
            pass

        try:
            for idx, line in enumerate(cross_lines):
                x1, y1, x2, y2 = line[0]
                if abs(x1-x2) > max_cross_length:
                    corss_line_index = idx
                    max_cross_length = abs(x1-x2)
        except:
            pass
        if stop_line_index != -1:
            stopLine = (int(stop_lines[stop_line_index][0][0]), int(stop_lines[stop_line_index][0][1])+int(self.pts1[0][1])-30, int(
                stop_lines[stop_line_index][0][2]), int(stop_lines[stop_line_index][0][3])+int(self.pts1[0][1])-30)
        # 횡단보도 = 정지선
        if cross_line_index != -1:
            corssWorkLine = (int(corss[corss_line_index][0][0]), int(corss_lines[corss_line_index][0][1])+int(self.pts1[0][1]+30), int(
                corss_lines[corss_line_index][0][2]), int(corss_lines[corss_line_index][0][3])+int(self.pts1[0][1])+30)
        # 정지선 X

        # 정지선이 존재한다면
        if stop_line_index != -1:
            # 횡단보도 존재한다면
            if cross_line_index != -1:
                # 정지선이랑 횡단보도 위치 계산해서 정지선이 더 위에있다면
                if corssWorkLine[1] + crossWorkLine[3] > stopLine[1] + stopLine[3]:
                    # 정지선 없는걸로 설정
                    stopLine = None
                # 정지선이 더 아래에 있다면
                else:
                    # 정지선을 정지선으로 설정
                    pass
            # 횡단보도 없다면
            else:
                # 정지선을 정지선으로 설정
                pass
        # 정지선이 존재하지 않는다면
        else:
            # 횡단보도가 있다면
            if cross_line_index != -1:
                # 횡단보도 좌표 낮춰서 설정
                corssWorkLine[1] -= 100
                corssWorkLine[3] -= 100
                stopLine = crossWorkLine
            # 횡단보도 없다면
            else:
                # 정지선은 없음
                stopLine = None

        try:

            for idx, line in enumerate(lane_lines):
                x1, y1, x2, y2 = line[0]
                if x1 < 50 or x1 > 910 or x2 < 50 or x2 > 910:
                    continue
                m, _ = draw_linear_graph(*line[0])

                if abs(x2-x1)+abs(y1-y2) > 20:
                    if abs(m) > 1.5:
                        mid_arr.append((idx, abs(m)))

                    else:
                        if m > 0:
                            right_arr.append((idx, m))
                        else:
                            left_arr.append((idx, m))
        except:
            pass

        # Right Lane
        if right_arr != []:
            right_arr.sort(key=lambda x: (x[1]))
            irx1, iry1, irx2, iry2 = lane_lines[right_arr[-1][0]][0]
            rm, rn = draw_linear_graph(
                irx1, iry1+self.pts1[0][1]-30, irx2, iry2+self.pts1[0][1]-30)
            rx1 = (self.pts1[0][1]-rn)//rm
            rx2 = (self.pts1[2][1]-rn)//rm

            if (np.isnan(irx1) == False or np.isnan(irx2) == False) and rm > 0.3:
                high_dict['right']['x'] = [int(rx1), int(rx2)]
                high_dict['right']['y'] = [
                    int(self.pts1[0][1]), int(self.pts1[2][1])]
        # Left Lane
        if left_arr != []:
            left_arr.sort(key=lambda x: (-x[1]))
            ilx1, ily1, ilx2, ily2 = lane_lines[left_arr[-1][0]][0]
            lm, ln = draw_linear_graph(
                ilx1, ily1+self.pts1[0][1]-30, ilx2, ily2+self.pts1[0][1]-30)
            lx1 = (self.pts1[0][1]-ln)//lm
            lx2 = (self.pts1[2][1]-ln)//lm

            if (np.isnan(ilx1) == False or lx < 600 or np.isnan(ilx2) == False) and lm < -0.3:
                high_dict['left']['x'] = [int(lx2), int(lx1)]
                high_dict['left']['y'] = [
                    int(self.pts1[2][1]), int(self.pts1[0][1])]

        # Mid Lane
        if mid_arr != []:
            mid_arr.sort(key=lambda x: (x[1]))
            x1, y1, x2, y2 = lane_lines[mid_arr[-1][0]][0]

            mm, mn = draw_linear_graph(
                x1, y1+self.pts1[0][1]-30, x2, y2+self.pts1[0][1]-30)
            mx1 = (self.pts1[0][1]-mn)//mm
            mx2 = (self.pts1[2][1]-mn)//mm

            if np.isnan(x1) == False or np.isnan(x2) == False:
                if left_arr == [] and right_arr != [] and x1 < 600:
                    high_dict['left']['x'] = [int(mx2), int(mx1)]
                    high_dict['left']['y'] = [
                        int(self.pts1[2][1]), int(self.pts1[0][1])]
                    irx1, iry1, irx2, iry2 = x1, y1, x2, y2
                    self.Lane_Change = True
                    lm = mm
                elif left_arr != [] and right_arr == [] and x2 > 200:
                    high_dict['right']['x'] = [int(mx1), int(mx2)]
                    high_dict['right']['y'] = [
                        int(self.pts1[0][1]), int(self.pts1[2][1])]
                    ilx1, ily1, ilx2, ily2 = x1, y1, x2, y2
                    self.Lane_Change = True
                    rm = mm

                elif left_arr == [] and right_arr == []:
                    pass
                else:
                    # 왼쪽이 더 큼
                    if mm < 0:
                        high_dict['left']['x'] = [int(mx2), int(mx1)]
                        high_dict['left']['y'] = [
                            int(self.pts1[2][1]), int(self.pts1[0][1])]
                        ilx1, ily1, ilx2, ily2 = x1, y1, x2, y2
                        self.Lane_Change = True
                        # 오른쪽이 더 큼
                    else:
                        high_dict['right']['x'] = [int(mx1), int(mx2)]
                        high_dict['right']['y'] = [
                            int(self.pts1[0][1]), int(self.pts1[2][1])]
                        irx1, iry1, irx2, iry2 = x1, y1, x2, y2
                        self.Lane_Change = True

        # 소실점 다시구해줌

        if lm * rm > 0:
            high_dict = copy.deepcopy({'left': {'x': [], 'y': [], 'state': None},
                                      'right': {'x': [], 'y': [], 'state': None}})
        else:
            try:
                lm, ln = draw_linear_graph(
                    high_dict['left']['x'][0], high_dict['left']['y'][0], high_dict['left']['x'][1], high_dict['left']['y'][1])
                rm, rn = draw_linear_graph(
                    high_dict['right']['x'][0], high_dict['right']['y'][0], high_dict['right']['x'][1], high_dict['right']['y'][1])
                x, y = intersection_point((lm, ln), (rm, rn))
                x, y = int(x), int(y)

                # Free Space 변경
    #             self.pts1[0][1] = y
    #             self.pts1[1][1] = y

                #

            except:
                pass

        # 차선이 Free Space를 벗어날경우 Transform이 어렵기 때문에 넘어간 차선을 FreeSpace 안으로 넣어줌
        # 해당 알고리즘은 차선과 FreeSpace의 윗변과의 교점을 찾아 왼쪽변,윗변,오른쪽변중 어떤변이 교점이 생기는지 판단 후 해당변과의 교점으로 바꿔줌
        # +1의 경우 정수로 바꾸면 내림이기 때문에 좌표밖을 벗어나는 경우가 있기떄문에 실행
        # 왼쪽차선 확인
        if len(high_dict['left']['x']) != 0:
            x, y = intersection_point(draw_linear_graph(high_dict['left']['x'][0], high_dict['left']['y'][0],
                                      high_dict['left']['x'][1], high_dict['left']['y'][1]), draw_linear_graph(*self.pts1[0], *self.pts1[1]))
            # 왼쪽에 닿을경우
            if self.pts1[0][0] > x:
                x, y = intersection_point(draw_linear_graph(high_dict['left']['x'][0], high_dict['left']['y'][0],
                                          high_dict['left']['x'][1], high_dict['left']['y'][1]), draw_linear_graph(*self.pts1[0], *self.pts1[3]))
                x += 1
            # 상단에 닿을경우
            elif self.pts1[0][0] < x < self.pts1[1][0]:
                pass
            # 오른쪽에 닿을경우
            elif self.pts1[1][0] < x:
                x, y = intersection_point(draw_linear_graph(high_dict['left']['x'][0], high_dict['left']['y'][0],
                                          high_dict['left']['x'][1], high_dict['left']['y'][1]), draw_linear_graph(*self.pts1[1], *self.pts1[2]))

            y += 1
            try:
                high_dict['left']['x'][1] = int(x)
                high_dict['left']['y'][1] = int(y)
            except:
                high_dict['left']['x'] = []
                high_dict['left']['y'] = []

        # 오른차선 확인
        if len(high_dict['right']['x']) != 0:
            x, y = intersection_point(draw_linear_graph(high_dict['right']['x'][0], high_dict['right']['y'][0],
                                      high_dict['right']['x'][1], high_dict['right']['y'][1]), draw_linear_graph(*self.pts1[0], *self.pts1[1]))
            # 왼쪽에 닿을경우
            if self.pts1[0][0] > x:
                x, y = intersection_point(draw_linear_graph(high_dict['right']['x'][0], high_dict['right']['y'][0],
                                          high_dict['right']['x'][1], high_dict['right']['y'][1]), draw_linear_graph(*self.pts1[0], *self.pts1[3]))
                x += 1
            # 상단에 닿을경우
            elif self.pts1[0][0] < x < self.pts1[1][0]:
                pass
            # 오른쪽에 닿을경우
            elif self.pts1[1][0] < x:
                x, y = intersection_point(draw_linear_graph(high_dict['right']['x'][0], high_dict['right']['y'][0],
                                          high_dict['right']['x'][1], high_dict['right']['y'][1]), draw_linear_graph(*self.pts1[1], *self.pts1[2]))

            y += 1
            try:
                high_dict['right']['x'][0] = int(x)
                high_dict['right']['y'][0] = int(y)
            except:
                high_dict['right']['x'] = []
                high_dict['right']['y'] = []

        if len(high_dict['right']['x']) == 0 and len(high_dict['left']['x']) == 2:
            high_dict['right']['x'] = copy.deepcopy(
                self.video_queue[-2]['lane']['right']['x'])
            high_dict['right']['y'] = copy.deepcopy(
                self.video_queue[-2]['lane']['right']['y'])

        elif len(high_dict['right']['x']) == 2 and len(high_dict['left']['x']) == 0:
            high_dict['left']['x'] = copy.deepcopy(
                self.video_queue[-2]['lane']['left']['x'])
            high_dict['left']['y'] = copy.deepcopy(
                self.video_queue[-2]['lane']['left']['y'])

        if len(high_dict['left']['x']) != 0:
            high_dict['left']['state'] = lane_state(
                resize_dst, high_dict['left']['x'], high_dict['left']['y'], self.pts1, self.pts2)
#             print('left_high',high_dict['left']['state'])
        if len(high_dict['right']['x']) != 0:
            high_dict['right']['state'] = lane_state(
                resize_dst, high_dict['right']['x'], high_dict['right']['y'], self.pts1, self.pts2)
#             print('right_high',high_dict['right']['state'])

        if len(high_dict['left']['x']) == 0 and len(high_dict['right']['x']) == 0:
            if len(self.video_queue) <= 2:
                return high_dict, stopLine
            else:

                return copy.deepcopy(self.video_queue[-2]['lane']), stopLine
        else:
            return high_dict, stopLine

    def violationDuplcationCheck(self, nowX, nowY):

        isDuplcation = False
        for before_cnt, (x, y) in self.dataDuplication:

            if (self.yolo_cnt - before_cnt < 200) and (abs(nowX-x) + abs(nowY-y) < 30):
                # 중복이라고 판단
                isDuplcation = True
                break

        # 중복 확인 큐 업데이트
        if len(self.dataDuplication) >= 5:
            self.dataDuplication.popleft()
        self.dataDuplication.append([self.yolo_cnt, (nowX, nowY)])

        return isDuplcation

    def Last_Check(self, frame, temp_dict):
        temp_dict['lane'], temp_dict['stopline'] = self.highAccCheck2(frame)
        self.temp_dict['lane'] = copy.deepcopy(temp_dict['lane'])
        self.temp_dict['stopline'] = copy.deepcopy(temp_dict['stopline'])


#         temp_dict['lane']=slide_window_search(lane_th,{'left':(ilx1,ily1,ilx2,ily2),'right':(irx1,iry1,irx2,iry2)})

        self.Car_State_Update(temp_dict)

        if len(self.video_queue) < 10:
            return

        viloation = {'lane': {'left': None, 'right': None},
                     'traffic light': None}

        left = {}
        right = {}
        trafficLight = {}

        for state in [self.video_queue[-1], self.video_queue[-2], self.video_queue[-3], self.video_queue[-4], self.video_queue[-5], self.video_queue[-6], self.video_queue[-7], self.video_queue[-8], self.video_queue[-9], self.video_queue[-10]]:

            if state['lane']['left']['state'] in left:
                left[state['lane']['left']['state']] += 1
            else:
                left[state['lane']['left']['state']] = 1

            if state['lane']['right']['state'] in right:
                right[state['lane']['right']['state']] += 1
            else:
                right[state['lane']['right']['state']] = 1

            if state['traffic light'] in trafficLight:
                trafficLight[state['traffic light']] += 1
            else:
                trafficLight[state['traffic light']] = 1
        left = list(left.items())
        left.sort(key=lambda x: -x[1])
        right = list(right.items())
        right.sort(key=lambda x: -x[1])
        trafficLight = list(trafficLight.items())
        trafficLight.sort(key=lambda x: -x[1])
        viloation['lane']['left'] = left[0][0]
        viloation['lane']['right'] = right[0][0]
        viloation['traffic light'] = trafficLight[0][0]

        stopline = temp_dict['stopline']
        vehicle = temp_dict['vehicle']
        traffic_light = temp_dict['traffic light'].split(' ')[0]
        viloationBefore = self.video_queue[-1]['vehicle']

        for vehicleInfo in temp_dict['vehicle'].items():

            violation = ''

            carNum = vehicleInfo[0]
            carPos = vehicleInfo[1]['pos']
            carType = vehicleInfo[1]['type']
            carState = vehicleInfo[1]['state']
            carSignal = vehicleInfo[1]['Turn_Signal']
            # 한프레임으로 검출가능한 것

            if carNum in self.car_counter:
                continue

            # 정지선위반
            if viloation['traffic light'] == 'TrafficLight_Red' and stopline:
                if (viloation['lane']['left'] == 'Yellow_Solid' or viloation['lane']['right'] == 'Yellow_Solid') and carState == 40:
                    pass
                else:
                    x, y, w, h = carPos
                    if stopline != None and traffic_light == 'TrafficLight_Red' and 200 < (stopline[1]+stopline[3])//2:
                        if x > 200 and x+w < 760 and stop_line_volation(*stopline, x, y, x+w, y+h, carType, carState):
                            violation += '1'
                # 신호위반

                # 이전프레임 봐야 검출 가능한 것

                # 이전에 차량이 있었을 경우
            if carNum in viloationBefore:
                # 진로변경위반
                if viloationBefore[carNum]['state'] == 20:
                    if carState == 40 and (viloation['lane']['left'] == 'White_Solid' or viloation['lane']['left'] == 'Yellow_Solid'):
                        violation += '4'

                if viloationBefore[carNum]['state'] == 40:
                    if carState == 20 and (viloation['lane']['left'] == 'White_Solid' or viloation['lane']['left'] == 'Yellow_Solid'):
                        violation += '4'
                    elif carState == 60 and (viloation['lane']['right'] == 'White_Solid'):
                        violation += '4'

                if viloationBefore[carNum]['state'] == 60:
                    if carState == 40 and (viloation['lane']['left'] == 'White_Solid'):
                        violation += '4'

                    # 제차조작신호불이행
#                 if viloationBefore[carNum]['state'] == 20 :
#                     if (carState == 40) and (carSignal== 1 or carSignal == 2):
#                         self.car_counter[carNum] = 200
#                         violation +='5'

#                 if viloationBefore[carNum]['state'] == 40:
#                     if (carState == 20) and (carSignal== 1 or carSignal == 3):
#                         self.car_counter[carNum] = 200
#                         violation +='5'
#                     if (carState == 60) and (carSignal== 1 or carSignal == 2):
#                         self.car_counter[carNum] = 200
#                         violation +='5'

#                 if viloationBefore[carNum]['state'] == 60:
#                     if (carState == 40) and (carSignal== 1 or carSignal == 3):
#                         self.car_counter[carNum] = 200
#                         violation +='5'

                    # 중앙선 침범
                if viloationBefore[carNum]['state'] == 20 and carState == 40 and viloation['lane']['left'] == 'Yello_Solid':
                    violation += '3'
                elif viloationBefore[carNum]['state'] == 40 and carState == 20 and (viloation['lane']['left'] == 'Yellow_Solid'):
                    violation += '3'

                if violation != '' and self.violationDuplcationCheck(x, y) == False:
                    self.car_counter.add(carNum)
                    print(temp_dict['Video_name'], temp_dict['Video_frame'],
                          '검출내용 :', violation, '차 번호 :', carNum)
                    f = open("result.txt", "a")
                    f.write(temp_dict['Video_name']+str(temp_dict['Video_frame']
                                                        )+'검출내용 :'+str(violation)+'차 번호 :'+str(carNum)+'\n')
                    f.close()
                    try:
                        cv2.imwrite('img/'+self.temp_dict['Video_name'].split('/')[-1]+str(
                            self.temp_dict['Video_frame'])+'.jpg', self.visyalize(-1))
                    except:
                        pass
                    self.wr.writerow(
                        [temp_dict['Video_name'], temp_dict['Video_frame'], violation, carNum])

    def car_counter_update(self):

        # 원래 차량 번호마다 200정도 넣어서 하나씩 줄여갔는데
        # 비효율적인 로직이라
        # 제일 최근차 -100 밑으로 들어오면 제거하는 로직으로 변경
        for key in self.car_counter:
            if key < self.yolo_cnt-100:
                del self.car_counter[key]


def is_cross_pt(x11, y11, x12, y12, x21, y21, x22, y22):
    b1 = is_divide_pt(x11, y11, x12, y12, x21, y21, x22, y22)
    b2 = is_divide_pt(x21, y21, x22, y22, x11, y11, x12, y12)
    if b1 and b2:
        return True
    return False


def is_divide_pt(x11, y11, x12, y12, x21, y21, x22, y22):
    f1 = (x12-x11)*(y21-y11) - (y12-y11)*(x21-x11)
    f2 = (x12-x11)*(y22-y11) - (y12-y11)*(x22-x11)
    if f1*f2 < 0:
        return True
    else:
        return False


def stop_line_volation(x11, y11, x12, y12, x21, y21, x22, y22, ty, beforeLine):

    if ty == 'Vehicle_Car':
        acc = 0.4
    else:
        acc = 0.3

    if beforeLine == 20 or beforeLine == 60:
        acc -= 0.1

    x11 = 0
    x12 = 960
    m1, n1 = draw_linear_graph(x11, y11, x12, y12)
    #     m2,n2 = draw_linear_graph((x21+x22)//2,y21,(x21+x22)//2,y22)

    x = m1*(x21+x22)//2 + n1
    y = m1*x+n1

#     if (y21) < (y11+ y12)//2 -20 :
#         return True
#     else:
#         return False
    if (y11 + y12) // 2 > y22:
        return True

    if abs(y22-y)/abs(y21-y) < acc:
        return True
    else:
        return False


if __name__ == '__main__':

    a = carViolation(video_path='Data', yolo_weight_path='yolov4-tiny-custom_best (5).weights', yolo_class_name_path='ClassNames.names', yolo_cfg_path='yolov4-tiny-custom.cfg', yolo_trafficlight_weight_path='yoloReBuild/yolotrafficlight.weights',
                     yolo_trafficlight_class_name_path='yoloReBuild/yolotrafficlight.names', yolo_trafficlight_cfg_path='yoloReBuild/yolotrafficlight.cfg', ckpt_path='epoch=3-step=99999.ckpt', queue_size=100, cuda=True)
    while True:
        a.frame_load()
