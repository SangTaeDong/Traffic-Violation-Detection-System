import cv2
import numpy as np

def yellow_filter(image):
    hue_box = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    lhl=np.array([20,0,0])
    uhl=np.array([32,255,255])
    
    yellow_box = cv2.inRange(hue_box,lhl,uhl)

    gray_box = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    real_box = cv2.bitwise_and(gray_box,yellow_box)

    return real_box

def turn_signal(img, v_dict, video_que):
    for v in v_dict.keys():
        x, y, w, h = v_dict[v]['pos']
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + w > 960:
            w = 960 - x
        if y + h > 540:
            h = 540 - y

        if 'Motor' in v_dict[v]['type']:
            box = img[y:y + h, x + w // 10:x + w - w // 10]
        else:
            box = img[y + h // 10:y + h - h // 10, x:x + w]

        if box.size==0:
            v_dict[v]['Turn_Signal_Left_Histogram'] = 0
            v_dict[v]['Turn_Signal_Right_Histogram'] = 0
            v_dict[v]['Turn_Signal'] = 1
            continue

        box = yellow_filter(box)
        box = cv2.resize(box, (360, 200))

        for i in range(2):
            split_box = box[::, i * box.shape[1] // 2:(i + 1) * box.shape[1] // 2]
            brightness_hist = cv2.calcHist([split_box], [0], None, [32], [0, 255])

            b_sum = 0
            left_sum = 0
            right_sum = 0
            for j, b in enumerate(brightness_hist):
                if b[0] == 0:
                    continue
                else:
                    b_sum += j * b[0]

            if i==0:
                v_dict[v]['Turn_Signal_Left_Histogram'] = b_sum

                for i in range(max(len(video_que) - 16, 0), len(video_que) - 1):
                    if v in video_que[i]['vehicle'].keys() and v in video_que[i + 1]['vehicle'].keys():
                        left_sum += abs(
                            video_que[i]['vehicle'][v]['Turn_Signal_Left_Histogram'] - video_que[i + 1]['vehicle'][v]['Turn_Signal_Left_Histogram'])
            elif i==1:
                v_dict[v]['Turn_Signal_Right_Histogram'] = b_sum

                for i in range(max(len(video_que) - 16, 0), len(video_que) - 1):
                    if v in video_que[i]['vehicle'].keys() and v in video_que[i + 1]['vehicle'].keys():
                        right_sum += abs(
                            video_que[i]['vehicle'][v]['Turn_Signal_Right_Histogram'] - video_que[i + 1]['vehicle'][v]['Turn_Signal_Right_Histogram'])

        if left_sum<30 and right_sum<30:
            v_dict[v]['Turn_Signal'] = 1
        elif left_sum>right_sum:
            v_dict[v]['Turn_Signal'] = 2
        elif left_sum<right_sum:
            v_dict[v]['Turn_Signal'] = 3
        else:
            v_dict[v]['Turn_Signal'] = 4