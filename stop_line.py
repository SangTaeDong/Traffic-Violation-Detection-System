import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_linear_graph(x1,y1,x2,y2):

    m = (y2 - y1) / (x2 - x1)
    n = y1 - (m * x1)
    return m,n

def transform(x,y,pts1,pts2):
    y_start=pts1[0][1]# 원래 이미지 y의 시작점

    h1 = abs(pts1[0][1] - pts1[2][1])# 원래 이미지 높이
    h2 = pts2[3][1]# 버드 이미지 높이
    w2 = pts2[1][0]# 버드 이미지 너비

    #y좌표
    result_y = y_start + (y/h2)*h1

    #등변사다리꼴 직선의 방정식 구해준다
    dm1, dn1 = draw_linear_graph(*pts1[3],*pts1[0])
    dm2, dn2 = draw_linear_graph(*pts1[2],*pts1[1])

    x1 = (result_y - dn1) / dm1
    x2 = (result_y - dn2) / dm2

    result_x = (x2-x1)*(x/w2) + x1

    return (result_x, result_y)

def draw_the_lines(image,lines):
    dst = np.copy(image)

    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    br=0

    p1=[]
    p2=[]

    if lines is None:
        return dst,p1,p2

    max_x=170

    for line in lines:
        for x1,y1,x2,y2 in line:
            # if x1 - x2 != 0 and (0.00001 < (abs((y1 - y2) / (x1 - x2)) < 0.15)) and max(x1,x2)>170:
            if x1 - x2 != 0 and (0.00001 < (abs((y1 - y2) / (x1 - x2)) < 0.15)) and max(x1,x2)>max_x:
                # cv2.line(dst, (x1, y1), (x2, y2), (0, 255, 0), 2,cv2.LINE_AA)
                max_x = max(x1,x2)
                # cv2.circle(dst, (int(x1), int(y1)), 3, (255, 0, 255), -1)
                # cv2.circle(dst, (int(x2), int(y2)), 3, (255, 0, 255), -1)
                # br=1
                p1 = [x1, y1]
                p2 = [x2, y2]
                # break
        # if br==1:
        #     break
    
    if len(p1)!=0:
        cv2.circle(dst, (int(p1[0]), int(p1[1])), 3, (255, 0, 255), -1)
        cv2.circle(dst, (int(p2[0]), int(p2[1])), 3, (255, 0, 255), -1)

    return dst,p1,p2

def yellow_filter(image):
    white = np.array([255,255,255], np.uint8)
    CMY_img = white - image

    cmy = cv2.split(CMY_img)
    black = cv2.min(cmy[0], cv2.min(cmy[1],cmy[2]))
    yellow,_,_ = cmy - black

    yellow = cv2.threshold(yellow,50,255,cv2.THRESH_BINARY)[1]

    return yellow

def white_filter(image,lower,upper):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    white = cv2.inRange(hls,lower,upper)

    return white

def warp(image,pts1,pts2):
    img = np.copy(image)
    perspect_mat = cv2.getPerspectiveTransform(pts1, pts2)
    minv = cv2.getPerspectiveTransform(pts2,pts1)
    dst = cv2.warpPerspective(img, perspect_mat, (350, 350), cv2.INTER_CUBIC)

    return dst, minv


def _stopline(img,good_dict,pts3,pts2,lower_white_hls,upper_white_hls,cuda,hide_img,video_queue):
    re_img = np.copy(img)

    pts1 = np.copy(pts3)

    y = pts1[2][1] - 120
    m1,n1 = draw_linear_graph(*pts1[3],*pts1[0])
    m2,n2 = draw_linear_graph(*pts1[2],*pts1[1])

    x1 = (y-n1)/m1
    x2 = (y-n2)/m2

    pts1[0]=(x1,y)
    pts1[1]=(x2,y)

    if cuda == False:
        for key,value in good_dict.items():
            cv2.rectangle(re_img,(value[0],value[1]),(value[0]+value[2],value[1]+value[3]),(0,0,0),cv2.FILLED)
    else:
        re_img = hide_img

    # birdeyesview
    dst, minv = warp(re_img, pts1, pts2)

    #color_filter
    white = white_filter(dst,lower_white_hls,upper_white_hls)
    yellow = yellow_filter(dst)

    ns_image = np.zeros((350,350), np.uint8)
    cv2.rectangle(ns_image,(0,0,350,300),255,-1)
    white = cv2.bitwise_and(white, ns_image)

    #yellow_left_remove
    yellow_histo = np.sum(yellow[:,:],axis=0)
    midpoint = yellow_histo.shape[0]//2

    left_yellow = yellow_histo[:midpoint]
    remove_point = 0
    for i in range(len(left_yellow)):
        if left_yellow[i]>9000:
            remove_point=i

    ns_image = np.zeros((350,350), np.uint8)
    cv2.rectangle(ns_image, (remove_point, 0, 350, 350), 255, -1)
    white = cv2.bitwise_and(white, ns_image)

    #process
    blur_image = cv2.GaussianBlur(white,(5,5),0)
    canny_image = cv2.Canny(blur_image,100,120)

    #houghline
    lines = cv2.HoughLinesP(canny_image,
                            rho=4,#4
                            theta=np.pi/180,
                            threshold=80, #100
                            lines=np.array([]),
                            minLineLength=20,#90#10#40
                            maxLineGap=120)#50

    iwl,p1,p2 = draw_the_lines(canny_image, lines)

    if len(p1)==0:
        return re_img, None
    #transform
    ns_image = np.zeros((350,350), np.uint8)
    cv2.circle(ns_image, (int(p1[0]), int(p1[1])), 2, 255, -1)
    cv2.circle(ns_image, (int(p2[0]), int(p2[1])), 2, 255, -1)

    dst = cv2.warpPerspective(ns_image,minv,(960,540),cv2.INTER_CUBIC)
    dst = np.nonzero(dst)

    # stopline_info = (dst[1][0],dst[0][0],dst[1][-1],dst[0][-1])

    s_m,s_n = draw_linear_graph(dst[1][0],dst[0][0],dst[1][-1],dst[0][-1])

    x1 = (s_n - n1) / (m1 - s_m)
    y1 = m1 * x1 + n1

    x2 = (s_n - n2) / (m2 - s_m)
    y2 = m2 * x2 + n2

    stopline_info = (x1, y1, x2, y2)

    return re_img, stopline_info