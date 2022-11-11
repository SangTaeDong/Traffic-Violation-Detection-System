import cv2
import numpy as np
import copy
import math
# import matplotlib.pyplot as plt
# from sklearn import linear_model

def draw_linear_graph(x1, y1, x2, y2):
    if x2==x1:
        x1=x1-1
    m = (y2 - y1) / (x2 - x1)
    n = y1 - (m * x1)
    return m, n


def findbase(a,b,y):
    m,n = draw_linear_graph(*a,*b)
    x = (y-n)/m

    if x<0:
        x=1
    if x>350:
        x=349
    y= m*x + n

    return(x,y)

def draw_the_lines(image,lines,base_x,top_x,angle):
    dst = np.copy(image)

    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    # br=0

    p1=[]
    p2=[]

    diff = 1000000

    if lines is None:
        return dst,p1,p2

    for line in lines:
        for x1,y1,x2,y2 in line:
            # if x1 - x2 != 0 and (2 < (abs((y1 - y2) / (x1 - x2)))) and abs(findbase((x1,y1),(x2,y2),350)[0]-base_x)<=30 and abs(findbase((x1,y1),(x2,y2),0)[0]-top_x)<=30:
            if x1 - x2 != 0 and (1 < (abs((y1 - y2) / (x1 - x2)))) and ((y1 - y2) / (x1 - x2))*angle>=0 and abs(findbase((x1,y1),(x2,y2),350)[0]-base_x)<=10 and abs(findbase((x1,y1),(x2,y2),0)[0]-top_x)<=10:
                # cv2.line(dst, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                a=(y1-y2)/(x1-x2)

                # p1 = [x1, y1]
                # p2 = [x2, y2]
                # br=1
                # break

                # if abs(angle-a)<diff:
                #     diff = abs(angle-a)
                #     p1 = [x1, y1]
                #     p2 = [x2, y2]

                if (abs(findbase((x1,y1),(x2,y2),0)[0]-top_x) + abs(findbase((x1,y1),(x2,y2),350)[0]-base_x))<=diff:
                    diff = abs(findbase((x1,y1),(x2,y2),0)[0]-top_x) + abs(findbase((x1,y1),(x2,y2),350)[0]-base_x)
                    p1 = [x1, y1]
                    p2 = [x2, y2]

                # if angle-0.3<=a<=angle+0.3:
                #     p1 = [x1, y1]
                #     p2 = [x2, y2]
                #     br=1
                #     break

                # if angle<0 and angle*1.05<=a<=angle*0.95:
                #     p1 = [x1, y1]
                #     p2 = [x2, y2]
                #     br=1
                #     break
                # if angle>=0 and angle*0.95<=a<=angle*1.05:
                #     p1 = [x1, y1]
                #     p2 = [x2, y2]
                #     br=1
                #     break
        # if br==1:
        #     break

    return dst,p1,p2

def lane_state(img, c, before_info):
    thr = np.copy(img)
    histogram = np.sum(thr[:, :], axis=1)
    lane = []
    for i in range(len(histogram)):
        if histogram[i] > 500:
            lane.append(i)
            
    if len(lane)==0:
        return c + before_info

    lane = lane[::-1]

    line = []
    blank = []

    start = lane[0]
    for i in range(len(lane)-1):
        if lane[i] - lane[i + 1] > 20:
            line.append(lane[i] - start)
            blank.append(lane[i] - lane[i + 1])
            start = lane[i + 1]
            if len(line) == 2:
                break
                
    if len(line)==0:
        line=[0,0]
    elif len(line)==1:
        line.append(0)
    if len(blank)==0:
        blank=[0,0]
    elif len(blank)==1:
        blank.append(0)
        
    if line[0] >= 140 or line[1] >= 140:
        state = '_Solid'
    elif line[0] >= 80 and blank[0] >= 80:
        state = '_Dashed'
    elif line[0] < blank[0] < line[0] * 2 + 5 and line[1] < blank[1] < line[1] * 2 + 5:
        state = '_Dashed'
    else:
        state = before_info

    return c + state

def slide_window_search(binary_warped, current, c, type, video_queue):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 15
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 30
    minpix = 40
    left_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height
        win_y_high = binary_warped.shape[0] - (w) * window_height
        win_xleft_low = current - margin
        win_xleft_high = current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)

        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                nonzero_x < win_xleft_high)).nonzero()[0]
        left_lane.append(good_left)

        if len(good_left) > minpix:
            current = np.int32(np.mean(nonzero_x[good_left]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]

    set_lefty = list(set(lefty))
    length = len(set_lefty)

    if length<=50:
        return video_queue[-1]['lane'][type]['state']

    state = c + '_Solid'
    for i in range(length):
        if i + 1 < length and 70<=set_lefty[i + 1] - set_lefty[i]<=200:
            state = c + '_Dashed'
            break

    return state


def yellow_filter(image):
    white = np.array([255, 255, 255], np.uint8)
    CMY_img = white - image

    cmy = cv2.split(CMY_img)
    black = cv2.min(cmy[0], cv2.min(cmy[1], cmy[2]))
    yellow, _, _ = cmy - black

    yellow = cv2.threshold(yellow, 50, 255, cv2.THRESH_BINARY)[1]

    masked = cv2.bitwise_and(image, image, mask=yellow)

    return masked


def color_filter(image, lower, upper):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    white_mask = cv2.inRange(hls, lower, upper)

    mask = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]).astype('uint8')
    masked = cv2.dilate(white_mask, mask,iterations=3)
    # masked = cv2.dilate(masked, mask)
    # masked = cv2.dilate(masked, mask)

    masked1 = cv2.bitwise_and(image, image, mask=masked)

    masked2 = yellow_filter(image)

    return masked1, masked2


def wrapping(img, source, destination):
    image = np.copy(img)

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (350, 350))

    return _image,transform_matrix,minv

def transform(p1,from_w,from_h,to_w,to_h,per_mat,from_pts,to_pts):
    ns_image = np.zeros((from_h,from_w), np.uint8)
    if from_w==350 and p1[1]<(from_pts[0][1]+from_pts[2][1])//2:
        cv2.circle(ns_image, (int(p1[0]), int(p1[1])+3), 4, 255, -1)
    elif from_w==350 and p1[1]<(from_pts[0][1]+from_pts[2][1])//2:
        cv2.circle(ns_image, (int(p1[0]), int(p1[1])-3), 4, 255, -1)
    else:
        cv2.circle(ns_image, (int(p1[0]), int(p1[1])), 4, 255, -1)

    dst = cv2.warpPerspective(ns_image,per_mat,(to_w,to_h),cv2.INTER_CUBIC)
    dst = np.nonzero(dst)

    # if len(dst[0])==0:
    #     return None,None

    return dst[1][0],dst[0][0]

def _lanedetect(img, pts1, pts2, lower_white_hls, upper_white_hls, video_queue):
    re_img = np.copy(img)

    ##########################################################################
    pts3 = np.copy(pts1)

    y = pts3[2][1] - 120
    m1,n1 = draw_linear_graph(*pts3[3],*pts3[0])
    m2,n2 = draw_linear_graph(*pts3[2],*pts3[1])

    x1 = (y-n1)/m1
    x2 = (y-n2)/m2

    pts3[0]=(x1,y)
    pts3[1]=(x2,y)

    tmp_cut, permat_cut, minverse_cut = wrapping(re_img, pts3, pts2)
    tmp1_cut, tmp2_cut = color_filter(tmp_cut, lower_white_hls, upper_white_hls)

    _gray = cv2.cvtColor(tmp1_cut, cv2.COLOR_BGR2GRAY)
    ret, thr1_cut = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

    _gray = cv2.cvtColor(tmp2_cut, cv2.COLOR_BGR2GRAY)
    ret, thr2_cut = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

    thr_cut = cv2.add(thr1_cut, thr2_cut)

    ns_image = np.zeros((350,350), np.uint8)
    cv2.rectangle(ns_image,(0,0,350,300),255,-1)
    thr_cut = cv2.bitwise_and(thr_cut, ns_image)
    ##########################################################################

    #birdeyesviews
    tmp,permat,minverse = wrapping(re_img, pts1, pts2)

    #colorfilter & threshold
    tmp1, tmp2 = color_filter(tmp, lower_white_hls, upper_white_hls)

    _gray = cv2.cvtColor(tmp1, cv2.COLOR_BGR2GRAY)
    ret, thr1 = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

    _gray = cv2.cvtColor(tmp2, cv2.COLOR_BGR2GRAY)
    ret, thr2 = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

    thr = cv2.add(thr1, thr2)

    ns_image = np.zeros((350,350), np.uint8)
    cv2.rectangle(ns_image,(0,0,350,300),255,-1)
    thr = cv2.bitwise_and(thr, ns_image)

    #define dictionary
    draw_info_dict = {'left': {'x': [], 'y': [], 'state': None}, 'right': {'x': [], 'y': [], 'state': None}}

    temp=2 #for checking missing both left and right

    for i in range(2):
        if i==0:
            #left
            if len(video_queue[-1]['lane']['left']['y'])!=0:
                # leftbase, leftangle
                lb_index = np.argmax(video_queue[-1]['lane']['left']['y'])
                lb = (video_queue[-1]['lane']['left']['x'][lb_index], video_queue[-1]['lane']['left']['y'][lb_index])
                lt_index = np.argmin(video_queue[-1]['lane']['left']['y'])
                lt = (video_queue[-1]['lane']['left']['x'][lt_index], video_queue[-1]['lane']['left']['y'][lt_index])

                if lt[1]<pts1[0][1]:
                    y = pts1[0][1]
                    m1, n1 = draw_linear_graph(*lb, *lt)

                    x1 = (y - n1) / m1

                    lt= (x1,y)
                    # print('left',x1,y,m1,n1)
                    if math.isnan(x1):
                        temp-=1
                        draw_info_dict['left'] = copy.deepcopy(video_queue[-1]['lane']['left'])
                        continue


                # print(pts1)
                # print('left x', video_queue[-1]['lane']['left']['x'])
                # print('left y', video_queue[-1]['lane']['left']['y'])

                # 점이 Freespace(ROI)를 넘는 경우에 except 처리가 됨
                try:
                    lbx,lby = transform(lb,960,540,350,350,permat,pts1,pts2)
                    ltx,lty = transform(lt,960,540,350,350,permat,pts1,pts2)
                except:
                    draw_info_dict['left'] = copy.deepcopy(video_queue[-1]['lane']['left'])
                    continue

                la = (lty - lby) / (ltx - lbx)

                ##########################################################################
                if lt[1]<pts3[0][1]:
                    y = pts3[0][1]
                    m1, n1 = draw_linear_graph(*lb, *lt)

                    x1 = (y - n1) / m1

                    lb_cut= lb
                    lt_cut= (x1,y)
                    # print('left',x1,y,m1,n1)
                    if math.isnan(x1):
                        temp-=1
                        draw_info_dict['left'] = copy.deepcopy(video_queue[-1]['lane']['left'])
                        continue

                else:
                    lb_cut= lb
                    lt_cut= lt

                # print(pts3)
                # print('left x', video_queue[-1]['lane']['left']['x'])
                # print('left y', video_queue[-1]['lane']['left']['y'])

                # 점이 Freespace(ROI)를 넘는 경우에 except 처리가 됨
                try:
                    lbx_cut,lby_cut = transform(lb_cut,960,540,350,350,permat_cut,pts3,pts2)
                    ltx_cut,lty_cut = transform(lt_cut,960,540,350,350,permat_cut,pts3,pts2)
                except:
                    draw_info_dict['left'] = copy.deepcopy(video_queue[-1]['lane']['left'])
                    continue

                la_cut = (lty_cut - lby_cut) / (ltx_cut - lbx_cut)
                ##########################################################################

                # if lbx==None or lby==None or ltx==None or lty==None:
                #     draw_info_dict['left'] = copy.deepcopy(video_queue[-1]['lane']['left'])
                #     break

                #left fit area
                zero_image = np.zeros((350, 350), np.uint8)
                cv2.rectangle(zero_image, (lbx, lby), (ltx, lty), 255, -1)
                # if la > 0:
                #     cv2.rectangle(zero_image, (lbx-10, 0), (lbx + 100, 350), 255, -1)
                #     cv2.rectangle(zero_image, (lbx, lby), (ltx, lty), 255, -1)
                # elif la < 0:
                #     cv2.rectangle(zero_image, (lbx + 10, 0), (lbx - 100, 350), 255, -1)
                # else:
                #     cv2.rectangle(zero_image,(lbx-50,0),(lbx+50,350),255,-1)
                left_thr = cv2.bitwise_and(thr, zero_image)

                ##########################################################################
                zero_image = np.zeros((350, 350), np.uint8)
#                 cv2.rectangle(zero_image, (lbx_cut, lby_cut), (ltx_cut, lty_cut), 255, -1)
                cv2.line(zero_image, (lbx_cut, lby_cut), (ltx_cut, lty_cut), 255, 40)
                # if la_cut > 0:
                #     cv2.rectangle(zero_image, (lbx_cut-10, 0), (lbx_cut + 100, 350), 255, -1)
                # elif la_cut < 0:
                #     cv2.rectangle(zero_image, (lbx_cut + 10, 0), (lbx_cut - 100, 350), 255, -1)
                # else:
                #     cv2.rectangle(zero_image,(lbx_cut-50,0),(lbx_cut+50,350),255,-1)
                left_thr_cut = cv2.bitwise_and(thr_cut, zero_image)  #####
                ##########################################################################

                # fit color
                left_c = 'White'

                yellow_left = cv2.bitwise_and(left_thr_cut, left_thr_cut, mask=thr2_cut)
                ylh = np.sum(yellow_left[:, :], axis=0)
                yls = np.sum(ylh)
#                 if yls >= 2000:
                before = copy.deepcopy(video_queue[-1]['lane']['left']['state'])
                
                if before==None:
#                     print('None left',before)
                    before_type='_Dashed'
                    if yls >= 20000: #0829까지40000이였음
                        left_c = 'Yellow'
                else:
#                     print('yes left',before)
                    before_color, before_type = before.split('_')
                    before_type = '_'+before_type
                    if yls >= 20000: #0829까지40000이였음
                        left_c = 'Yellow'
                    elif 10000<=yls<20000: #20000 40000
                        left_c = before_color

                # left houghline
                blur_image = cv2.GaussianBlur(thr, (5, 5), 0)
                canny_image = cv2.Canny(blur_image, 100, 120)
                lines = cv2.HoughLinesP(canny_image,
                                        rho=4,  # 4
                                        theta=np.pi / 180,
                                        threshold=100,
                                        lines=np.array([]),
                                        minLineLength=50,  # 10#40
                                        maxLineGap=100)  # 50
                iwll, lp1, lp2 = draw_the_lines(canny_image, lines,lbx,ltx,la)

                if len(lp1)!=0:
                    warped_left_top_point = findbase(lp1, lp2, 1)
                    warped_left_btm_point = findbase(lp1, lp2, 349)

                    # print(warped_left_top_point,warped_left_btm_point)
                    # print(pts2)
                    # print(pts1)

                    real_left_top_x, real_left_top_y = transform(warped_left_top_point, 350, 350, 960, 540, minverse,pts2,pts1)
                    real_left_btm_x, real_left_btm_y = transform(warped_left_btm_point, 350, 350, 960, 540, minverse,pts2,pts1)

                    '''if lp1[1]<lp2[1]:
                        # state = slide_window_search(left_thr, lp2[0], left_c, 'left', video_queue)
                        state = slide_window_search(left_thr_cut, lp2[0], left_c, 'left', video_queue)####
                    else:
                        # state = slide_window_search(left_thr, lp1[0], left_c, 'left', video_queue)
                        state = slide_window_search(left_thr_cut, lp1[0], left_c, 'left', video_queue)####'''
                    
                    
                    state = lane_state(left_thr_cut, left_c, before_type)


                    draw_info_dict['left']['x'].append(int(real_left_top_x))
                    draw_info_dict['left']['x'].append(int(real_left_btm_x))
                    draw_info_dict['left']['y'].append(int(real_left_top_y))
                    draw_info_dict['left']['y'].append(int(real_left_btm_y))

                    draw_info_dict['left']['state']=state

                else:
                    temp-=1
                    draw_info_dict['left'] = copy.deepcopy(video_queue[-1]['lane']['left'])

        if i==1:

            #right
            if len(video_queue[-1]['lane']['right']['y'])!=0:
                # rightbase, rightangle
                rb_index = np.argmax(video_queue[-1]['lane']['right']['y'])
                rb = (video_queue[-1]['lane']['right']['x'][rb_index], video_queue[-1]['lane']['right']['y'][rb_index])
                rt_index = np.argmin(video_queue[-1]['lane']['right']['y'])
                rt = (video_queue[-1]['lane']['right']['x'][rt_index], video_queue[-1]['lane']['right']['y'][rt_index])

                if rt[1]<pts1[0][1]:
                    y = pts1[0][1]
                    m1, n1 = draw_linear_graph(*rb, *rt)

                    x1 = (y - n1) / m1

                    rt= (x1,y)
                    # print('left',x1,y,m1,n1)
                    if math.isnan(x1):
                        temp-=1
                        draw_info_dict['left'] = copy.deepcopy(video_queue[-1]['lane']['left'])
                        continue

                # print(pts1)
                # print('right x', video_queue[-1]['lane']['right']['x'])
                # print('right y', video_queue[-1]['lane']['right']['y'])

                # 점이 Freespace(ROI)를 넘는 경우에 except 처리가 됨
                try:
                    rbx, rby = transform(rb, 960, 540, 350, 350, permat,pts1,pts2)
                    rtx, rty = transform(rt, 960, 540, 350, 350, permat,pts1,pts2)
                except:
                    draw_info_dict['right'] = copy.deepcopy(video_queue[-1]['lane']['right'])
                    continue

                ra = (rty - rby) / (rtx - rbx)

                ##########################################################################
                if rt[1]<pts3[0][1]:
                    y = pts3[0][1]
                    m1, n1 = draw_linear_graph(*rb, *rt)

                    x1 = (y - n1) / m1

                    rb_cut= rb
                    rt_cut= (x1,y)
                    # print('right',x1, y, m1, n1)
                    if math.isnan(x1):
                        temp-=1
                        draw_info_dict['right'] = copy.deepcopy(video_queue[-1]['lane']['right'])
                        continue
                else:
                    rb_cut= rb
                    rt_cut= rt

                # print(pts3)
                # print('right x', video_queue[-1]['lane']['right']['x'])
                # print('right y', video_queue[-1]['lane']['right']['y'])

                # 점이 Freespace(ROI)를 넘는 경우에 except 처리가 됨
                try:
                    rbx_cut,rby_cut = transform(rb_cut,960,540,350,350,permat_cut,pts3,pts2)
                    rtx_cut,rty_cut = transform(rt_cut,960,540,350,350,permat_cut,pts3,pts2)
                except:
                    draw_info_dict['right'] = copy.deepcopy(video_queue[-1]['lane']['right'])
                    continue

                ra_cut = (rty_cut - rby_cut) / (rtx_cut - rbx_cut)
                ##########################################################################

                # if rbx==None or rby==None or rtx==None or rty==None:
                #     draw_info_dict['right'] = copy.deepcopy(video_queue[-1]['lane']['right'])
                #     break



                #right fit area
                zero_image = np.zeros((350, 350), np.uint8)
                cv2.rectangle(zero_image, (rbx, rby), (rtx, rty), 255, -1)
                # if ra > 0:
                #     cv2.rectangle(zero_image, (rbx-10, 0),( rbx + 100, 350), 255, -1)
                # elif ra < 0:
                #     cv2.rectangle(zero_image, (rbx+10, 0), (rbx - 100, 350), 255, -1)
                # else:
                #     cv2.rectangle(zero_image, (rbx-50,0), (rbx+50,350),255,-1)
                right_thr = cv2.bitwise_and(thr, zero_image)

                ##########################################################################
                zero_image = np.zeros((350, 350), np.uint8)
#                 cv2.rectangle(zero_image, (rbx_cut, rby_cut), (rtx_cut, rty_cut), 255, -1)
                cv2.line(zero_image, (rbx_cut, rby_cut), (rtx_cut, rty_cut), 255, 40)
                # if ra_cut > 0:
                #     cv2.rectangle(zero_image, (rbx_cut-10, 0), (rbx_cut + 100, 350), 255, -1)
                # elif ra_cut < 0:
                #     cv2.rectangle(zero_image, (rbx_cut + 10, 0), (rbx_cut - 100, 350), 255, -1)
                # else:
                #     cv2.rectangle(zero_image,(rbx_cut-50,0),(rbx_cut+50,350),255,-1)
                right_thr_cut = cv2.bitwise_and(thr_cut, zero_image)  #####
                ##########################################################################

                # fit color
                right_c = 'White'

                yellow_right = cv2.bitwise_and(right_thr_cut, right_thr_cut, mask=thr2_cut)
                yrh = np.sum(yellow_right[:, :], axis=0)
                yrs = np.sum(yrh)
#                 if yrs >= 3000:
                before = copy.deepcopy(video_queue[-1]['lane']['right']['state'])
                if before==None:
#                     print('right',before)
                    if yrs >= 20000: #0829까지40000이였음
                        right_c = 'Yellow'
                    before_type='_Dashed'
                else:
#                     print('right',before)
                    before_color, before_type = before.split('_')
                    before_type = '_'+before_type
                    if yrs >= 20000: #0829까지40000이였음
                        right_c = 'Yellow'
                    elif 10000<=yrs<20000: #20000 40000
                        right_c = before_color

                # right houghline
                blur_image = cv2.GaussianBlur(thr, (5, 5), 0)
                canny_image = cv2.Canny(blur_image, 100, 120)
                lines = cv2.HoughLinesP(canny_image,
                                        rho=4,  # 4
                                        theta=np.pi / 180,
                                        threshold=100,
                                        lines=np.array([]),
                                        minLineLength=50,  # 10#40
                                        maxLineGap=100)  # 50
                iwlr, rp1, rp2 = draw_the_lines(canny_image, lines,rbx,rtx,ra)

                if len(rp1)!=0:
                    warped_right_top_point = findbase(rp1, rp2, 1)
                    warped_right_btm_point = findbase(rp1, rp2, 349)

                    real_right_top_x, real_right_top_y = transform(warped_right_top_point, 350, 350, 960, 540, minverse,pts2,pts1)
                    real_right_btm_x, real_right_btm_y = transform(warped_right_btm_point, 350, 350, 960, 540, minverse,pts2,pts1)

                    '''if rp1[1]<rp2[1]:
                        # state = slide_window_search(right_thr, rp2[0], right_c, 'right', video_queue)
                        state = slide_window_search(right_thr_cut, rp2[0], right_c, 'right', video_queue)####
                    else:
                        # state = slide_window_search(right_thr, rp1[0], right_c, 'right', video_queue)
                        state = slide_window_search(right_thr_cut, rp1[0], right_c, 'right', video_queue)####'''
                    
                    state = lane_state(right_thr_cut, right_c, before_type)

                    draw_info_dict['right']['x'].append(int(real_right_top_x))
                    draw_info_dict['right']['x'].append(int(real_right_btm_x))
                    draw_info_dict['right']['y'].append(int(real_right_top_y))
                    draw_info_dict['right']['y'].append(int(real_right_btm_y))

                    draw_info_dict['right']['state']=state

                else:
                    temp-=1
                    draw_info_dict['right'] = copy.deepcopy(video_queue[-1]['lane']['right'])

    if len(draw_info_dict['left']['x'])!=0 and len(draw_info_dict['right']['x'])!=0:
        lm, ln = draw_linear_graph(draw_info_dict['left']['x'][0],draw_info_dict['left']['y'][0],draw_info_dict['left']['x'][1],draw_info_dict['left']['y'][1])
        rm, rn = draw_linear_graph(draw_info_dict['right']['x'][0], draw_info_dict['right']['y'][0],draw_info_dict['right']['x'][1], draw_info_dict['right']['y'][1])

        if rm!=lm:
            x = (rn-ln)/(lm-rm)
            y = (lm*x) + ln

            # if 0<=y<= 540:
            if pts1[0][1]<=y<=pts1[2][1]:
                min_left_index = np.argmin(draw_info_dict['left']['y'])
                min_right_index = np.argmin(draw_info_dict['right']['y'])

                draw_info_dict['left']['x'][min_left_index] = int(x)
                draw_info_dict['left']['y'][min_left_index] = int(y)
                draw_info_dict['right']['x'][min_right_index] = int(x)
                draw_info_dict['right']['y'][min_right_index] = int(y)

    # if temp==0:
    #     return {'left': {'x': [], 'y': [], 'state': None}, 'right': {'x': [], 'y': [], 'state': None}}

    return draw_info_dict