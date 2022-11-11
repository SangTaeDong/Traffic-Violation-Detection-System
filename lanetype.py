import numpy as np
import cv2

def draw_linear_graph(x1, y1, x2, y2):
    if x2==x1:
        x1=x1-1
    m = (y2 - y1) / (x2 - x1)
    n = y1 - (m * x1)
    return m, n

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

def lane_state(img,x_list,y_list,pts1,pts2):
    thr = np.copy(img)

    lower_white_hls = np.array([0,180,0])
    upper_white_hls = np.array([179,255,255])

    height, width = thr.shape[:2]
    ns_image = np.zeros((height, width, 3), np.uint8)

    cv2.line(ns_image, (x_list[0], y_list[0]), (x_list[1], y_list[1]), 255, 40)

    re_img = cv2.bitwise_and(thr,ns_image)

    ###################################################
    pts3 = np.copy(pts1)

    y = pts3[2][1] - 120
    m1,n1 = draw_linear_graph(*pts3[3],*pts3[0])
    m2,n2 = draw_linear_graph(*pts3[2],*pts3[1])

    x1 = (y-n1)/m1
    x2 = (y-n2)/m2

    pts3[0]=(x1,y)
    pts3[1]=(x2,y)
    
    # new warp and ns_image
    new_cut, _, _ = wrapping(ns_image, pts3, pts2)
    
    _gray = cv2.cvtColor(new_cut, cv2.COLOR_BGR2GRAY)
    ret, thr3_cut = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

    # warp and white,yellow
    tmp_cut, permat_cut, minverse_cut = wrapping(re_img, pts3, pts2)
    tmp1_cut, tmp2_cut = color_filter(tmp_cut, lower_white_hls, upper_white_hls)

    _gray = cv2.cvtColor(tmp1_cut, cv2.COLOR_BGR2GRAY)
    ret, thr1_cut = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

    _gray = cv2.cvtColor(tmp2_cut, cv2.COLOR_BGR2GRAY)
    ret, thr2_cut = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

    #white+yellow#cut
    thr_cut = cv2.add(thr1_cut, thr2_cut)

    #밑에 제거
    ns_image = np.zeros((350,350), np.uint8)
    cv2.rectangle(ns_image,(0,0,350,300),255,-1)
    thr_cut = cv2.bitwise_and(thr_cut, ns_image)
    thr2_cut = cv2.bitwise_and(thr2_cut, ns_image)
    ###################################################

    ####color#####
    color='White'

    yh = np.sum(thr2_cut[:, :], axis=0)
    ys = np.sum(yh)

    if ys >= 20000: #0829까지 20000
        color = 'Yellow'
    
#     lh = np.sum(thr3_cut[:,:], axis=0)
#     ls = np.sum(lh)
    
#     color='Yellow'
    
#     wh = np.sum(thr1_cut[:,:], axis=0)
#     ws = np.sum(wh)
    
#     if ls !=0 and (ws//ls)>=0.2:
#         color = 'White'

    ###################

    #####type######

    histogram = np.sum(thr_cut[:, :], axis=1)
    lane = []
    for i in range(len(histogram)):
        if histogram[i] > 500:
            lane.append(i)
    
    if len(lane)==0:
        return color + '_Dashed'
    
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

    if line[0] >= 140 or line[1] >= 140:
        state = '_Solid'
    else:
        state = '_Dashed'

    return color + state