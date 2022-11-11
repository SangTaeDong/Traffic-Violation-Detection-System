import numpy as np
import cv2

def slide_window_search(img, dict):
    result = {'left':{'x':[],'y':[],'state':None}, 'right':{'x':[],'y':[],'state':None}}

    nwindows = 60 #15
    window_height = np.int32(img.shape[0]/nwindows)
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 30
    minpix = 40 #50

    color = [0,255,0]
    thickness = 2

    current_list = []
    if dict['left'][1] > dict['left'][3]:
        current_list.append(dict['left'][0])
    else:
        current_list.append(dict['left'][2])

    if dict['right'][1] > dict['right'][3]:
        current_list.append(dict['right'][0])
    else:
        current_list.append(dict['right'][2])

    for i,current in enumerate(current_list):
        lane = []
        for w in range(nwindows):
            win_y_low = img.shape[0] - (w+1)*window_height
            win_y_high = img.shape[0] - (w)*window_height
            win_x_low = current - margin
            win_x_high = current + margin

            good_nz = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)).nonzero()[0]
            lane.append(good_nz)

            if len(good_nz)>minpix:
                current = np.int32(np.mean(nonzero_x[good_nz]))

        lane = np.concatenate(lane)

        x = nonzero_x[lane]
        y = nonzero_y[lane]

        if len(x)==0:
            if i==0:
                result['left']['x'].append(dict['left'][0])
                result['left']['x'].append(dict['left'][2])
                result['left']['y'].append(dict['left'][1])
                result['left']['y'].append(dict['left'][3])
            else:
                result['right']['x'].append(dict['left'][0])
                result['right']['x'].append(dict['left'][2])
                result['right']['y'].append(dict['left'][1])
                result['right']['y'].append(dict['left'][3])
            continue

        fit = np.polyfit(y,x,2)

        ploty = np.linspace(min(y), max(y) - 1, max(y)-min(y))
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

        fitx = np.trunc(fitx)

        if i==0:
            result['left']['x']=fitx.astype('int')
            result['left']['y']=ploty.astype('int')
        else:
            result['right']['x']=fitx.astype('int')
            result['right']['y']=ploty.astype('int')

    return result