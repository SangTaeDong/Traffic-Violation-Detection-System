import cv2
import numpy as np

def cover_not_interest(img, first_check, video_queue):
    image = np.copy(img)

    if first_check== True:
        return image

    if video_queue[-1]['lane']['left']['state'] != None and 'Yellow' in video_queue[-1]['lane']['left']['state']:
        lb_index = np.argmax(video_queue[-1]['lane']['left']['y'])
        lt_index = np.argmin(video_queue[-1]['lane']['left']['y'])

        top_x = video_queue[-1]['lane']['left']['x'][lt_index] - 20
        top_y = video_queue[-1]['lane']['left']['y'][lt_index] - 20
        bottom_x = video_queue[-1]['lane']['left']['x'][lb_index] - 20
        bottom_y = video_queue[-1]['lane']['left']['y'][lb_index] - 20
        pts = np.array([(0,0),(top_x,0),(top_x,top_y),(bottom_x,bottom_y),(0,bottom_y)])
        image = cv2.fillPoly(image, [pts.astype('int')], (0, 0, 0))

    if video_queue[-1]['lane']['right']['state'] != None and 'Yellow' in video_queue[-1]['lane']['right']['state']:
        rb_index = np.argmax(video_queue[-1]['lane']['right']['y'])
        rt_index = np.argmin(video_queue[-1]['lane']['right']['y'])

        top_x = video_queue[-1]['lane']['right']['x'][rt_index] + 20
        top_y = video_queue[-1]['lane']['right']['y'][rt_index] - 20
        bottom_x = video_queue[-1]['lane']['right']['x'][rb_index] + 20
        bottom_y = video_queue[-1]['lane']['right']['y'][rb_index] - 20
        pts = np.array([(image.shape[1],0),(top_x,0),(top_x,top_y),(bottom_x,bottom_y),(image.shape[1],bottom_y)])
        image = cv2.fillPoly(image, [pts.astype('int')], (0, 0, 0))

    return image