import matplotlib.pyplot as plt
import numpy as np
import ujson as json
import os
import random
from PIL import Image
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

jpg_path = "//media/heekon/beta/NIA/gwangyuk/png/"
json_path = "//media/heekon/beta/NIA/gwangyuk/json/"
ff = os.listdir(jpg_path)
random.shuffle(ff)
# ff.sort()
resize_w, resize_h = 800, 800
for f in ff:
    
    img = plt.imread(jpg_path+f)
    original_h, original_w, _ = img.shape
    resize_w, resize_h = original_w, original_h
    img = cv2.resize(img, (resize_w, resize_h))
    print(img.shape)
    with open(json_path+f[:-4]+".json") as json_file:
        json_data = json.load(json_file)

    target_map = np.zeros((800,800),dtype=np.int32)
    annotation = json_data["annotations"]
    for item in annotation:
        obj_class = item["class"]
        # pos = item["label"]["data"]
        if obj_class == "traffic_lane":
            pos = item["data"]
            poly_points = np.array([([pt["x"]*resize_w/original_w, pt["y"]*resize_h/original_h]) for pt in pos]).astype(np.int32)
            plt.plot(poly_points[:,0],poly_points[:,1],color='blue')
            cv2.polylines(target_map, [poly_points], False, 1,10)
        if obj_class == "stop_line":
            pos = item["data"]
            poly_points = np.array([([pt["x"]*resize_w/original_w, pt["y"]*resize_h/original_h]) for pt in pos]).astype(np.int32)
            cv2.polylines(target_map, [poly_points], False, 2,10)
            plt.plot(poly_points[:,0],poly_points[:,1],color='red')
        if obj_class == "crosswalk":
            pos = item["data"]
            poly_points = np.array([([pt["x"]*resize_w/original_w, pt["y"]*resize_h/original_h]) for pt in pos]).astype(np.int32)
            cv2.fillPoly(target_map, [poly_points], 3)
            plt.plot(poly_points[:,0],poly_points[:,1],color='yellow')
            plt.plot([poly_points[0,0],poly_points[-1,0]],[poly_points[0,1],poly_points[-1,1]],color='yellow')
    
    plt.imshow(img)
    # plt.figure()
    # plt.imshow(target_map)
    plt.show()