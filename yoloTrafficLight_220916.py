import cv2
import copy
import numpy as np


def yoloTrafficLight(image, classes, yolo_model):
    img = np.copy(image)

    h,w = img.shape[:2]
    img = img[:h//2,600:w-600] ### 이미지 자르기

    height, width, channels = img.shape
    # blob = cv2.dnn.blobFromImage(img, 1.0 / 256, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(img, 1.0 / 256, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    
    layer_names = yolo_model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in yolo_model.getUnconnectedOutLayers()]

    yolo_model.setInput(blob)
    output3 = yolo_model.forward(out_layers)

    class_ids, confidences, boxes = [], [], []
    for output in output3:
        for vec85 in output:
            scores = vec85[5:]
            scores[1] *= 4
            scores[3:] *= 4
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
                w, h = int(vec85[2] * width), int(vec85[3] * height)
                x, y = int(centerx - w / 2), int(centery - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    TL = ""
    tl_acc = -1

    ### TrafficLight만 확인
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            text = str(classes[class_ids[i]])
            if 'Traffic' in text and tl_acc < confidences[i]:
                TL = text
                tl_acc = confidences[i]

    return TL