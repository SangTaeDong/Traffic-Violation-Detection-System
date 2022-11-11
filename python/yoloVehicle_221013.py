import cv2
from python.histo_compare import *
import copy


# import bunhopan

def yoloVehicle(img, y_dict, classes, yolo_model, cnt,isfirst=True):
    hide_img = np.copy(img)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1.0 / 256, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    layer_names = yolo_model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in yolo_model.getUnconnectedOutLayers()]

    yolo_model.setInput(blob)
    output3 = yolo_model.forward(out_layers)

    class_ids, confidences, boxes = [], [], []
    for output in output3:
        for vec85 in output:
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: # Before 0.7

                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
                w, h = int(vec85[2] * width), int(vec85[3] * height)
                x, y = int(centerx - w / 2), int(centery - h / 2)
                if w > h*1.8: # Before 2
                    continue
                    
                if 'Vehicle_Car' == classes[class_id]:
                    if (w * h) <= 4500: # +1500
                        continue
                if 'Vehicle_Bus' == classes[class_id]:
                    if (w * h) <= 8000: # +3500
                        continue
                if 'Vehicle_Unknown' == classes[class_id]:
                    if (w * h) <= 5250: # +1000
                        continue
                if 'Vehicle_Motorcycle' == classes[class_id]:
                    if (w * h) <= 750: # +1000
                        continue

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    temp_dict = {}
    #처음 돌리는 경우
    if isfirst == True:
        for i in range(len(boxes)):
            #검출된 객체들을 순회하면서
            if i in indexes:
                x, y, w, h = boxes[i]
                text = str(classes[class_ids[i]])
                #차량, 노면표시 검정색으로 지워주는 처리
                if text != 'RoadMark_StopLine':
                    cv2.rectangle(hide_img, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
                #차량 좌표들만 추가해줌
                if 'Vehicle' in text:
                    y_dict[cnt] = {'pos': [x, y, w, h],
                                   'type': text,
                                   'state': None}
                    cnt += 1
        temp_dict = copy.deepcopy(y_dict)
    else:
        cnt+=1

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                text = str(classes[class_ids[i]])
#                 print('text : ',text, 'pos : ',x,y,w,h)
                #차량, 노면표시 검정색으로 지워주는 처리
                if text != 'RoadMark_StopLine':
                    cv2.rectangle(hide_img, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
                #차량이라면
                if 'Vehicle' in text:
                    #원래 이미지의 차량 정보들을 순회하면서
                    for key, value in y_dict.items():
                        #좌표를 가져온뒤
                        value = value['pos']
                        dx, dy, dw, dh = value[0], value[1], value[2], value[3]
                        #이미지가 같은지 히스토그램으로 한번 분석하고
                        #비슷한 차량일수 있으니 x,y좌표를 비교 후 차의 절대값의 합이 50 이하일경우 적용 후 종료
                        if isSame(img[y:y + h, x:x + w], img[dy:dy + dh, dx:dx + dw]) and (abs(x-dx) + abs(y-dy)) <= 50:
                            temp_dict[key] = {'pos': [x, y, w, h],
                                              'type': text,
                                              'state': None}
                            del y_dict[key]
                            break
                    #다 순회했는데 같은걸 못찾았을경우 새로운 차량으로 생각하고
                    #새로 카운터해서 좌표에 넣어줌
                    else:
                        temp_dict[cnt] = {'pos': [x, y, w, h],
                                          'type': text,
                                          'state': None}
                        cnt += 1

    return temp_dict, hide_img, cnt
