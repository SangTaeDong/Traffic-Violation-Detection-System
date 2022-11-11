import cv2, numpy as np

def isSame(img1,img2):
    accuracy = 0.4 # 정확도  부정확 0 ~ 1 정확
    '''
    이미지 두개를 받아서 두 이미지의 유사도를 확인함
    '''
    
    # HSV 영역으로 변환
    try:
        hsv_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    except:
        return False
    # H,S 채널 히스토그램 계산
    hist_1 = cv2.calcHist([hsv_1], [0,1], None, [180,256], [0,180,0, 256])
    hist_2 = cv2.calcHist([hsv_2], [0,1], None, [180,256], [0,180,0, 256])
    # 정규화(0~1)
    hist_1 =cv2.normalize(hist_1, hist_1, 0, 1, cv2.NORM_MINMAX)
    hist_2 =cv2.normalize(hist_2, hist_2, 0, 1, cv2.NORM_MINMAX)
    
    #cv2.HISTCMP_CORREL
    #cv2.HISTCMP_CHISQR
    #cv2.HISTCMP_INTERSEC
    #cv2.HISTCMP_BHATTACHARYYA
    ret = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)
    
    # 0.5 <= 인자 바꿔주면 정확도 변경 낮을수록 정확도 낮아짐.
    if ret > accuracy :
        return True
    else:
        return False
    

