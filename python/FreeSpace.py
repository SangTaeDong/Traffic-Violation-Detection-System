import cv2
import numpy as np

# 직선의 방정식 구해주는 함수
def draw_linear_graph(x1,y1,x2,y2):
    
    if x2-x1 == 0:
        return 0,x1

    m = (y2 - y1) / (x2 - x1)
    n = y1 - (m * x1)
    
#     plt.plot(x,y)
    
    return m,n

# 두직선에서 교점 구하는 함수
def intersection_point(a1,a2):
    m1,n1,m2,n2 = a1[0],a1[1],a2[0],a2[1]
    x = (n2-n1) / (m1-m2)
    y = m1*x+n1
    
    return x,y


# free_space 가장밑 부분 구하는 함수
# 직선중 가장 낮은곳 찾는것
def bottom_min (arr):
    min_pos = 0
    for pos in arr:
        x1,y1,x2,y2 = pos
        
        if y1 > min_pos:
            min_pos = y1
        if y2 > min_pos:
            min_pos = y2
    
    return min_pos

#마지막 오른쪽 왼쪽 좌표들 구해주기
def left_right (arr,bottom,top,width):
    close_left = 0
    index_left = -1
    close_right = width 
    index_right = -1
    mid = width//2
    
    
    ## 중간에서 가장 가까운 오른쪽거 왼쪽거 찾아줌
    for idx,(x1,y1,x2,y2) in enumerate(arr):
        #중간보다 왼쪽에 있을 경우
        if x1 < mid and x2 < mid:
            temp = max(x1,x2)
            if close_left < temp:
                close_left = temp
                index_left = idx
  
        #중간보다 오른쪽에 있을경우 
        elif x1 > mid and x2 > mid:
            temp = min(x1,x2)
            if close_right > temp:
                close_right = temp
                index_right = idx
                
    
    lm, _ = draw_linear_graph(*arr[index_left])
    ln = bottom
    
    rm, _ = draw_linear_graph (*arr[index_right])
    rn = bottom - width*rm    
    if lm * rm > 0:
        return False,False
    
    
    
    #return [(0,bottom,int((top-ln)//lm),top), (int((bottom-rn)//rm),bottom,int((top-rn)//rm) ,top)]
    return [(0,bottom,int((top-ln)//lm),top), (int((bottom-rn)//rm),bottom,int((top-rn)//rm) ,top)]
            
    
def left_right_ver2 (arr,bottom,top,width):
    close_left = 0
    index_left = -1
    close_right = width 
    index_right = -1
    mid = width//2
    
    
    ## 중간에서 가장 가까운 오른쪽거 왼쪽거 찾아줌
    for idx,(x1,y1,x2,y2) in enumerate(arr):

        #중간보다 왼쪽에 있을 경우
        if x1 < mid and x2 < mid:
            temp = max(x1,x2)
            if close_left < temp:
                close_left = temp
                index_left = idx
  
        #중간보다 오른쪽에 있을경우 
        elif x1 > mid and x2 > mid:
            temp = min(x1,x2)
            if close_right > temp:
                close_right = temp
                index_right = idx
                
    
    lm, _ = draw_linear_graph(*arr[index_left])
    ln = bottom
    
    rm, _ = draw_linear_graph (*arr[index_right])
    rn = bottom - width*rm    
    print(rm,lm)
    if lm * rm > 0 or abs(abs(lm) - abs(rm)) >= 0.1:
        return False,False
    
    
    
    #return [(0,bottom,int((top-ln)//lm),top), (int((bottom-rn)//rm),bottom,int((top-rn)//rm) ,top)]
    return [(0,bottom,int((top-ln)//lm),top), (int((bottom-rn)//rm),bottom,int((top-rn)//rm) ,top)]
            
# def stop_line_volation(x11,y11,x12,y12,x21,y21,x22,y22,ty):
    
#     if ty == 'Vehicle_Car':
#         acc = 0.2
#     else:
#         acc = 0.15
    
    
#     m1,n1 = draw_linear_graph(x11,y11,x12,y12)
#     #     m2,n2 = draw_linear_graph((x21+x22)//2,y21,(x21+x22)//2,y22)
    
#     x = m1*(x21+x22) + n1
#     y = m1*x+n1

    
#     if (y11 + y12) // 2 > y22:
#         return True
    
#     if abs(y22-y)/abs(y21-y) < acc:
#         return True
#     else:
#         return False