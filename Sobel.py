import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)

def prepareimage(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    f = cv2.medianBlur(gray, 5)

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    clahe_img1 = clahe.apply(f)

    return clahe_img1

def find_rect1(mask_frame): 

    gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGRA2GRAY)

    ret1, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = [cv2.boundingRect(cnt) for cnt in contours]
    coord = []

    for rect in rectangles:
        coord.append(rect)
        cv2.rectangle(mask_frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

    # cv2.imshow('Rectangles', mask_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return coord

def Sobel_segment(clahe_img, cord, img1, params):
    ret1, bin_img = cv2.threshold(clahe_img, 100, 255, cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
  
    delta = 0
    ddepth = cv2.CV_16S
    
    grad_x = np.array([])
    grad_y = np.array([])

    grad_x = cv2.Sobel(bin_img, ddepth, 1,0, grad_x,
                       params[0], params[1], delta = delta, 
                       borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(bin_img, ddepth, 0,1, grad_y,
                       params[0], params[1], delta = delta, 
                       borderType = cv2.BORDER_DEFAULT)
    
    

    Abs_grad_x= cv2.convertScaleAbs(grad_x)
    Abs_grad_y= cv2.convertScaleAbs(grad_y)
    Sobel1 = cv2.addWeighted(Abs_grad_x, 0.5, Abs_grad_y, 0.5,0)
    mask = np.ones_like(img1)
    
    

    for i in cord:
        outputSobel = cv2.dilate(Sobel1, kernel, iterations=2)
        contours, hierarchy = cv2.findContours( outputSobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        for contour in contours:
            if not cv2.isContourConvex(contour):
                epsilon = 0.0001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.fillPoly(mask[i[1]:i[1] + i[3], i[0]:i[0] + i[2]], [approx], (0, 0, 255))
                #cv2.drawContours(img1[i[1]:i[1] + i[3], i[0]:i[0] + i[2]], [approx], 0, (0, 0, 255), 2)
    return mask

def poisk1(img, h, dispersia, b):  

    disp = []  
    disp_cord = []

    for y in range(img[h[1]-b:h[1] + h[3]+b, h[0]-b:h[0] + h[2]+b].shape[0]):
        if y + h[3] > img[h[1]-b:h[1] + h[3]+b, h[0]-b:h[0] + h[2]+b].shape[0]:  # проверка на выход за край по вертикали
            break
        else:
            for x in range(img[h[1]-b:h[1] + h[3]+b, h[0]-b:h[0] + h[2]+b].shape[1]):
                if x + h[2] > img[h[1]-b:h[1] + h[3]+b, h[0]-b:h[0] + h[2]+b].shape[1]:  # проверка на выход за край по горизонтали
                    break
                else:
                    disp.append(np.var(img[y+h[1]-b:y+h[1]-b + h[3], x+h[0]-b:x+h[2]+h[0]-b]))
                    disp_cord.append((x, y))

    result_dispers, result_index = find_closest_value(disp, dispersia)

    return disp_cord[result_index][0]-b+h[0], disp_cord[result_index][1]-b+h[1], result_dispers

def find_closest_value(numbers, target):
    closest_value = min(numbers, key=lambda x: abs(x - target))
    closest_index = numbers.index(closest_value)
    return closest_value, closest_index

def video_process_sobel(frames, params, masked_first_frame):
    video = []

    cord = find_rect1(masked_first_frame)
    img1 = frames[0].copy()  # считываю первый кадр
    clahe_img = prepareimage(img1)
    dispers1 = np.var(
        img1[cord[0][1]:cord[0][1] + cord[0][3], cord[0][0]:cord[0][0] + cord[0][2]])  # дисперсия для первой области
    dispers2 = np.var(
        img1[cord[1][1]:cord[1][1] + cord[1][3], cord[1][0]:cord[1][0] + cord[1][2]])  # дисперсия для второй области

    for i in range(0, len(frames)):
        img1 = frames[i].copy()
        clahe_img = prepareimage(img1)  # тут делаем эквализацию и тд
        x1, y1, dispers1 = poisk1(clahe_img, cord[0], dispers1, 5)  # получаем положение окна и значение дисперсии для области 1
        x2, y2, dispers2 = poisk1(clahe_img, cord[1], dispers2, 5)  # получаем положение окна и значение дисперсии для области 2

        cord = [(x1, y1, cord[0][2], cord[0][3]), (x2, y2, cord[1][2], cord[1][3])]

        mask = Sobel_segment(clahe_img, cord, img1, params)
        video.append(mask)
    return video
