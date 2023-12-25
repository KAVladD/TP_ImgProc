# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:07:13 2023

@author: Полина
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)



def prepare(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    filtered = cv2.medianBlur(gray, 5)

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    clahe_img = clahe.apply(filtered)

    return clahe_img


def find_rect(masked_frame): 

    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGRA2GRAY)

    ret1, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = [cv2.boundingRect(cnt) for cnt in contours]
    coord_rectangles = []

    for rect in rectangles:
        coord_rectangles.append(rect)
        cv2.rectangle(masked_frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

    cv2.imshow('Rectangles', masked_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return coord_rectangles


def Sobel_segment(clahe_img, cord, img1):
    ret1, bin_img = cv2.threshold(clahe_img, 100, 255, cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(bin_img, ddepth, 1,0, ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(bin_img, ddepth, 0,1, ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)

    Abs_grad_x= cv2.convertScaleAbs(grad_x)
    Abs_grad_y= cv2.convertScaleAbs(grad_y)
    Sobel1 = cv2.addWeighted(Abs_grad_x, 0.5, Abs_grad_x, 0.5,0)



    for i in cord:
        outputSobel = cv2.dilate(Sobel1, kernel, iterations=2)
        contours, hierarchy = cv2.findContours( outputSobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        for contour in contours:
            if not cv2.isContourConvex(contour):
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.fillPoly(img1[i[1]:i[1] + i[3], i[0]:i[0] + i[2]], [approx], (0, 0, 255))
                #cv2.drawContours(img1[i[1]:i[1] + i[3], i[0]:i[0] + i[2]], [approx], 0, (0, 0, 255), 2)


    return img1


def poisk(img, h, dispersia, b):  

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

path_to_frames = 'C:\\anaconda 3\\test2\\2\\' 
masked_first_frame = cv2.imread('C:\\anaconda 3\\test2\\masked\\1.png') 
path_to_folder = 'C:\\anaconda 3\\test2\\Sobel\\' 

cord = find_rect(masked_first_frame)  


frames = os.listdir(path_to_frames)

img1 = cv2.imread(path_to_frames + 'frame0.jpg')  
clahe_img = prepare(img1)
dispers1 = np.var(img1[cord[0][1]:cord[0][1] + cord[0][3], cord[0][0]:cord[0][0] + cord[0][2]])  
dispers2 = np.var(img1[cord[1][1]:cord[1][1] + cord[1][3], cord[1][0]:cord[1][0] + cord[1][2]])  

for i in range(1, len(frames)):

    img1 = cv2.imread(path_to_frames + 'frame'+str(i)+'.jpg')
    clahe_img = prepare(img1) 
    x1, y1, dispers1 = poisk(clahe_img, cord[0], dispers1, 5) 
    x2, y2, dispers2 = poisk(clahe_img, cord[1], dispers2, 5) 

    cord = [(x1, y1, cord[0][2], cord[0][3]), (x2, y2, cord[1][2], cord[1][3] )]


    Sobel = Sobel_segment(clahe_img, cord, img1)
    cv2.imwrite(path_to_folder+ 'frame'+str(i)+'.jpg', Sobel)

# тут видео собираю

output_video = 'C:\\anaconda 3\\test2\\Sobel\\result12.mp4'

first_image = cv2.imread(os.path.join(path_to_folder , os.listdir(path_to_folder )[0]))
height, width, _ = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for image_name in os.listdir(path_to_folder ):
    image_path = os.path.join(path_to_folder , image_name)
    image = cv2.imread(image_path)
    video_writer.write(image)

video_writer.release()