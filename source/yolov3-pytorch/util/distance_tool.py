# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

def cal_answer(estim_distance):
    # 단위 : cm
    y_offset=15         # 카메라와 Chessboard와의 지면 직선 거리 

    distance_x = estim_distance[1]
    distance_y = estim_distance[0]+y_offset

    return [distance_x, distance_y]

def cal_distance(estim_distance):
    # 단위 : cm
    y_offset=30         # 카메라와 Chessboard와의 지면 직선 거리 
    z_offset=16.5      # 지면기준 카메라의 높이
    distance = math.sqrt((estim_distance[0]+y_offset)**2+abs(estim_distance[1])**2+z_offset**2)
    # 그림 설명 참조

    return distance

# 출력을 위한 Image, homography Matrix, 원하는 위치의 Point 좌표(x,y) / 실 적용시에는 감지된 BBOX의 중점
def draw_distance(image, homography,image_points):
    DATA_SIZE=len(image_points)
    
    #(u,v) -> (u,v,1), 카메라에 비춰진 이미지 동차좌표계
    append_image_points = np.append(image_points.reshape(DATA_SIZE,2), np.ones([1,DATA_SIZE]).T,axis=1)

    
    for image_point,append_image_point in zip(image_points,append_image_points):
        # estimation point(object_point) -> homography * src(image_point[u,v,1])
        estimation_distance = np.dot(homography, append_image_point)
        x= estimation_distance[0]
        y= estimation_distance[1]
        z= estimation_distance[2]
        distance = cal_distance([x/z,y/z,z/z])

        img_x,img_y = image_point
        cv2.putText(image, f"{int(distance)}cm", (int(img_x),int(img_y)-5), 1, 1, (255, 255, 0), 1)
        cv2.circle(image, (int(img_x),int(img_y)),1,(0,0,0),3)

    return image
def bbox2cxcy(bbox):
    bbox = bbox.numpy()
    xmin, ymin, xmax, ymax = bbox[:4]
    cx = (float(xmax)*(640/416)+float(xmin)*(640/416))/2
    cy = (float(ymax)*(480/416)+float(ymin)*(480/416))/2

    return np.array([[cx,cy]])

def homo_answer_list(homography, box_list,img_name):
    answer_list=[]
    image_points = np.empty((0,2))
    for bbox in box_list:
        image_points= np.append(image_points, bbox2cxcy(bbox),axis=0)
    DATA_SIZE=len(box_list)
    #(u,v) -> (u,v,1), 카메라에 비춰진 이미지 동차좌표계
    append_image_points = np.append(image_points.reshape(DATA_SIZE,2), np.ones([1,DATA_SIZE]).T,axis=1)

    distances = []
    for image_point,append_image_point in zip(image_points,append_image_points):
        # estimation point(object_point) -> homography * src(image_point[u,v,1])
        estimation_distance = np.dot(homography, append_image_point)
        x= estimation_distance[0]
        y= estimation_distance[1]
        z= estimation_distance[2]
        distance = cal_answer([x/z,y/z,z/z])
        # img_x,img_y = image_point
        # cv2.putText(image, f"{int(distance)}cm", (int(img_x),int(img_y)-5), 1, 1, (255, 255, 0), 1)
        # cv2.circle(image, (int(img_x),int(img_y)),1,(0,0,0),3)
        distances.append(distance)

    distances.sort()
    output_dist = [img_name] + sum(distances, [])    # 2차원 -> 1차원
    answer_list.append(output_dist)
    
    return answer_list

