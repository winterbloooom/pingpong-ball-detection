#!/usr/bin/env python
#-*- coding:utf-8 -*-

import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import undistort

"""
Undistort raw images from xycar camera
"""

def undistort_imgs(raw_path_list, output_path, camera_matrix, dist_coeffs):
    """Undistort raw images and Save them

    Args
    ===
    raw_path_list (str) : path of raw images
    output_path (str) : output directory path to save undistorted images
    camera_matrix (np.ndarray)
    dist_coeffs (np.ndarray)
    """

    if not os.path.isdir(output_path):
        os.makedirs(output_path)    # 왜곡을 보정한 사진을 저장할 폴더 생성

    for image_path in raw_path_list:
        file_name = list(image_path.split('\\'))[-1]
            # ~~.jpg 형식의 파일 이름만 떼어내기
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # 원본 이미지 불러오기
        # cv2.imshow("raw", image)

        image_undist = cv2.undistort(image, camera_matrix, dist_coeffs, None)
            # 원본 이미지를 카메라 캘리브레이션 값들을 이용해 왜곡 보정하기
        file_name = output_path + '/' + file_name   
            # 출력할 파일 경로 생성
        print(file_name)
        cv2.imwrite(file_name, image_undist, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
            # 왜곡 보정한 파일 내보내기


def show_img(path):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    display_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(display_image)
    plt.show()


if __name__ =="__main__":
    raw_path_list = glob.glob("C:\\coding\\pingpong-ball-detection\\source\\img\\*.jpg")
    output_path = "C:\\coding\\pingpong-ball-detection\\source\\output"

    # distance coeff
    newrun_dist = np.array([[-0.3208765, 0.14017744,  0.00962414, -0.01470881, -0.03640567]])
    xycar_dist = np.array([[-0.303637,  0.070669,  0.002487,  -0.000932, 0.000000]])

    # intrinsic calib
    newrun_intrin = np.array([[383.42387256, 0.0, 335.70612309], [0.0, 382.93669554, 203.52649621], [0.0, 0.0, 1.0]])
    xycar_intrin = np.array([[344.774722, 0.000000, 312.944372], [0.000000, 346.412761, 207.474168], [0.000000, 0.000000, 1.000000]])
    
    undistort_imgs(raw_path_list, output_path, newrun_intrin, newrun_dist)

    # Show undistorted images
    # path = output_path + "\\img_1.jpg"
    # show_img(path)