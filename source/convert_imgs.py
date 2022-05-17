#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys, os
import cv2
import glob
from cv2 import undistort

def undistort_imgs(camera_matrix, dist_coeffs):
    raw_path_list = glob.glob("/home/lumos/tricat/src/pingpong-ball-detection/source/images/*.jpg")
        # TODO 상대경로로 하니 잘 안 나옴. 자이카 적용 시 경로 바꾸기
    output_path = "/home/lumos/tricat/src/pingpong-ball-detection/source/output"
        # TODO 자이카 적용 시 경로 바꾸기

    if not os.path.isdir(output_path):
        os.makedirs(output_path)    # 왜곡을 보정한 사진을 저장할 폴더 생성

    for image_path in raw_path_list:
        file_name = list(image_path.split('/'))[-1]
            # ~~.jpg 형식의 파일 이름만 떼어내기
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # 원본 이미지 불러오기
        # cv2.imshow("raw", image)

        image_undist = cv2.undistort(image, camera_matrix, dist_coeffs, None)
            # 원본 이미지를 카메라 캘리브레이션 값들을 이용해 왜곡 보정하기
        file_name = output_path + '/' + file_name   
            # 출력할 파일 경로 생성
        # print(file_name)
        cv2.imwrite(file_name, image_undist, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
            # 왜곡 보정한 파일 내보내기

        # if cv2.waitKey(0) == 27:
        #     continue