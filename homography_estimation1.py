"""
순서
1. 실측을 통한 obj point , img point 설정
2. intrinsic, distort coefficients 정보 불러오기
3. undistort
4. obj point -> homogeneous 좌표계로 변환
5. solvePnP, drawFrameAxes로 좌표축 확인, 0,0,0 좌표로 그려져야 맞는것
6. findHomography 로 perspective matrix 추출
7. image point -> homogeneous 좌표계로 변환
8. homography와 image point를 내적하면 x,y,z 가 추출됨.
"""

import cv2
import json
import os, sys
import glob

import numpy as np
import matplotlib.pyplot as plt
import calibration_parser
import distance_tool1 as d_tool

#from cv2 import calibrateCamera

if __name__ == "__main__":
    calibration_json_filepath = os.path.abspath("calibration.json")
    camera_matrix=calibration_parser.read_json_file(calibration_json_filepath)

    # change dirpath
    image = cv2.imread(os.path.join(".\\source","images","img_1.jpg"),cv2.IMREAD_COLOR)
    image =cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)

    #FIXME
    t_image = cv2.imread(os.path.join(".\\source","images","test_img_1.jpg"),cv2.IMREAD_COLOR)
    t_image =cv2.cvtColor(t_image ,cv2.COLOR_BGR2RGB)

    """
    Extrinsic Calibration img_1.jpg(chessboard and ball)
    [0, 1]
    186, 395
    479, 395
    16.5, 0.0, 0.0
    16.5, 25.0, 0,0

    [2, 3]
    298, 395
    356, 395
    16.5, 10.0, 0.0
    16.5, 15.0, 0.0

    [4, 5]
    232, 326
    420, 326
    16.5, 0.0, 17.5
    16.5, 25.0, 17.5

    [6, 7]
    306, 326
    344, 326
    16.5, 10.0, 17.5
    16.5, 15.0, 17.5

    [8, 9]
    202, 388
    456, 388
    16.5, 1.3, 1.3
    16.5, 23.7, 1.3
    """

    image_points =np.array([
        [186, 395],
        [479, 395],

        [298, 395], 
        [356, 395], 

        [232, 326], 
        [420, 326],

        [306, 326],
        [344, 326],

        [202, 388],
        [456, 388],
    ],dtype=np.float32)

    #FIXME
    test_image_points=np.array([
        [156, 252],
        [425, 255],
    ],dtype=np.float32)

    # X Y Z, X->down, Z-> forward, Y->Right
    x = 0.0 # 16.5
    object_points =np.array([
        [x, -12.5, 0.0],    
        [x, 12.5, 0.0],
        [x, -2.5, 0.0], 
        [x, 2.5, 0.0], 
        [x, -12.5, 17.5], 
        [x, 12.5, 17.5],
        [x, -2.5, 17.5],
        [x, 2.5, 17.5],
        [x, -11.2, 1.3],
        [x, 11.2, 1.3],
    ],dtype=np.float32)
 
    DATA_SIZE = 10
    homo_object_point = np.append(object_points[:,2:3], -object_points[:,1:2],axis=1)
    homo_object_point = np.append(homo_object_point, np.ones([1,DATA_SIZE]).T, axis=1)

    #object point
    # X : forward , Y : Left+, Z : 1

    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None, flags = cv2.SOLVEPNP_EPNP)
    image = cv2. drawFrameAxes(image, camera_matrix, None, rvec, tvec, 5,7)
    homography, _ = cv2.findHomography(image_points, homo_object_point)
    
    # change t_image->image
    t_image = d_tool.draw_distance(t_image, homography, test_image_points)

    plt.imshow(t_image)
    plt.show()
