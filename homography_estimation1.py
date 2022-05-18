# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import calibration_parser
import distance_tool1 as d_tool
import pickle       # Homography Matrix 저장 포맷

if __name__ == "__main__":
    calibration_json_filepath = os.path.abspath("calibration.json")             # intrinsic matrix 포함 json 파일
    camera_matrix=calibration_parser.read_json_file(calibration_json_filepath)

    # Homography Matrix을 구하기 위한 Image
    train_image = cv2.imread(os.path.join(".\\source","images","img_1.jpg"),cv2.IMREAD_COLOR)
    train_image = cv2.cvtColor(train_image ,cv2.COLOR_BGR2RGB)

    # Test Image
    test_image = cv2.imread(os.path.join(".\\source","images","test_img_1.jpg"),cv2.IMREAD_COLOR)
    test_image =cv2.cvtColor(test_image ,cv2.COLOR_BGR2RGB)

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
    
    # Homography Matrix를 구하고 싶다면 True, 이미 구해져 있어 Test가 필요하다면 False
    get_homo_matrix = False

    if get_homo_matrix:
        # Homography Matrix 추출에 필요한 image 좌표(Chessboard와 탁구공 활용)
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
        # Homography Matrix 추출에 필요한 obj point(Chessboard의 맨 앞 중앙을 (0,0,0) 기준점으로 선정)
        # X Y Z, X->down, Z-> forward, Y->Left & Right
        object_points =np.array([
            [0, -12.5, 0.0],    
            [0, 12.5, 0.0],
            [0, -2.5, 0.0], 
            [0, 2.5, 0.0], 
            [0, -12.5, 17.5], 
            [0, 12.5, 17.5],
            [0, -2.5, 17.5],
            [0, 2.5, 17.5],
            [0, -11.2, 1.3],
            [0, 11.2, 1.3],
        ],dtype=np.float32)
        DATA_SIZE = len(image_points)
        #object point
        # X : forward , Y : Left & Right , Z : 1
        homo_object_point = np.append(object_points[:,2:3], object_points[:,1:2],axis=1)
        homo_object_point = np.append(homo_object_point, np.ones([1,DATA_SIZE]).T, axis=1)

        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None, flags = cv2.SOLVEPNP_EPNP)
        train_image = cv2. drawFrameAxes(train_image, camera_matrix, None, rvec, tvec, 5,7)
        homography, _ = cv2.findHomography(image_points, homo_object_point)
        with open("homography.pickle","wb") as fw:
            pickle.dump(homography, fw)

        image = d_tool.draw_distance(train_image, homography, image_points)

    else:
        with open("homography.pickle","rb") as fr:
            homography = pickle.load(fr)

        test_image_points=np.array([
            [156, 252],
            [425, 255],
        ],dtype=np.float32)

        image = d_tool.draw_distance(test_image, homography, test_image_points)

    plt.imshow(image)
    plt.show()
