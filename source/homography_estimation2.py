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

import os, json
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import homogeneity_completeness_v_measure

PLT_VIEW = False
visualize = False
testing = False

camera_height = -16.5

def estimation_distance():
    ######## step 2. intrinsic, distort coefficients information
    calibration_jsonfile_path = os.path.join("./calibration.json")
    with open(calibration_jsonfile_path, 'r') as calibration_file:
        calibration_info = json.load(calibration_file)
        intrinsic = calibration_info['intrinsic']
        fx, fy, cx, cy = intrinsic['fx'], intrinsic['fy'], intrinsic['cx'], intrinsic['cy']
        camera_matrix = np.array([
                                [fx,      0.00000,      cx],
                                [0.00000,      fy,      cy],
                                [0.00000, 0.00000, 1.00000],
                                ], dtype=np.float64)
        dist_coeff = np.array([calibration_info['extrinsic']['distortion_coff']],
                                dtype=np.float64)


    img_file_path = "./source/images/img_1.jpg"
    img = cv2.imread(img_file_path)
    print(img.shape)

    ######## step 3. undistort
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeff, None, None, (img.shape[1], img.shape[0]), 5)
    undistort_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # measurement
    if PLT_VIEW:
        # both_img = cv2.hconcat([undistort_img, img])
        plt_img = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2RGB)
        plt.imshow(plt_img)
        plt.show()

    """
    높이 16.5cm, 가로 25cm, 세로 30cm
    """

    img_points = np.array([
                        [170, 426], # bottom left
                        [459, 426], # bottom right
                        [193, 409], # second left
                        [433, 409], # second right
                        [206, 380], # third left
                        [416, 380], # third right
                        [216, 366], # top left
                        [403, 366], # top right
                        ], dtype=np.float32)


    obj_points = np.array([
                        [camera_height, 00.0, 00.0], # bottom left
                        [camera_height, 25.0, 00.0], # bottom right
                        [camera_height, 00.0, 10.0], # second left
                        [camera_height, 25.0, 10.0], # second right
                        [camera_height, 00.0, 20.0], # third left
                        [camera_height, 25.0, 20.0], # third right
                        [camera_height, 00.0, 30.0], # top left
                        [camera_height, 25.0, 30.0], # top right
                        ], dtype=np.float32)

    data_size = len(img_points)

    obj_points = obj_points / obj_points[0,0]
    homo_obj_points = cv2.hconcat([obj_points[:,1], obj_points[:,2], obj_points[:,0]])
    homo_obj_points[:,1] = 0 - homo_obj_points[:,1]
    
    print(homo_obj_points)

    if visualize:
        ######## get rotation , translation vector using obj points and img points
        _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, distCoeffs=None, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)
        undistort_img = cv2.drawFrameAxes(undistort_img, camera_matrix, distCoeffs=dist_coeff, rvec=rvec, tvec=tvec, length=2, thickness=5)

        ######## image points, object points 의 pair 쌍이 맞는지 재투영을 통해 확인
        proj_image_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, None)
        for proj_image_point, img_point in zip(proj_image_points, img_points):
            print(img_point, proj_image_point, "\n")

        ######## 카메라 축 확인
        img = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    homography, _ = cv2.findHomography(img_points, homo_obj_points)

    print(f"data_size {data_size}")

    return homography

if __name__ =="__main__":
    # get homography
    img = cv2.imread("img_1.jpg", cv2.IMREAD_ANYCOLOR)


    ### bbox example
    top_left     = (160,255) #(416,248)
    top_right    = (168,255) #(426,248)
    bottom_left  = (160,247) #(417,258)
    bottom_right = (168,247) #(427,258)

    center_point = (bottom_right[0] - bottom_left[0], bottom_left[1] - top_left[1])

    homography = estimation_distance()
    
    img_point = np.array([center_point[0],center_point[1], camera_height], dtype=np.float32)

    ######## inference
    estimation = np.dot(homography, img_point)
    x,y,z = estimation[0] ,estimation[1],estimation[2]
    distance = math.sqrt(x**2 + y**2 + z**2) 

    ######## visualize
    cv2.rectangle(img, (top_left), (bottom_right), (255,255,0), 2)
    cv2.putText(img, "distance : "+str(round(distance,2)) + "cm", top_left, 2, 0.5, (10,250,10),1)

    cv2.imshow("img", img)
    cv2.waitKey()

    print(f"distance {round(distance,2)}cm")    
