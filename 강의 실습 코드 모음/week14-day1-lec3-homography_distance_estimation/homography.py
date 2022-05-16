import cv2
import json
import os
import numpy as np

import matplotlib.pyplot as plt

import calibration_parser


if __name__ == "__main__":
    calibration_json_filepath = os.path.join("image", "cologne_000065_000019_camera.json")
    camera_matrix = calibration_parser.read_json_file(calibration_json_filepath)
    image = cv2.imread(os.path.join("image", "cologne_000065_000019_leftImg8bit.png"), cv2.IMREAD_ANYCOLOR)

    # extrinsic -> homography src, dst
    # prior dst -> image coordinate
    # present dst -> vehicle coordinate (=camera coordinate)

    # lane (inner) width -> 2.5m, lane width -> 0.25m
    # lane lenght -> 2.0m
    # lane interval -> 2.0m

    """
    Extrinsic Calibration for Ground Plane
    [0, 1]
    464, 833 -> 0.0, 0.0, 0.0
    1639, 833 -> 0.0, 3.0, 0.0

    [2, 3]
    638, 709 -> 0.0, 0.0, 2.0
    1467, 709 -> 0.0, 3.0, 2.0

    [4, 5]
    742, 643 -> 0.0, 0.0, 4.0
    1361, 643 -> 0.0, 3.0, 4.0

    [6, 7]
    797, 605 -> 0.0, 0.0, 6.0
    1310, 605 -> 0.0, 3.0, 6.0
    """
    image_points = np.array([
        [464, 833],
        [1639, 833],
        [638, 709],
        [1467, 709],
        [742, 643],
        [1361, 643],
        [797, 605],
        [1310, 605]
    ], dtype=np.float32)

    # X Y Z, X -> down, Z -> forward, Y -> Right
    # 실측이 중요하다.
    object_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 3.7, 0.0],
        [0.0, 0.0, 6.0],
        [0.0, 3.7, 6.0],
        [0.0, 0.0, 12.0],
        [0.0, 3.6, 12.0],
        [0.0, 0.0, 18.0],
        [0.0, 3.7, 18.0]
    ], dtype=np.float32)

    DATA_SIZE = 8
    homo_object_point = np.append(object_points[:,2:3], -object_points[:,1:2], axis=1)
    homo_object_point = np.append(homo_object_point, np.ones([1, DATA_SIZE]).T, axis=1)

    print(homo_object_point)

    # object point
    # X: forward, Y: left, Z: 1

    # 지면에 대해서 위치와 자세 추정이 가능하다면,
    # 임의의 포인트를 생성하여 이미지에 투영할수있다.
    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distCoeffs=None, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)

    # 잘 맞지 않는다.
    # 왜냐하면, 이미지 좌표와 실제 오브젝트와의 관계가 부정확하기 때문
    # 실제 측정을 통해 개선이 가능하다.
    image = cv2.drawFrameAxes(image, camera_matrix, None, rvec, tvec, 1, 5)

    # proj_image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)

    homography, _ = cv2.findHomography(image_points, homo_object_point)
    # print(proj_image_points.shape)
    
    # (u, v) -> (u, v, 1)
    appned_image_points = np.append(image_points.reshape(8, 2), np.ones([1, DATA_SIZE]).T, axis=1)
    # print(homography.shape)
    for image_point in appned_image_points:
        # estimation point(object_point) -> homography * src(image_point)
        estimation_distance = np.dot(homography, image_point)

        x = estimation_distance[0]
        y = estimation_distance[1]
        z = estimation_distance[2]

        print(x/z, y/z, z/z)

    # 여기서 중요한 점
