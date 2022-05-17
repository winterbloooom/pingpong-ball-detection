import numpy as np
import cv2
from glob import glob
import os
import json


def get_intrinsic_info(image_path=None, CHESSBOARD_WIDTH=9, CHESSBOARD_HEIGHT=6):
    image_list = glob(image_path + "/*.jpg")

    CHESSBOARD_WIDTH = 9
    CHESSBOARD_HEIGHT = 6
    SQUARE_SIZE = 0.025  # meter
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    visualize = False

    obj_point = np.zeros((CHESSBOARD_HEIGHT * CHESSBOARD_WIDTH, 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT].T.reshape(-1, 2) * SQUARE_SIZE

    object_points = list()  # 3d point in real world space
    image_points = list()  # 2d points in image plane.

    for image_file in image_list:
        img = cv2.imread(image_file, cv2.IMREAD_ANYCOLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None)

        if ret:
            object_points.append(obj_point)
            image_points.append(corners)

            # visualize
            if visualize:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(img, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners2, ret)
                cv2.imshow("draw chessboard corners", img)
                cv2.waitKey(0)

    # get calibration information
    ret, camera_matrix, distcoffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )

    for rvec, tvec, op, ip in zip(rvecs, tvecs, object_points, image_points):
        imagePoints, jacobian = cv2.projectPoints(op, rvec, tvec, camera_matrix, distcoffs)

        # intrinsic이 잘 되었나를 확인하기 위해 재투영을 하여 오류를 계산함
        # sub = 0
        # for det, proj in zip(ip, imagePoints):
        #     sub += sum((det - proj)[0])
        # print(sub)

    intrinsic_info = dict()

    intrinsic_info['fx'] = camera_matrix[0, 0]
    intrinsic_info['fy'] = camera_matrix[1, 1]
    intrinsic_info['cx'] = camera_matrix[0, 2]
    intrinsic_info['cy'] = camera_matrix[1, 2]

    intrinsic = dict()

    intrinsic["intrinsic"] = intrinsic_info

    distcoffs = distcoffs.tolist()
    print(type(distcoffs))
    
    extrinsic_info = dict()

    extrinsic_info['rvecs'] = rvecs
    extrinsic_info['tvecs'] = tvecs
    extrinsic_info['distortion coff'] = distcoffs

    extrinsic = dict()

    extrinsic["intrinsic"] = extrinsic_info

    with open("./config/intrinsic.json", "w") as config_file:
        json.dump(intrinsic, config_file, indent=4)
        json.dump(extrinsic, config_file, indent=4)

if __name__ == "__main__":
    get_intrinsic_info("./img")