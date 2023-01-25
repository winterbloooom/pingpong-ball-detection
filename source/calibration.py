import json
from glob import glob

import cv2
import numpy as np

"""
Get Intrinsic/Extrinsic calibration parameters using chessboard and Save in json file format
"""


def get_calibration_info(image_path=None, CHESSBOARD_WIDTH=9, CHESSBOARD_HEIGHT=6):
    """Get Camera intrinsic/extrinsic parameters with chessboard"""
    
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

    calibration = dict()

    intrinsic_info = dict()
    intrinsic_info['fx'] = camera_matrix[0, 0]
    intrinsic_info['fy'] = camera_matrix[1, 1]
    intrinsic_info['cx'] = camera_matrix[0, 2]
    intrinsic_info['cy'] = camera_matrix[1, 2]
    calibration["intrinsic"] = intrinsic_info
    
    extrinsic_info = dict()
    rvecs_list = [rvec.tolist() for rvec in rvecs]
    extrinsic_info['rvecs'] = rvecs_list
    tvecs_list = [tvec.tolist() for tvec in tvecs]
    extrinsic_info['tvecs'] = tvecs_list
    distcoffs = distcoffs.tolist()
    extrinsic_info['distortion_coff'] = distcoffs
    calibration["extrinsic"] = extrinsic_info

    with open("../calibration.json", "w") as config_file:
        json.dump(calibration, config_file, indent=4)


if __name__ == "__main__":
    get_calibration_info("../chessboard")