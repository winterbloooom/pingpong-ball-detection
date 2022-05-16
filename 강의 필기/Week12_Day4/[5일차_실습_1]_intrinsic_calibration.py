from tkinter import image_names
import cv2
import glob
import numpy as np
import time

DISPLAY_IMAGE = True

# Get Image Path List
image_path_list = glob.glob("images/*.jpg")
# print(image_path_list)
# # ['images\\left01.jpg', 'images\\left02.jpg', 'images\\left03.jpg', 'images\\left04.jpg', 
# # 'images\\left05.jpg', 'images\\left06.jpg', 'images\\left07.jpg', 'images\\left08.jpg', 
# # 'images\\left09.jpg', 'images\\left11.jpg', 'images\\left12.jpg', 'images\\left13.jpg', 'images\\left14.jpg']

# Chessboard Config
BOARD_WIDTH = 9
BOARD_HEIGHT = 6
SQUARE_SIZE = 0.025

# Window-name Config
window_name = "Intrinsic Calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # WINDOW_NORMAL이면 창 크기 조절 가능

# Calibration Config
flags = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
    + cv2.CALIB_CB_FAST_CHECK
)
pattern_size = (BOARD_WIDTH, BOARD_HEIGHT)
counter = 0

image_points = list()

for image_path in image_path_list:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # 이미지 자체는 gray처럼 보이는데, 처리를 쉽게 하기 위해서는 RGB로 읽는 것이 좋아 IMREAD_COLOR로 함

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(image_gray, pattern_size, flags)
        # 체스보드에서 코너를 검출

    if ret == True:
        # print(corners)

        if DISPLAY_IMAGE:
            image_draw = cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                # 찾은 코더를 화면에 그림

            for corner in corners:
                counter_text = str(counter)
                point = (int(corner[0][0]), int(corner[0][1]))
                    # corner은 [[값], [값]] 식으로 들어가 있음
                    # 원래는 아래 putText에 corner을 넣어야 하는데 int으로 바꿔줘야 하니까
                cv2.putText(image_draw, counter_text, point, 2, 0.5, (0, 0, 255), 1)
                counter += 1

            counter = 0
            cv2.imshow(window_name, image_draw)
            cv2.waitKey(0)

        image_points.append(corners)

object_points = list()
# print(np.shape(image_points))

##################

# image_points
# (13, 54, 1, 2)
# (image count, featuer count(9*6), list, image_point(u, v))

#-> 아래에서 하려는 것

# ojbect_points
# (13, 54, 1, 3)
# (image count, featuer count, list, object_point(x, y, z))

"""
 forward: Z
 right: Y
 down: X
"""

for i in range(len(image_path_list)):
    object_point = list()
    height = 0
    for _ in range(0, BOARD_HEIGHT):
        # Loop Width -> 9
        width = 0
        for _ in range(0, BOARD_WIDTH):
            # Loop Height -> 6
            point = [[height, width, 0]]
            object_point.append(point)
            width += SQUARE_SIZE
        height += SQUARE_SIZE

        # object_point의 0번 인덱스에는 [0, 0, 0], 1번에는 [0, 0.025, 0], ... 행 하나 지나면 [0.025, 0, 0], ...
    object_points.append(object_point)

object_points = np.asarray(object_points, dtype=np.float32)

###################
# 카메라 캘리브레이션

tmp_image = cv2.imread("images/left01.jpg", cv2.IMREAD_ANYCOLOR)
image_shape = np.shape(tmp_image)
image_height = image_shape[0]
image_width = image_shape[1]
image_size = (image_width, image_height)
# print(image_size)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

print("=" * 20)
print(f"re-projection error\n {ret}\n")
print(f"camera matrix\n {camera_matrix}\n")
print(f"distortion coefficientes error\n {dist_coeffs}\n")
print(f"extrinsic for each image\n {len(rvecs)} {len(tvecs)}")
print("=" * 20)

for rvec, tvec, op, ip in zip(rvecs, tvecs, object_points, image_points):
    imagePoints, jacobian = cv2.projectPoints(op, rvec, tvec, camera_matrix, image_points)
    for det, proj in zip(ip, imagePoints):
        print(det, proj)
            # 탐지(det)한 점과 예측(proj)한 점을 출력.
            # 둘이 차이 없어야 완벽한 것
            # ret은 re-projection error인데, det와 proj의 차이처럼 calibration한 값을 가지고 다시 projection했을 때와 비교한 차이값임

###############################
#모든 이미지 보정. 시간 측정

start_time = time.process_time()
for image_path in image_path_list:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_undist = cv2.undistort(image, camera_matrix, dist_coeffs, None)
end_time = time.process_time()
print(end_time - start_time)

start_time = time.process_time()
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, None, image_size, cv2.CV_32FC1)
for image_path in image_path_list:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_undist = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
end_time = time.process_time()
print(end_time - start_time)

cv2.getOptimalNewCameraMatrix() #이거 해보기