import cv2
import glob
import numpy as np
import time

DISPLAY_IMAGE = False

# Get Image Path List
image_path_list = glob.glob("images/*.jpg")

# Chessboard Config
BOARD_WIDTH = 9
BOARD_HEIGHT = 6
SQUARE_SIZE = 0.025

# Window-name Config
window_name = "Intrinsic Calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
    # OpneCV Color Space -> BGR
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(image_gray, pattern_size, flags)
    if ret == True:
        if DISPLAY_IMAGE:
            image_draw = cv2.drawChessboardCorners(image, pattern_size, corners, ret)
            for corner in corners:
                counter_text = str(counter)
                point = (int(corner[0][0]), int(corner[0][1]))
                cv2.putText(image_draw, counter_text, point, 2, 0.5, (0, 0, 255), 1)
                counter += 1

            counter = 0
            cv2.imshow(window_name, image_draw)
            cv2.waitKey(0)

        image_points.append(corners)

object_points = list()
print(np.shape(image_points))

# (13, 54, 1, 2)
# (image count, featuer count, list, image_point(u, v))
# ojbect_points
# (13, 54, 1, 3)
# (image count, featuer count, list, object_point(x, y, z))

"""
 forward: Z
 right: Y
 down: X
"""

BOARD_WIDTH = 9
BOARD_HEIGHT = 6

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
    object_points.append(object_point)


object_points = np.asarray(object_points, dtype=np.float32)

tmp_image = cv2.imread("images/left01.jpg", cv2.IMREAD_ANYCOLOR)
image_shape = np.shape(tmp_image)

image_height = image_shape[0]
image_width = image_shape[1]
image_size = (image_width, image_height)
print(image_size)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

print("=" * 20)
print(f"re-projection error\n {ret}\n")
print(f"camera matrix\n {camera_matrix}\n")
print(f"distortion coefficientes error\n {dist_coeffs}\n")
print(f"extrinsic for each image\n {len(rvecs)} {len(tvecs)}")
print("=" * 20)

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

"""
0, 0, 0 -> index 0
0, 0.025, 0 -> index 1
"""