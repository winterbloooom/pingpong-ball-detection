# Python Package import
import numpy as np
import cv2 as cv
import glob
import os

# Part 1 Camera intrinsic calibration for undistort
# Step A. Configure initial variables
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

CHESSBOARD_WIDTH = 9
CHESSBOARD_HEIGHT = 6
SQUARE_SIZE = 0.025 # 2.5cm -> 0.025m

objp = np.zeros((CHESSBOARD_HEIGHT * CHESSBOARD_WIDTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT].T.reshape(-1, 2) * SQUARE_SIZE
print(objp)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
image_points = []  # 2d points in image plane.

# / -> linux/ubuntu
# \ -> windows
images = glob.glob(os.path.join("images", "left*.jpg"))
print(f"image list:\n{images}")

# Step B. Find chessboard corners for every image
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        object_points.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners)
        # Draw and display the corners
        # cv.drawChessboardCorners(img, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners2, ret)
        # cv.imshow("Draw Chessboard Corners", img)
        # cv.waitKey(800)


# Step C. Get camera calibration information
ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)

print("Camera Calibration\n", camera_matrix)
print(gray.shape[::-1])

# camera_matrix -> camera intrinsic calibration
# dist -> camera distortion coefficientes

# Part 2. Find homography matrix
# Step A. Select images
img1 = cv.imread(os.path.join("images", "left01.jpg"))
ret, corners1 = cv.findChessboardCorners(img1, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT))
corners1 = corners1.reshape(-1, 2)

img2 = cv.imread(os.path.join("images", "right13.jpg"))
ret, corners2 = cv.findChessboardCorners(img2, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT))
corners2 = corners2.reshape(-1, 2)

# for corn1, corn2 in zip(corners1, corners2):
#     print(corn1, corn2)

## Homography
# src -> dst point transformation -> find matrix

# Step B. Find homography
homography, status = cv.findHomography(corners1, corners2, cv.RANSAC)
print(homography)

# Step C. Display Result
img_draw_matches = cv.hconcat([img1, img2])
for i in range(len(corners1)):
    pt1 = np.array([corners1[i][0], corners1[i][1], 1])
    pt1 = pt1.reshape(3, 1)
    pt2 = np.dot(homography, pt1)
    pt2 = pt2 / pt2[2]
    end = (int(img1.shape[1] + pt2[0]), int(pt2[1]))
    start = (int(pt1[0][0]), int(pt1[1][0]))

    color = list(np.random.choice(range(256), size=3))
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv.line(img_draw_matches, start, end, tuple(color), 5)

cv.namedWindow("Draw Matches", cv.WINDOW_NORMAL)
cv.imshow("Draw Matches", img_draw_matches)
cv.imwrite("draw.png", img_draw_matches)
# cv.waitKey(0)

dst_image = cv.warpPerspective(img1, homography, (img1.shape[1], img1.shape[0]))
img_draw_transform = cv.hconcat([img1, dst_image])
cv.namedWindow("Display Transform", cv.WINDOW_NORMAL)
cv.imshow("Display Transform", img_draw_transform)
cv.imwrite("trans.png", img_draw_transform)

cv.waitKey(0)
