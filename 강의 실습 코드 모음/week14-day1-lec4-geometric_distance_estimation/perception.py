import sys
import json
import cv2
from cv2 import undistort
import matplotlib.pyplot as plt
import os
import numpy as np

json_file_path = os.path.join("data", "000076.json")
image_file_path = os.path.join("data", "1616343619200.jpg")

window_name = "Perception"

with open(json_file_path, "r") as json_file:
    labeling_info = json.load(json_file)

image = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)

camera_matrix = np.asarray(labeling_info["calib"]["cam01"]["cam_intrinsic"], dtype=np.float32)
    # [[958.8315, -0.1807, 934.9871],
    #  [0.0, 962.3082, 519.1207],
    #  [0.0, 0.0, 1.0]]
dist_coeff = np.asarray(labeling_info["calib"]["cam01"]["distortion"], dtype=np.float32)
    # [-0.3121, 0.1024, 0.00032953, -0.00039793, -0.0158]
undist_image = cv2.undistort(image, camera_matrix, dist_coeff, None, None)
    # 받은 이미지를 보정한 이미지

labeling = labeling_info["frames"][0]["annos"]
    # 키가 names, boxes_3d, boxes_2d, pose인 딕셔너리 형태
class_names = labeling["names"]
    # "Car", "Car", "Pedestrian", ...
boxes_2d = labeling["boxes_2d"]["cam01"]
    # [좌표 4개], [좌표4개], ...

CAMERA_HEIGHT = 1.3
    # 지면으로부터 카메라의 높이

# distance = f * height / img(y)
# 종/횡 방향으로 분리된 거리가 아닌, 직선거리
# FOV 정보를 알면 -> 종/횡 분리가 가능하다.

index = 0
for class_name, bbox in zip(class_names, boxes_2d):
    xmin, ymin, xmax, ymax = bbox
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
        continue    # invalid한 정보는 -1로 표기되어 있으므로 그런 경우는 continue


    width = xmax - xmin
    height = ymax - ymin

    # Normalized Image Plane
    y_norm = (ymax - camera_matrix[1][2]) / camera_matrix[1][1]

    distance = 1 * CAMERA_HEIGHT / y_norm
        # normalized image plane에서는 focal length를 1로 // Z값이 1

    print(int(distance))    # 객체까지 거리

    cv2.rectangle(undist_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3) 
        # bounding box를 이미지 위에 표기
    cv2.putText(undist_image, f"{index}-{class_name}-{int(distance)}", (xmin, ymin+25), 1, 2, (255, 255, 0), 2)
        # 거리와 클래스 정보를 표기
    index += 1

    display_image = cv2.cvtColor(undist_image, cv2.COLOR_BGR2RGB)
        # 화면에 보이기 위해 다시 RBG로 변경
    plt.imshow(display_image)
    plt.show()