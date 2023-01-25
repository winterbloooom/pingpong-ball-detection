#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predict distances(depth) of bounding boxes of ping-pong balls with xycar camera.
Using projection method!

Notes:
    * What do we need before running this script?
        * height from ground to camera [m]
        * instrinsic calibration results (+distance coefficients) in json file format
        * width(height) of ping-pong balls
        * FOVs of camera
"""

import json
import math
import sys

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from yolov3_trt_ros.msg import BoundingBox, BoundingBoxes


class Detect:
    def __init__(self):
        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback, queue_size=1)
        self.bridge = CvBridge()
        self.image = None

        rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.bbox_callback, queue_size=1)
        self.bboxes = []
        self.distances = []

        self.camera_height = 0.165    # 자이카에서 카메라까지 높이
        self.json_path = "/home/nvidia/xycar_ws/src/newrun/yolo_trt_ros/src/calibration.json" # TODO 파일 경로 넣기
        self.camera_matrix, self.dist_coeff = self.parse_calibration(self.json_path)
        

    def parse_calibration(self, path):
        """Get intrinsic, extrinsic calibration result from json file
        
        Returns
        ===
        camera_matrix (np.ndarray): intrinsic camera matrix
        dist_coeff (np.ndarray): distance coefficient from extrinsic calibration
        """

        with open(path, "r",) as f:
            calibration_json = json.load(f)

        camera_matrix = self.parse_intrinsic_calibration(calibration_json["intrinsic"])
        dist_coeff = calibration_json["extrinsic"]["distortion_coff"][0]
        
        return camera_matrix, np.array(dist_coeff)
        

    def parse_intrinsic_calibration(self, intrinsic):
        fx = intrinsic["fx"]
        fy = intrinsic["fy"]
        cx = intrinsic["cx"]
        cy = intrinsic["cy"]
        camera_matrix = np.zeros([3, 3], dtype=np.float32)
        camera_matrix[0][0] = fx
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = fy
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1.0

        return camera_matrix


    def bbox_callback(self, msg):
        boxes = []
        for box in msg.bounding_boxes:
            new_box = BoundingBox()
            new_box.xmin = box.xmin * float(640) / 416
            new_box.xmax = box.xmax * float(640) / 416
            new_box.ymin = box.ymin * float(480) / 416
            new_box.ymax = box.ymax * float(480) / 416

            boxes.append(new_box)

        self.bboxes = boxes


    def img_callback(self, msg):
        image_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image = cv2.undistort(image_raw, self.camera_matrix, self.dist_coeff, None)


    def predict(self):
        """Get distances for each bounding boxes and Draw information on images"""

        distances = []
        for bbox in self.bboxes:
            dist = self.projection_method(bbox)
            distances.append(dist)    # projection 방식
        image_drawn = self.draw_on_image(distances)
        
        cv2.imshow("show", image_drawn)
        if cv2.waitKey(1)== 27:
            return


    def projection_method(self, bbox):
        """Get distance of single bounding box using projection method"""

        # camera info
        f = self.camera_matrix[0][0] # focal length
        FOV_H_half = 46.7
        FOV_V_half = 38

        # image info
        image_width_half = 320 
        image_height_half = 240

        # bounding box info
        bbox_width = abs(bbox.xmax - bbox.xmin) # [px]
        cx_from_mid = (bbox.xmax + bbox.xmin) / 2 - image_width_half
        cy_from_mid = (bbox.ymax + bbox.ymin) / 2 - image_height_half
        xz_azimuth = FOV_H_half * cx_from_mid / image_width_half
        yz_azimuth = (FOV_V_half * cy_from_mid) / image_height_half
        # print("xz_azimuth {} / yz_azimuth {}".format(xz_azimuth, yz_azimuth))
        
        # object info
        real_width = 0.04   # object width, [m]

        # distances
        Z = (real_width * f) / bbox_width # 종방향
        X = abs(Z * math.tan(math.radians(xz_azimuth))) # 횡방향
        Y = abs(Z * math.tan(math.radians(yz_azimuth)))  # TODO Wrong!
        # print("X {} Y {} Z {}".format(X, Y, Z))

        dist = math.sqrt(X ** 2 + Z ** 2)

        return dist


    # def projection_method_longitudinal(self):
    #     """
    #     종방향(카메라가 보는 방향)으로의 거리만 추정하는 방법.
    #     논문의 내용을 구현
    #     """
    #     undist_image = cv2.undistort(self.image, self.camera_matrix, self.dist_coeff, None, None)
    #         # 받은 이미지를 보정한 이미지

    #     index = 0
    #     for bbox in self.bboxes:
    #         xmin = bbox.xmin
    #         ymin = bbox.ymin
    #         xmax = bbox.xmax
    #         ymax = bbox.ymax
    #         cls = bbox.Class

    #         y_norm = (ymax - self.camera_matrix[1][2]) / self.camera_matrix[1][1]
    #             # Normalized Image Plane

    #         distance = 1 * self.camera_height / y_norm
    #             # normalized image plane에서는 focal length를 1로 // Z값이 1
    #         print(int(distance))

    #         cv2.rectangle(undist_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3) 
    #             # bounding box를 이미지 위에 표기
    #         cv2.putText(undist_image, "idx={} / class={} / dist={}".format(index, cls, distance), (xmin, ymin+25), 1, 1, (255, 255, 0), 2)
    #             # 거리와 클래스 정보를 표기
    #             # TODO 위치 재조정
    #         index += 1

    #         display_image = cv2.cvtColor(undist_image, cv2.COLOR_BGR2RGB)
    #             # 화면에 보이기 위해 다시 RBG로 변경
    #         cv2.imshow("result", display_image)
    #         if cv2.waitKey(1) == 27:
    #             break


    def draw_on_image(self, distances):
        """Draw bounding boxes and distance info on image"""

        img = self.image
        for bbox, dist in zip(self.bboxes, distances):
            dist = str(round(dist, 3))
            cv2.rectangle(img, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymax)), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, dist, (int(bbox.xmin), int(bbox.ymin + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        return img


if __name__ == '__main__':
    rospy.init_node("detection", anonymous=False)
    d = Detect()
    rospy.sleep(1)
    while not rospy.is_shutdown():
        d.predict()
        