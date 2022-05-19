#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import rospy
import cv2, math, json, sys
import numpy as np
from yolov3_trt.msg import BoundingBoxes, BoundingBox

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class BoundingBox:
    def __init__(self):
        self.xmin = 297
        self.xmax = 328
        self.ymin = 297
        self.ymax = 328

class Detect:
    def __init__(self):
        rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.bbox_callback, queue_size=1)
        self.bboxes = []
        self.distances = []

        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        self.bridge = CvBridge()
        self.image = None

        self.camera_height = 0.165    # 자이카에서 카메라까지 높이

        self.json_path = "C:\\coding\\pingpong-ball-detection\\calibration.json" # TODO 파일 경로 넣기
        self.camera_matrix, self.dist_coeff = self.parse_calibration(self.json_path)
        

    def parse_calibration(self, path):
        with open(path, "r",) as f:
            calibration_json = json.load(f)

        camera_matrix = self.parse_intrinsic_calibration(calibration_json["intrinsic"])
        dist_coeff = calibration_json["extrinsic"]["distortion_coff"]

        return camera_matrix, dist_coeff
        
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
        camera_matrix[2][2] = 0.0

        return camera_matrix


    def bbox_callback(self, msg):
        for box in msg.bounding_boxes:
            self.bboxes.append(box)


    def img_callback(self, msg):
        image_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image = self.undistort_imgs(image_raw)


    def predict(self):
        distances = []
        for bbox in self.bboxes:
            dist = self.projection_method(bbox)
            #dist = self.homo       # TODO homography 방식
            distances.append(dist)    # projection 방식
        self.draw_on_image(distances)
        cv2.imshow("show", self.image)


    def projection_method(self, bbox):
        # image_width_half = 320  # TODO 확인
        # image_height_half = 240
        image_width_half = self.image.shape[0] / 2  # TODO 확인
        image_height_half = self.image.shape[1] / 2
        print("image_width_half {}".format(image_width_half))

        bbox_width = abs(bbox.xmax - bbox.xmin) #px
        cx_from_mid = (bbox.xmax + bbox.xmin) / 2 - image_width_half
        cy_from_mid = (bbox.ymax + bbox.ymin) / 2 - image_height_half
        
        f = self.camera_matrix[0][0]
        FOV_H_half = 46.7
        FOV_V_half = 38

        real_width = 0.04   # m

        xz_azimuth = FOV_H_half * cx_from_mid / image_width_half
        yz_azimuth = (FOV_V_half * cy_from_mid) / image_height_half
        print("xz_azimuth {} / yz_azimuth {}".format(xz_azimuth, yz_azimuth))

        Z = (real_width * f) / bbox_width
        X = abs(Z * math.tan(math.radians(xz_azimuth)))
        Y = abs(Z * math.tan(math.radians(yz_azimuth)))  # TODO Wrong!
        print("X {} Y {} Z {}".format(X, Y, Z))

        dist = math.sqrt(X ** 2 + Z ** 2)

        return dist


    def projection_method_longitudinal(self):
        """
        종방향(카메라가 보는 방향)으로의 거리만 추정하는 방법.
        논문의 내용을 구현
        """
        undist_image = cv2.undistort(self.image, self.camera_matrix, self.dist_coeff, None, None)
            # 받은 이미지를 보정한 이미지

        index = 0
        for bbox in self.bboxes:
            xmin = bbox.xmin
            ymin = bbox.ymin
            xmax = bbox.xmax
            ymax = bbox.ymax
            cls = bbox.Class

            y_norm = (ymax - self.camera_matrix[1][2]) / self.camera_matrix[1][1]
                # Normalized Image Plane

            distance = 1 * self.camera_height / y_norm
                # normalized image plane에서는 focal length를 1로 // Z값이 1
            print(int(distance))

            cv2.rectangle(undist_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3) 
                # bounding box를 이미지 위에 표기
            cv2.putText(undist_image, "idx={} / class={} / dist={}".format(index, cls, distance), (xmin, ymin+25), 1, 1, (255, 255, 0), 2)
                # 거리와 클래스 정보를 표기
                # TODO 위치 재조정
            index += 1

            display_image = cv2.cvtColor(undist_image, cv2.COLOR_BGR2RGB)
                # 화면에 보이기 위해 다시 RBG로 변경
            cv2.imshow("result", display_image)
            if cv2.waitKey(1) == 27:
                break


    def draw_on_image(self, distances):
        for bbox, dist in zip(self.bboxes, distances):
            print(bbox.xmin, bbox.ymin, dist)
            dist = str(round(dist, 3))
            cv2.rectangle(self.image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(self.image, dist, (bbox.xmin, bbox.ymin + 1), cv2.FONT_HERSHEY_SIMPLEX, (0, 0, 255), 1)


d = Detect()
# bbox = BoundingBox()    # TODO temp
d.predict()