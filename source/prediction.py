#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import rospy
import cv2, math, json, sys
import numpy as np
# from yolov3_trt.msg import BoundingBoxes, BoundingBox

# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

class Detect:
    def __init__(self):
        # rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.bbox_callback, queue_size=1)
        # self.bboxes = []

        # rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        # self.bridge = CvBridge()
        self.image = None

        self.camera_height = 0.14    # 자이카에서 카메라까지 높이

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
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

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
        
    def projection_method(self, bbox_x, bbox_y, bbox_w, bbox_h):
        FOV_verti = 127.5
        half_FOV_verti = FOV_verti / 2
        FOV_horiz = 170
        half_FOV_horiz = FOV_horiz / 2
        fx = self.camera_matrix[0][0]
        fy = self.camera_matrix[1][1]
        cx = self.camera_matrix[0][2]    # TODO calibration의 값을 써야 할지??
        cy = self.camera_matrix[1][2]

        azimuth_horiz = abs(cx - bbox_x) * (FOV_horiz / 2) / cx
        # d = fx * (bbox_h / (2*math.tan(azimuth_horiz))) / ((bbox_h / 2 * math.tan(azimuth_horiz)) - fx)
        azimuth_verti = abs(cy - bbox_y) * (FOV_verti / 2) / cy
        print("azimuth_horiz {0:03.3f} / azimuth_verti {1:03.3f}".format(azimuth_horiz, azimuth_verti))

        d_ZX = (fx * (bbox_w/(2 * math.tan(azimuth_horiz))))/((bbox_w / (2 * math.tan(azimuth_horiz))) - fx) #(self.camera_matrix[0][0] * bbox_w) / (bbox_w - 2 * self.camera_matrix[0][0] * math.tan(azimuth_horiz / 2))
        d_YZ = (fy * (bbox_h/(2 * math.tan(half_FOV_verti))))/((bbox_h / (2 * math.tan(half_FOV_verti))) - fy)#(self.camera_matrix[1][1] * bbox_h) / (bbox_h - 2 * self.camera_matrix[1][1] * math.tan(azimuth_verti / 2))
        print("d_ZX {} / d_YZ {}".format(d_ZX, d_YZ))

        X = d_ZX * math.sin(azimuth_horiz)
        Y = d_YZ * math.sin(FOV_verti / 2)
        print("X {} / Y {}".format(X, Y))

        Z_from_horiz = d_ZX * math.cos(azimuth_horiz)
        Z_from_verti = d_YZ * math.cos(FOV_verti / 2)
        print("Z_from_horiz {} \nZ_from_verti {}".format(Z_from_horiz, Z_from_verti))

    def projection_method_longitudinal(self):
        y_norm = (328 - self.camera_matrix[1][2]) / self.camera_matrix[1][1]
            # Normalized Image Plane
        print(y_norm)

        distance = float(1 * 0.15 / y_norm)
            # normalized image plane에서는 focal length를 1로 // Z값이 1
        print(distance)

    def projection_method_longitudinal2(self):
        x_norm = (640 - self.camera_matrix[0][2]) / self.camera_matrix[0][0]
            # Normalized Image Plane

        distance = float(1 * 0.4 / x_norm)
            # normalized image plane에서는 focal length를 1로 // Z값이 1
        print(distance)

d = Detect()
# d.projection_method(313, 312, 31, 31)
# d.projection_method(630, 304, 31, 28)
d.projection_method_longitudinal2()

