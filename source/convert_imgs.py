import cv2
import glob
from cv2 import undistort

def undistort_imgs(camera_matrix, dist_coeffs):
    raw_path_list = glob.glob("images/*.jpg")
    output_path = "output/"

    for image_path in raw_path_list:
        file_name = list(image_path.split('/'))[-1]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image_undist = cv2.undistort(image, camera_matrix, dist_coeffs, None)
        file_name = output_path + file_name
        cv2.imwrite(file_name, image_undist, params=[cv2.IMWRITE_PNG_COMPRESSION,0])