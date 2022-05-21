import numpy as np
import json

def read_json_file(path):
    with open(path, "r") as f :
        calibration_json = json.load(f)

    camera_matrix = parse_intrinsic_calibration(calibration_json["intrinsic"])
    dist_coeff = calibration_json["extrinsic"]["distortion_coff"][0]
    
    return camera_matrix, np.array(dist_coeff)

def parse_intrinsic_calibration(intrinsic):
    fx = intrinsic["fx"]
    fy = intrinsic["fy"]
    cx = intrinsic["cx"]
    cy = intrinsic["cy"]

    camera_matrix = np.zeros([3,3],dtype=np.float32)
    camera_matrix[0][0] = fx
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = fy
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1.0

    return camera_matrix



