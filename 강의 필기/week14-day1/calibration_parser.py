import json
import numpy as np

def read_json_file(path):
    print("=" * 50)
    print("Read JSON File: ", path)
    print("=" * 50)
    with open(path, "r",) as f:
        calibration_json = json.load(f)

    intrinsic = calibration_json["intrinsic"]
    print("Intrinsic Calibration\n", intrinsic)
    extrinsic = calibration_json["extrinsic"]
    print("Extrinsic Calibration\n", extrinsic)

    camera_matrix = parse_intrinsic_calibration(intrinsic)

    return camera_matrix

def parse_intrinsic_calibration(intrinsic):
    fx = intrinsic["fx"]
    fy = intrinsic["fy"]
    cx = intrinsic["u0"]
    cy = intrinsic["v0"]
    camera_matrix = np.zeros([3, 3], dtype=np.float32)
    camera_matrix[0][0] = fx
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = fy
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 0.0

    return camera_matrix
