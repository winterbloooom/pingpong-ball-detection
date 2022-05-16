import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

import albumentations as A

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

image = cv2.imread("od/image.png")

# cv2.imshow("TEST", image)
# cv2.waitKey(0)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

""" KITTI Dataset Format
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""

CLASS_TABLE = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
    "DontCare": 8
}

category_id_to_name = dict()
for key, value in CLASS_TABLE.items():
    category_id_to_name[value] = key

# TODO YOLO Parse
bboxes = list()
category_ids = list()

with open("od/label.txt", "r", encoding="UTF-8") as od_label:
    lines = od_label.readlines()

""" COCO Dataset Format
[x_min y_min width height]
"""

for line in lines:
    line = line.split(' ')
    
    class_id = CLASS_TABLE[line[0]]
    left = float(line[4])
    top = float(line[5])
    right = float(line[6])
    bottom = float(line[7])

    width = right - left
    height = bottom - top

    bboxes.append([left, top, width, height])
    category_ids.append(class_id)

image_size = np.shape(image)
print(image_size)
image_height = image_size[0]
image_width = image_size[1]

transform = A.Compose([
    # TODO: add audmentation methods
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']

print(transformed_bboxes)

visualize(transformed_image, transformed_bboxes, category_ids, category_id_to_name)
