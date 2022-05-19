import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "source\\yolov3-pytorch\\datasets\\train\\Annotations"
CLUSTERS = 6

def load_dataset(path):
    dataset = []
    height = 480
    width = 640
    for txt_file in glob.glob("{}\\*.txt".format(path)):
        
        with open(txt_file,"r") as file:
            for i in file.readlines():
                txt_split = i[:-2].split(" ")
                ds=[float(txt_split[3])*width, float(txt_split[4])*height]
                dataset.append(ds)

    return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))