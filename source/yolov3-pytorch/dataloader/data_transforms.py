import numpy as np
import cv2
import torch
from torchvision import transforms as tf

#  pip install imgaug
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from util.tools import minmax2cxcy, xywh2xyxy_np

def get_transformations(cfg_param = None, is_train = None):
    # TODO 3 : Add Augumentation method
    if is_train:
        data_transform = tf.Compose([AbsoluteLabels(),
                                     FlipAug_tstl(),
                                     #PadSquare(),
                                     DefaultAug(),
                                     #ImageBaseAug(),
                                     RelativeLabels(),
                                     ResizeImage(new_size = (cfg_param['in_width'], cfg_param['in_height'])),
                                     ToTensor(),])

    elif not is_train:
        data_transform = tf.Compose([AbsoluteLabels(),
                                     RelativeLabels(),
                                     ResizeImage(new_size = (cfg_param['in_width'], cfg_param['in_height'])),
                                     ToTensor(),]) 
    return data_transform

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def add(self, transform):
        self.transforms.append(transform)


class ToTensor(object):
    def __init__(self, max_objects=50, is_debug=False):
        self.max_objects = max_objects
        self.is_debug = is_debug

    def __call__(self, data):
        image, labels = data
        if self.is_debug == False:
            image = torch.tensor(np.ascontiguousarray(np.transpose(np.array(image, dtype=float) / 255,(2,0,1))),dtype=torch.float32)
        elif self.is_debug == True:
            image = torch.tensor(np.array(image, dtype=float),dtype=torch.float32)
        labels = torch.FloatTensor(np.array(labels))
        return image, labels

class KeepAspect(object):
    def __init__(self):
        pass

    def __call__(self, data):
        image, label = data

        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        image_new = np.pad(image, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = image_new.shape

        # Extract coordinates for unpadded + unscaled image
        x1 = w * (label[:, 0] - label[:, 2]/2)
        y1 = h * (label[:, 1] - label[:, 3]/2)
        x2 = w * (label[:, 0] + label[:, 2]/2)
        y2 = h * (label[:, 1] + label[:, 3]/2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        label[:, 0] = ((x1 + x2) / 2) / padded_w
        label[:, 1] = ((y1 + y2) / 2) / padded_h
        label[:, 2] *= w / padded_w
        label[:, 3] *= h / padded_h

        return image_new, label

class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.new_size = tuple(new_size) #  (w, h)
        self.interpolation = interpolation

    def __call__(self, data):
        image, label = data
        image = cv2.resize(image, self.new_size, interpolation=self.interpolation)

        return image, label

class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        self.seq = iaa.Sequential(
            [
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                # iaa.OneOf([
                #     iaa.GaussianBlur((0, 0.3)),
                #     iaa.AverageBlur(k=(0, 2)),
                #     iaa.MedianBlur(k=(1, 3)),
                # ]),
                iaa.OneOf([
                    # Color
                    iaa.AddToHue((-10, 10)),
                    iaa.AddToHueAndSaturation((-10, 10)),
                    iaa.AddToSaturation((-10, 10))
                    #iaa.Grayscale(alpha=(0.0, 1.0))
                ]),
                # Sharpen each image, overlay tdhe result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                # Add gaussian noise to some images.
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # Add a value of -5 to 5 to each pixel.
                sometimes(iaa.Add((-2, 2), per_channel=0.5)),
                # Change brightness of images (80-120% of original value).
                sometimes(iaa.Multiply((0.9, 1.1), per_channel=0.5)),
                # Improve or worsen the contrast of images.
                # sometimes(iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5)),
                iaa.AddToBrightness((-40, 60)), # (mul=(0.5, 1.5), add=(-30, 30))
                iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(1.0, 2.5)),         
            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def __call__(self, data):
        seq_det = self.seq.to_deterministic()
        image, label = data
        image = seq_det.augment_images([image])[0]
        return image, label

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape)

        if len(bounding_boxes) != 0:
            origin_box = bounding_boxes[0]

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        if len(self.augmentations.find_augmenters_by_name('fliplr_tstl')) != 0 and len(bounding_boxes) != 0:
            augmented_box = bounding_boxes[0]
            if origin_box.x1 != augmented_box.x1 or origin_box.x2 != augmented_box.x2:
                use_flip = True # 좌표가 달라졌다면 flip 된 것.
            else:
                use_flip = False

            if use_flip:
                # 좌회전과 우회전을 flip 통해 바꿔줬다면 라벨도 바꿔주기
                for box_idx, box in enumerate(bounding_boxes):
                    if box.label == 0:
                        bounding_boxes[box_idx].label = 1
                    elif box.label == 1:
                        bounding_boxes[box_idx].label = 0
        
        # Clip out of image boxes
        bounding_boxes = bounding_boxes.remove_out_of_image_fraction(0.4)
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5), dtype=np.float64)
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes

class DefaultAug(ImgAug):
    def __init__(self, ):
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),                
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(1.0, 2.5)),         
            iaa.AddToBrightness((-40, 60)), # (mul=(0.5, 1.5), add=(-30, 30))
            sometimes(iaa.OneOf([
                    # Color
                    iaa.AddToHue((-10, 10)),
                    iaa.AddToHueAndSaturation((-10, 10)),
                    iaa.AddToSaturation((-10, 10))
                    #iaa.Grayscale(alpha=(0.0, 1.0))
                ])),
                
            ],
        random_order=True
        )

#flip augmentation for tstl dataset
#if flip occured, change label of the box between "left sign" and "right sign"
class FlipAug_tstl(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
                iaa.Fliplr(0.5, name='fliplr_tstl')
        ])


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        label[:, [1, 3]] *= w
        label[:, [2, 4]] *= h
        return image, label

class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        label[:, [1, 3]] /= w
        label[:, [2, 4]] /= h
        return image, label

class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])

class AffineAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.4, aug)
        self.seq = iaa.Sequential(
            [
                sometimes(iaa.Affine(scale = 0.8)),
                # sometimes(iaa.Affine(translate_percent=0.1))
            ],
            random_order=True
        )
    
    def __call__(self, data):
        seq_det = self.seq.to_deterministic()
        image, label = data
        img_h, img_w = image.shape[:2]
        ia_bboxes = []
        for box in label:
            xmin, xmax = img_w * (box[0] - box[2]/2), img_w * (box[0] + box[2]/2)
            ymin, ymax = img_h * (box[1] - box[3]/2), img_h * (box[1] + box[3]/2)
            ia_bboxes.append(ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax))
        label_ia =ia.BoundingBoxesOnImage(ia_bboxes, shape=image.shape)
        image = seq_det.augment_images([image])[0]
        label_ia = seq_det.augment_bounding_boxes([label_ia])[0]


        for i, bbox in enumerate(label_ia):
            label[i] = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2])

        return image, label