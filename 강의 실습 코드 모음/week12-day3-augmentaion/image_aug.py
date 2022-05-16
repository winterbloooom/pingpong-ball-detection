import cv2
import random
import albumentations as A

def visualization(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Visualization", image)
    cv2.waitKey(0)
    
# OpenCV -> BGR
image = cv2.imread("image/dog.png", cv2.IMREAD_ANYCOLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

visualization(image)

# Step 1: Basic Transform
transform = A.HorizontalFlip(p=0.5)
# random.seed(7)
augmented_image = transform(image=image)['image']
visualization(augmented_image)

# Step 2: Rotation
transform = A.ShiftScaleRotate(p=0.5)
random.seed(7) 
augmented_image = transform(image=image)['image']
visualization(augmented_image)

# Step 3: Compose
# TODO Add more augmentation method
transform = A.Compose([
    A.CLAHE(),
    A.RandomRotate90(),
    A.Transpose(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    A.Blur(blur_limit=3),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.HueSaturationValue(),
])

random.seed(42)
for _ in range(10):
    augmented_image = transform(image=image)['image']
    visualization(augmented_image)
