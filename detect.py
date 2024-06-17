import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.models import ResNet18_Weights, resnet18

from object_detector import ObjectDetector
import albumentations as A

img_test_transform = A.Compose([
        A.Normalize(always_apply=True),
        ToTensorV2()]
)

CLASSES = ["rectangle"]

if __name__ == '__main__':

    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    objectDetector = ObjectDetector(backbone, len(CLASSES), confidence=0.01)

    im0 = cv2.imread("dataset/ds0/img/image_110.jpg")
    img_w, img_h = im0.shape[1], im0.shape[0]
    augmented = img_test_transform(image=im0)
    image = augmented['image']

    image = torch.unsqueeze(image, 0)

    output = objectDetector(image)

    print(output)


