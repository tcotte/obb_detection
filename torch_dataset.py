import os
import platform

import cv2
import imutils.paths
# import supervisely as sly
import numpy as np
from typing import List
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import List


def get_os() -> str:
    return platform.system()



def get_labelfile_from_imgfile(img_path):
    path = os.path.normpath(img_path)
    splitted_path = path.split(os.sep)
    if get_os() == "Windows":
        label_path = os.path.join("C:\\", *splitted_path[1:-2], "ann", splitted_path[-1] + ".json")
    else:
        label_path = "/" + os.path.join(*splitted_path[1:-2], "ann", splitted_path[-1] + ".json")
    return label_path


class OrientedBoundingBoxes(Dataset):
    def __init__(self, image_paths, mask_paths, classes: List, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

        self.classes = classes
        # self.classes = ["living",  "dead", "necrotized"]

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(list(imutils.paths.list_images(self.image_paths)))

    def __getitem__(self, idx):

        img_name = list(imutils.paths.list_images(self.image_paths))[idx]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_file = get_labelfile_from_imgfile(img_path=img_name)

        polygons, height_img, width_img = read_sly_json(json_file_path=ann_file)

        masks = []
        for polygon in polygons:
            full_size_mask = np.zeros((height_img, width_img), dtype=np.uint8)
            masks.append(cv2.fillPoly(full_size_mask, pts=[np.array(polygon["points"]["exterior"])],
                                      color=255).astype(bool))

        if image.shape[1] > image.shape[0]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            for idx in range(len(masks)):
                masks[idx] = cv2.rotate(masks[idx], cv2.ROTATE_90_CLOCKWISE)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            transformed = self.transforms(image=image, masks=masks)
            image = transformed['image']
            masks = transformed['masks']

        obb = []
        for mask in masks:
            numpy_mask = (mask.type(torch.uint8)*255).numpy()
            cnts, hierarchy = cv2.findContours(numpy_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(cnts[0])
            if rect[1][1] > rect[1][0]:
                angle = 90 - rect[2]
            else:
                angle = -rect[2]

            center, lengths = rect[:2]

            box = cv2.boxPoints(rect)
            # print (box)
            box = np.int0(box)
            obb.append(box)
        obb = np.array(obb)

        labels = torch.ByteTensor([0]*len(obb))


        # return a tuple of the image and its mask
        return image, labels, torch.stack(masks), torch.Tensor(obb).type(torch.uint16), torch.Tensor((*center, *lengths, angle)).type(torch.float32)

    # def get_bitmap(self, json_file, object_index=0) -> sly.Bitmap:
    #     objs, h_img, w_img = read_sly_json(json_file_path=json_file)
    #
    #     bitmaps = []
    #     for object_index in range(len(objs)):
    #         encoded_string = objs[object_index]['bitmap']['data']
    #
    #         if objs[object_index]['classTitle'] == "necrotized":
    #             cls = 1
    #         else:
    #             cls = self.classes.index(objs[object_index]['classTitle'])
    #
    #         bitmap = sly.Bitmap(data=sly.Bitmap.base64_2_data(encoded_string),
    #                             origin=sly.PointLocation(objs[object_index]['bitmap']['origin'][0],
    #                                                      objs[object_index]['bitmap']['origin'][1]),
    #                             class_id=cls)
    #         bitmaps.append(bitmap)
    #
    #     return bitmaps, h_img, w_img
    #
    # def bitmaps2masks(self, bitmaps: List[sly.Bitmap], h_img: int, w_img: int) -> np.array:
    #     masks = []
    #
    #     for cls in self.classes:
    #         mask_full_img = np.zeros((h_img, w_img), int)
    #         masks.append(mask_full_img)
    #
    #     for bitmap in bitmaps:
    #         index_mask = bitmap.class_id
    #         masks[index_mask] = self.op_mask_bitmap(full_target_mask=masks[index_mask], bitmap=bitmap,
    #                                                 bit_op=np.logical_or)
    #
    #     return masks
    #
    # @staticmethod
    # def reconstitute_mask(bitmap: sly.Bitmap, h_img: int, w_img: int) -> np.ndarray:
    #     mask_full_img = np.zeros((h_img, w_img), dtype=np.uint8)
    #
    #     mask_full_img[bitmap.origin.col:bitmap.origin.col + bitmap.data.shape[0],
    #     bitmap.origin.row: bitmap.origin.row + bitmap.data.shape[1]] = bitmap.data
    #     return mask_full_img

    # @staticmethod
    # def op_mask_bitmap(full_target_mask: np.array, bitmap: sly.Bitmap, bit_op) -> np.array:
    #     full_size = full_target_mask.shape[:2]
    #     origin, mask = bitmap.origin, bitmap.data
    #     full_size_mask = np.full(full_size, False, bool)
    #     full_size_mask[
    #     origin.col: origin.col + mask.shape[0],
    #     origin.row: origin.row + mask.shape[1],
    #     ] = mask
    #
    #     new_mask = bit_op(full_target_mask, full_size_mask).astype(int)
    #     return new_mask

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def read_sly_json(json_file_path: str):
    """
    Read Supervisely output json file and retrieve only data which interesting us
    :param json_file_path: path of the Supervisely json file
    :return:
            - Multiple bitmap data in a list
            - Original picture height
            - Original picture width
    """
    with open(json_file_path) as json_file:
        data = json.load(json_file)

    h, w = data['size']['height'], data['size']['width']
    return data['objects'], h, w


if __name__ == "__main__":
    transform = A.Compose([
        A.Normalize(),
        # A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ])

    obb_dataset = OrientedBoundingBoxes(image_paths=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\dataset\ds0\img",
                          mask_paths=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\dataset\ds0\ann",
                          classes=["rec"],
                          transforms=transform)

    item = 1
    sample = obb_dataset[item]
    print(sample["labels"])

    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    fig, axs = plt.subplots(nrows=len(sample["masks"]) + 1, ncols=1)
    raw_image = ((unorm(sample["image"]).numpy().transpose(2, 1, 0))*255).astype('uint8').copy()

    for index, value in enumerate(sample["masks"]):
        axs[index + 1].imshow(np.transpose(value), cmap="gray")

    for rec in sample["obb_point_based"]:
        for pt in rec:
            cv2.circle(raw_image, tuple(pt.numpy().tolist()[::-1]), 3, (255, 0, 0), -1)

    axs[0].imshow(raw_image)

    plt.show()

    # print(obb_dataset[1])
