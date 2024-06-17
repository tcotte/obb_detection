import json
import os

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm


def hex_to_rgb(hexa):
    hexa = hexa[1:]
    return tuple(int(hexa[i:i + 2], 16) for i in (0, 2, 4))


output_directory = "dataset/ds1"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    os.makedirs(os.path.join(output_directory, "img"))
    os.makedirs(os.path.join(output_directory, "ann"))

dataset_length = 200

list_colors = list(matplotlib.colors.cnames.values())

if __name__ == "__main__":
    extension_file = ".jpg"
    height_img = 512
    width_img = 512

    # os.mkdir(os.path.join(output_directory, "img"))
    # os.mkdir(os.path.join(output_directory, "ann"))

    for i in tqdm(range(1, dataset_length + 1)):
        filename = f"image_{str(i)}{extension_file}"
        img = np.zeros((height_img, width_img, 3), dtype=np.uint8)
        nb_rectangles = np.random.randint(1, 5)

        masks = []
        sly_oriented_rectangles = []

        for i in range(nb_rectangles):
            width, height = (np.random.randint(20, 150), np.random.randint(2, 150))
            max_side_length = max(width, height)
            random_position = (np.random.randint(max_side_length, width_img - max_side_length),
                               np.random.randint(max_side_length, height_img - max_side_length))

            angle = np.random.randint(0, 360)

            # (x, y), (width, height), angle = cv2.minAreaRect(contours[0])
            # print(x, y ,width, height, angle)
            x, y = random_position
            obb = cv2.RotatedRect((x, y), (width, height), angle)
            # print("ok")
            #
            # points = np.around(obb.points()).astype(int)
            # for i in range(len(points)):
            #     cv2.line(img, points[i], points[(i+1)%4], (255, 0, 0), 5)
            box = np.intp(cv2.boxPoints(obb))
            color = hex_to_rgb(list_colors[np.random.randint(0, len(list_colors))])

            cv2.drawContours(img, [box], 0, color, -1)

            # mask = np.zeros((height_img, width_img, 3), dtype=np.uint8)
            # cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
            # mask = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_BINARY)[1]
            # mask = mask.astype(bool)
            # masks.append(mask)

            sly_oriented_rectangles.append(box)

        # plt.imshow(img)
        # plt.show()
        cv2.imwrite(os.path.join(output_directory, "img", filename), img)

        oriented_rectangle_sly_objects = []
        for oriented_rectangle in sly_oriented_rectangles:
            oriented_rectangle_sly_objects.append({
                "classId": 1,
                "description": "",
                "geometryType": "polygon",
                "tags": [],
                "classTitle": "rotated_rectangle",
                "points": {
                    "exterior": oriented_rectangle.tolist(),
                    "interior": []
                }
            })
        sly_json = {
            "description": "",
            "size": {
                "height": height_img,
                "width": width_img
            },
            "objects": oriented_rectangle_sly_objects,
            "tags": []
        }

        with open(os.path.join(output_directory, "ann", filename[:-4] + extension_file + ".json"), "w") as outfile:
            json.dump(sly_json, outfile)

        # for mask in masks:
        #     plt.imshow(mask, cmap="gray")
        #     plt.show()
