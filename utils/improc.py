'''
 # @ Author: Your name
 # @ Create Time: 2024-01-22 15:39:11
 # @ Modified by: Your name
 # @ Modified time: 2024-01-22 15:39:57
 # @ Description:
 '''

import cv2


def process_image(img_path, re_size=(1200, 720)):

    image_bgr = cv2.imread(img_path)
    if re_size is not None:
        image_bgr = cv2.resize(image_bgr, re_size, interpolation = cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb, image_bgr
