'''
 # @ Author: Your name
 # @ Create Time: 2024-01-22 15:20:39
 # @ Modified by: Your name
 # @ Modified time: 2024-01-22 15:21:04
 # @ Description:
 '''

import os
import sys
import time
import numpy as np
import torch
import urllib.request
from matplotlib import pyplot as plt
from super_gradients.training import models
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
import utils.improc as improc
import cv2



OBJ_MODEL_ARCH = 'yolo_nas_l'
MODEL_TYPE = "vit_h"
SAM_MODEL_ARCH = "sam_vit_h_4b8939.pth"
IMAGE_TYPE = '.jpeg'
WORKSPACE = os.getcwd()
MODEL_DIR = os.path.join(WORKSPACE, "./models/pth")
TEST_DIR = os.path.join(WORKSPACE, "./data/test/Input_JPEG")
SAM_MODEL_HTTPS = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
do_objdet = False


print("WORKSPACE:", WORKSPACE)
print("MODEL DIR:", MODEL_DIR)

SAM_MODEL_PATH = os.path.join(MODEL_DIR, SAM_MODEL_ARCH)

if not os.path.exists(SAM_MODEL_PATH):
    urllib.request.urlretrieve(SAM_MODEL_HTTPS, SAM_MODEL_PATH)

if os.path.exists(SAM_MODEL_PATH):
    print("MODEL PATH:", SAM_MODEL_PATH)
else:
    print("MODEL PATH NOT EXIST:", SAM_MODEL_PATH)
    sys.exit(0)


DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

if do_objdet:
    obj_model = models.get(OBJ_MODEL_ARCH, pretrained_weights="coco").to(DEVICE)

sam_model = sam_model_registry[MODEL_TYPE](checkpoint=SAM_MODEL_PATH).to(device=DEVICE)


#  Get all the Test Images
impaths = list()
for root, dirs, files in os.walk(TEST_DIR):
    for file in files:
        if file.endswith(IMAGE_TYPE):
            impaths.append(os.path.join(root, file))

print("{} test images found.".format(len(impaths)))

#  Process the images
image_rgb, image_bgr = improc.process_image(impaths[0], re_size=(1200, 720))

#  Get the mask
mask_generator = SamAutomaticMaskGenerator(sam_model)

output_dir = "./data/output"
os.makedirs(output_dir, exist_ok=True)

for impath in impaths:
    image_rgb, image_bgr = improc.process_image(impath, re_size=(1200, 720))
    sam_result = mask_generator.generate(image_rgb)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    cv2.imshow("image", annotated_image)

    outfile = os.path.join(output_dir, os.path.basename(impath))
    cv2.imwrite(outfile, annotated_image)

    if cv2.waitKey(5) == ord('q'):
        break


# sam_result = mask_generator.generate(image_rgb)

# mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
# detections = sv.Detections.from_sam(sam_result=sam_result)
# annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

# sv.plot_images_grid(
#     images=[image_bgr, annotated_image],
#     grid_size=(1, 2),
#     titles=['source image', 'segmented image']
# )

# plt.imshow(image_rgb)
# plt.show()



