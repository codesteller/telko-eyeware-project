'''
 # @ Author: Pallab Maji
 # @ Create Time: 2024-01-29 17:38:12
 # @ Modified time: 2024-01-29 18:06:38
 # @ Description: Create Masks from input images
 '''

import os
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import numpy as np
import matplotlib.pyplot as plt
import gc

def show_mask(mask, ax, random_color=False):
    print(mask.shape)
    print(str(mask))

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        print(str(color))
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  plt.show()
  del mask
  gc.collect()

WORKSPACE = os.getcwd()
MODEL_DIR = os.path.join(WORKSPACE, "./models/pth")
TEST_DIR = os.path.join(WORKSPACE, "./data/test/Input_JPEG")

MODEL_TYPE = "vit_h"
SAM_MODEL_ARCH = "sam_vit_h_4b8939.pth"
SAM_HMODEL_PATH = os.path.join(MODEL_DIR, SAM_MODEL_ARCH)
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_HMODEL_PATH).to(device=DEVICE)

image_path = "data/test/Input_JPEG/BTR4527C2_img_01.jpeg"

bbox_coords = [1297, 1058, 3005, 1617]
# # Yolo Detect BBox
# bbox_coords = [0.503511, 0.469628, 0.503511+0.399813, 0.469628+0.196278]

# Load image
bgr_img = cv2.imread(image_path)
# Convert to RGB
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# Convert bbox to SAM format
img_size = rgb_img.shape[:2]
# Multiply wit image size and convert to int
bbox_prompt = np.array(bbox_coords)
# bbox_prompt = [int(bbox_coords[0]*img_size[1]), int(bbox_coords[1]*img_size[0]), int(bbox_coords[2]*img_size[1]), int(bbox_coords[3]*img_size[0])]
# print(bbox_prompt)

# #Draw bbox on image
# cv2.rectangle(bgr_img, (bbox_coords[0], bbox_coords[1]), (bbox_coords[2], bbox_coords[3]), (255,0,0), 2)
# cv2.imshow("Image", bgr_img)
# cv2.waitKey(0)

# Set up the SAM model with the encoded image
mask_predictor = SamPredictor(sam)
mask_predictor.set_image(rgb_img)

# Predict mask with bounding box prompt
masks, scores, logits = mask_predictor.predict(
box=bbox_prompt,
multimask_output=False
)  
cv2.rectangle(rgb_img, (bbox_coords[0], bbox_coords[1]), (bbox_coords[2], bbox_coords[3]), (255,255,0), 2)
# combine all masks into one for easy visualization
final_mask = None
for i in range(len(masks) - 1):
  if final_mask is None:
    final_mask = np.bitwise_or(masks[i][0], masks[i+1][0])
  else:
    final_mask = np.bitwise_or(final_mask, masks[i+1][0])

# Plot the bounding box prompt and predicted mask
# plt.imshow(rgb_img)
# show_mask(masks[0], plt.gca())
# # plt.imshow(final_mask)
# plt.show()
    
# Plot the bounding box prompt and predicted mask
# show_mask(masks[0], plt.gca())
# plt.show()

# Multiply image with mask
masked_image = np.multiply(rgb_img, masks[0][..., None])

# Add white background
masked_image = masked_image + (1 - masks[0][..., None]) * 255

plt.imshow(masked_image)
plt.show()

imwrite_path = "data/test/Processed_Output/BTR4527C2_img_01.jpeg"
masked_image = np.float32(masked_image)
cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(imwrite_path, masked_image)







