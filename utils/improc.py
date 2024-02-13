'''
 # @ Author: Your name
 # @ Create Time: 2024-01-22 15:39:11
 # @ Modified by: Your name
 # @ Modified time: 2024-01-22 15:39:57
 # @ Description:
 '''

import cv2
import os
from segment_anything import SamPredictor
import tqdm
import numpy as np
import gc
import matplotlib.pyplot as plt
import imagesize


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


def process_image(img_path, re_size=(1200, 720)):

    image_bgr = cv2.imread(img_path)
    if re_size is not None:
        image_bgr = cv2.resize(image_bgr, re_size, interpolation = cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb, image_bgr


def get_bbox_prompt(anno_path, imsize=None):
    '''
    TODO: Implement Object Detector to get bbox prompt
    Currently Reading from Annotation File
    '''
    with open(anno_path, "r") as f:
        lines = f.readlines()
        bbox_coords = lines[0].split(" ")[1:]
        bbox_coords = [i for i in bbox_coords]
        bbox_coords = [bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]]

        bbox_prompt = list(map(float, bbox_coords))
        bbox_prompt[0] = bbox_prompt[0] - bbox_prompt[2] / 2
        bbox_prompt[1] = bbox_prompt[1] - bbox_prompt[3] / 2
        bbox_prompt[2] = bbox_prompt[0] + bbox_prompt[2] 
        bbox_prompt[3] = bbox_prompt[1] + bbox_prompt[3] 

        bbox_prompt = np.array(bbox_prompt)
        if imsize is not None:
            # print("Image Size: ", imsize)
            # convert normalized coords to pixel coords
            bbox_prompt[0] = bbox_prompt[0] * imsize[0]
            bbox_prompt[1] = bbox_prompt[1] * imsize[1]
            bbox_prompt[2] = bbox_prompt[2] * imsize[0]
            bbox_prompt[3] = bbox_prompt[3] * imsize[1]
            bbox_prompt = bbox_prompt.astype(int)
            # print("Bbox Prompt: ", bbox_prompt)
            # bbox_coords = bbox_coords.astype(int)
    return bbox_prompt


def get_image_list(image_dir):
    image_list = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".jpeg"):
                image_list.append(os.path.join(root, file))
    return image_list


def get_filepaths_list(input_dir, file_ext, exclude_file=None):
    filepath_list = []
    if file_ext[0] != ".":
        file_ext = "." + file_ext

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if exclude_file not in file and file.endswith(file_ext):
                filepath_list.append(os.path.join(root, file))
    return filepath_list


def get_image_with_annotation(image_list, annotation_list):
    '''
    '''
    database = {}
    for image_path in image_list:
        anno_path = image_path.replace("NATT360JPEG", "NATT360JPEG_Annotated").replace(".jpeg", ".txt")
        if anno_path in annotation_list:
            imsize = imagesize.get(image_path)
            database[image_path] = [anno_path, imsize]
        else:
            print("Annotation not found for: ", image_path)
    return database

def center_crop(img, percentange=0.8):
    '''
    '''
    h, w, c = img.shape
    h_crop = int(h * percentange)
    w_crop = int(w * percentange)
    h_start = int((h - h_crop) / 2)
    w_start = int((w - w_crop) / 2)
    img_cropped = img[h_start:h_start+h_crop, w_start:w_start+w_crop, :]
    return img_cropped



def create_segmentation_mask(db, sam_model):

    mask_predictor = SamPredictor(sam_model)

    for image_path in tqdm.tqdm(db.keys()):
        anno_path, imsize = db[image_path]
        # Load & Preprocess image
        image_rgb, image_bgr = process_image(image_path, re_size=None)
        # Get bbox prompt
        bbox_prompt = get_bbox_prompt(anno_path, imsize=imsize)

        mask_predictor.set_image(image_rgb)
        # Predict mask with bounding box prompt
        masks, scores, logits = mask_predictor.predict(
            box=bbox_prompt,
            multimask_output=False
            )  

        # Multiply image with mask
        masked_image = np.multiply(image_bgr, masks[0][..., None])

        # Add white background
        masked_image = masked_image + (1 - masks[0][..., None]) * 255

        masked_image_cropped = center_crop(masked_image, percentange=0.75)

        # cv2.imshow("Image", masked_image)
        # cv2.waitKey(0)
        # # Get the output path & write image
        imwrite_path = image_path.replace("NATT360JPEG", "NATT360JPEG_Processed")
        os.makedirs(os.path.dirname(imwrite_path), exist_ok=True)
        cv2.imwrite(imwrite_path, masked_image_cropped)

    return None


