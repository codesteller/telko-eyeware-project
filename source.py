'''
 # @ Author: Pallab Maji
 # @ Create Time: 2024-01-29 17:22:56
 # @ Modified time: 2024-01-29 21:54:32
 # @ Description: Enter description here
 '''

import os
import cv2
import numpy as np
from segment_anything import sam_model_registry
from utils import improc
from utils.RMBG import bg_subtraction as rmbg
import torch



def main(MODEL_TYPE, data_dir):
    WORKSPACE = os.getcwd()
    MODEL_DIR = os.path.join(WORKSPACE, "./models/pth")

    SAM_MODEL_ARCH = "sam_vit_h_4b8939.pth"
    SAM_HMODEL_PATH = os.path.join(MODEL_DIR, SAM_MODEL_ARCH)
    DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=SAM_HMODEL_PATH).to(device=DEVICE)

    image_dir = os.path.join(data_dir, "NATT360JPEG")
    anno_dir = os.path.join(data_dir, "NATT360JPEG_Annotated")

    # Get Image List
    image_list = improc.get_image_list(image_dir)  
    print("Total Images: ", len(image_list))

    # Get Annotation List
    anno_list = improc.get_filepaths_list(anno_dir, ".txt", "classes.txt")
    print("Total Annotations: ", len(anno_list))

    # Get Image with Annotation
    database = improc.get_image_with_annotation(image_list, anno_list)

    print("Total Images with Annotation: ", len(database))

    # improc.create_segmentation_mask(database, sam_model)


    rmbg_model = rmbg.RMBG14()
    # rmbg_model.remove_background_from_image_list(image_list)
    rmbg_model.remove_background_from_db(database)
    




    # # Test bbox prompt
    # image_path = list(database.keys())[0]
    # anno_path, imsize = database[image_path]

    # # Load & Preprocess image
    # # image_rgb, image_bgr = improc.process_image(image_path, re_size=None)
    # image_rgb, image_bgr = improc.process_image(image_path, re_size=(1200, 720))
    # # Get bbox prompt
    # bbox_prompt = improc.get_bbox_prompt(anno_path, imsize=(1200, 720))


    # cv2.rectangle(image_bgr, (bbox_prompt[0], bbox_prompt[1]), (bbox_prompt[2], bbox_prompt[3]), (0, 255, 0), 2)
    # cv2.imshow("Image", image_bgr)
    # cv2.waitKey(0)


if __name__ == "__main__":

    # data_dir = "/home/pallab/Dataset/telko-eyeware/Fotos_360_CRUDAS/NATT/"
    data_dir = "/workspace/data/NATT/"
    MODEL_TYPE = "vit_h"
    main(MODEL_TYPE, data_dir)



