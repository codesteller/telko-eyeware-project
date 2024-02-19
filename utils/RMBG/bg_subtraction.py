from skimage import io
import torch, os
from PIL import Image
import numpy as np
import cv2
from .briarmbg import BriaRMBG
from .utilities import preprocess_image, postprocess_image
import tqdm



class RMBG14:
    def __init__(self):
        self.net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.model_input_size = [1024,1024]

    
    def remove_background(self, im_path, save_path=None):

        # If save_path is None, return the PIL image
        # Check is Save path exists
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        orig_im = io.imread(im_path)
        orig_im_size = orig_im.shape[0:2]
        image = preprocess_image(orig_im, self.model_input_size).to(self.device)
        result=self.net(image)
        result_image = postprocess_image(result[0][0], orig_im_size)
        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
        if save_path:
            orig_image = Image.open(im_path)
            no_bg_image.paste(orig_image, mask=pil_im)
            no_bg_image.save(save_path)
        else:
            return no_bg_image


    def remove_background_from_image(self, image, save_path=None):
        # If save_path is None, return the PIL image
        # Check is Save path exists
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        orig_im = image
        orig_im_size = orig_im.shape[0:2]
        image = preprocess_image(orig_im, self.model_input_size).to(self.device)
        result=self.net(image)
        result_image = postprocess_image(result[0][0], orig_im_size)
        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
        if save_path:
            orig_image = Image.fromarray(orig_im)
            no_bg_image.paste(orig_image, mask=pil_im)
            no_bg_image.save(save_path)
        else:
            return no_bg_image

    
    def get_bbox_prompt(self, anno_path, imsize=None):
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

    def crop_image(self, image_path, anno_path, imsize):

        im = io.imread(image_path)
        
        # Get bbox prompt
        bbox_prompt = self.get_bbox_prompt(anno_path, imsize)
        anno = (bbox_prompt[0], bbox_prompt[1], bbox_prompt[2], bbox_prompt[3])
        im = im[anno[1]:anno[3], anno[0]:anno[2]]
        return im

    
    def remove_background_from_dir(self, im_dir):
        for root, dirs, files in tqdm.tqdm(os.walk(im_dir)):
            for file in files:
                if file.endswith(".jpeg"):
                    image_path = os.path.join(root, file)
                    imwrite_path = image_path.replace("NATT360JPEG", "NATT360JPEG_BGRemoved")
                    self.remove_background(image_path, imwrite_path)

    def remove_background_from_image_list(self, image_list):
        for image_path in tqdm.tqdm(image_list):
            imwrite_path = image_path.replace("NATT360JPEG", "NATT360JPEG_BGRemoved").replace(".jpeg", "_nobg.png")
            self.remove_background(image_path, imwrite_path)


    def remove_background_from_db(self, database):
        for image_path in tqdm.tqdm(database.keys()):
            imwrite_path = image_path.replace("NATT360JPEG", "NATT360JPEG_BGRemoved").replace(".jpeg", "_nobg.png")
            anno_path, imsize = database[image_path]
            cropped_im = self.crop_image(image_path, anno_path, imsize)

            self.remove_background_from_image(cropped_im, imwrite_path)
    
    