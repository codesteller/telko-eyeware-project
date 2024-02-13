from skimage import io
import torch, os
from PIL import Image
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
import tqdm



class RMBG14:
    def __init__(self):
        self.net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.model_input_size = [1024,1024]

    
    def remove_background(self, im_path, save_path=None):
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
    
    def remove_background_from_dir(self, im_dir):
        for root, dirs, files in tqdm.tqdm(os.walk(im_dir)):
            for file in files:
                if file.endswith(".jpeg"):
                    image_path = os.path.join(root, file)
                    imwrite_path = image_path.replace("NATT360JPEG", "NATT360JPEG_BGRemoved")
                    self.remove_background(image_path, imwrite_path)
        

    
    