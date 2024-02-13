'''
 # @ Author: Pallab Maji
 # @ Create Time: 2024-02-11 09:07:56
 # @ Modified time: 2024-02-11 14:54:33
 # @ Description: Enter description here
 '''


import os
import cv2
import numpy as np

class Database:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "NATT360JPEG")
        self.anno_dir = os.path.join(data_dir, "NATT360JPEG_Annotated")
        self.image_list = self.get_image_list()  
        self.anno_list = self.get_anno_list()
        self.database = self.get_database()

    def get_image_list(self):
        # Get Image List
        image_list = []
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(".jpeg"):
                    image_list.append(os.path.join(root, file))

        return image_list
    
    def get_anno_list(self):
        # Get Annotation List
        anno_list = []
        for root, dirs, files in os.walk(self.anno_dir):
            for file in files:
                if file.endswith(".txt"):
                    anno_list.append(os.path.join(root, file))

        return anno_list
    
    def get_database(self):
        # Get Image with Annotation
        database = {}
        for image_path in self.image_list:
            for anno_path in self.anno_list:
                if os.path.basename(image_path).split(".")[0] == os.path.basename(anno_path).split(".")[0]:
                    database[image_path] = anno_path

        return database
    

class DatasetPreparation:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.db = Database(data_dir)
        self.image_list = self.db.image_list
        self.anno_list = self.db.anno_list
        self.database = self.db.database

    def read_annotation(self, anno_path):
        # Read Annotation
        with open(anno_path, "r") as file:
            lines = file.readlines()
            anno = []
            for line in lines:
                anno.append(line.strip().split(" "))

        return anno

    def read_image(self, image_path):
        # Read Image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def draw_bbox(self, image, anno):
        # Draw Bounding Box
        for box in anno:
            x1, y1, x2, y2 = map(int, box[:4])
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image
    
    def crop_image(self, image, anno):
        # Crop Image
        cropped_images = []
        for box in anno:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

        return cropped_images

    def save_cropped_image(self, cropped_image, save_path):
        # Save Cropped Image
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cropped_image)
       

    def save_image(self, image, image_path):
        # Save Image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image)

    def save_annotation(self, anno, anno_path):
        # Save Annotation
        with open(anno_path, "w") as file:
            for box in anno:
                box = " ".join(box)
                file.write(box + "\n")

    def create_dataset(self, save_dir="./data/NATT/dataset"):
        
        os.makedirs(save_dir, exist_ok=True)

        # Create Dataset
        for image_path, anno_path in self.database.items():
            # Read Image and Annotation
            image = self.read_image(image_path)
            anno = self.read_annotation(anno_path)
            # Crop Image
            cropped_images = self.crop_image(image, anno)

            # Make save path
            right_path = image_path.split("NATT360JPEG")[-1]
            save_path = os.path.join(save_dir, right_path)

            # Save Cropped Image
            self.save_cropped_image(cropped_images[0], save_path)

        print("Dataset Created!")
    
if __name__ == "__main__":
    data_dir = "./data/NATT/"
    db = Database(data_dir)
    print("Total Images: ", len(db.image_list))
    print("Total Annotations: ", len(db.anno_list))
    print("Total Images with Annotation: ", len(db.database))

    dp = DatasetPreparation(data_dir)
    dp.create_dataset()
    
