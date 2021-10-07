import os
import glob
import pdb
import pickle
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

class SegInfDataset(Dataset):

    def __init__(self, images, save_path, org_height, org_width, resize=512):

        self.org_width = org_width
        self.org_height = org_height
        self.save_path = save_path
        if save_path !=None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        self.resize = resize
        self.files = images
        self.data_size = len(self.files)
        self._generate_data()

    def _generate_data(self):
        
        self.images = []
        for idx, image in enumerate(self.files):

            # cv2.imshow('org image', image)
            # cv2.waitKey()


            (thresh, image) = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

            image = cv2.medianBlur(image, 3)

            # cv2.imshow('sharpen image', image)
            # cv2.waitKey()

            #pdb.set_trace()

            # paste in the square background 
            img_h, img_w = image.shape

            image = cv2.resize(image, (self.org_width, self.org_height), interpolation=cv2.INTER_AREA)

            #b_img = np.zeros([self.org_height,self.org_height],dtype=np.uint8)
            #b_img.fill(255)

            #b_img[0:image.shape[0], 0:image.shape[1]] = image

            # cv2.imshow('org image',b_img)
            # cv2.waitKey()

            ### cut the image from bottom 
            # b_img = image[image.shape[0]-image.shape[1]:image.shape[0],:]

            if self.save_path != None:
                cv2.imwrite(os.path.join(self.save_path,str(idx)+'.png'),image)


            b_img = cv2.resize(image, (self.resize, self.resize))

            # cv2.imshow('org image',b_img)
            # cv2.waitKey()

            self.images.append(b_img)


    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):

        inputs = self.images[idx]
                
        inputs = (inputs.astype(float)-127.5)/127.5

        inputs = torch.tensor(inputs).unsqueeze(0).float()

        return inputs

if __name__ == "__main__":

    # for i in CARD_TYPE:
    
    #     ImageCreate_seg_fullintyg(i, "./fk_doc_data/new-template")

    data_path = 'fk_doc_data/seg_data/test'
    
    test_images = glob.glob(data_path+'/*.jpg')

    input_images = []
    for img in test_images:
        input_images.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    
    SegInfDataset(input_images)
    

