import glob
import cv2
import os
import pdb
import numpy as np
from statistics import mode

import pdb
from model import SegPredModel
from dataset import SegInfDataset
from torch.utils.data import DataLoader
import torch
import ast

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.measure import find_contours
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import color, morphology

from skimage.morphology import binary_dilation, binary_erosion, convex_hull_image, label

def find_box(image, dilate=3, erosion=0, connectivity=2, max_object_num=1):

    if image.ndim > 2:
        raise ValueError("Input must be a 2D image")

    if connectivity not in (1, 2):
        raise ValueError('`connectivity` must be either 1 or 2.')

    # binary_mask = binary_opening(input)
    
    image = binary_dilation(image, selem=np.ones((dilate, dilate)))

    labeled_im = label(image, connectivity=connectivity, background=0)
    convex_obj = np.zeros(image.shape, dtype=bool)
    convex_img = np.zeros(image.shape, dtype=bool)

    area = []
    convex_objs = []
    for i in range(1, labeled_im.max() + 1):
        convex_obj = convex_hull_image(labeled_im == i)
        area.append((convex_obj==True).sum())
        
        convex_objs.append(convex_obj)

    convex_objs = [x for _,x in sorted(zip(area,convex_objs), reverse=True)]

    convex_objs = convex_objs[0:max_object_num]

    for i in convex_objs:
        convex_img = np.logical_or(convex_img, i)

    if erosion > 0:
        convex_img = binary_erosion(convex_img, selem=np.ones((erosion, erosion)))

    convex_label = label(convex_img, connectivity=connectivity, background=0)
    
    return convex_img, convex_label


def merge_class(mask, dilate=3, erosion=3, connectivity=2, max_object_num=10):

    binary_mask = (mask>0).astype(mask.dtype)
    
    binary_mask, label_mask = find_box(binary_mask, dilate, erosion, max_object_num=10)

    for i in range(1, label_mask.max() + 1):

        temp_mask = mask[label_mask==i]

        temp_mode = mode(temp_mask)

        changed_mask = [temp_mode if x!=0 else 0 for x in temp_mask]

        mask[label_mask==i] = changed_mask

    return mask
        
    

def draw_color_masks(image,masks,output):
    
    color_mask = np.zeros((image.shape[0], image.shape[1], 3))
    color_mask[masks[0]] = [255, 0, 0]  
    color_mask[masks[1]] = [255, 128, 0] 
    color_mask[masks[2]] = [255, 255, 0] 
    color_mask[masks[3]] = [128, 255, 0] 
    color_mask[masks[4]] = [0, 255, 0]  
    color_mask[masks[5]] = [0, 255, 255] 
    color_mask[masks[6]] = [0, 0, 255] 
    color_mask[masks[7]] = [127, 0, 255] 
    color_mask[masks[8]] = [255, 51, 255]  
    
    img_color = np.dstack((image, image, image))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.3

    img_masked = color.hsv2rgb(img_hsv)*255

    cv2.imwrite(output, img_masked)


def predict(model, data, type='real'):

    trained_model = SegPredModel(path_to_model=model_path, device='cpu')
    
    # print model hyperparameters
    print(trained_model.hparams)

    image_params = trained_model.hparams['image']

    image_params = ast.literal_eval(str(image_params))

    if type == 'real':
        data_path = os.path.join(data_folder, 'test_real')

        test_images = glob.glob(data_path+'/*.jpg')
        test_images.sort()

        input_images = []
        for img in test_images:
            input_images.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))

    
        dataset=SegInfDataset(input_images, data_path, image_params['heigh'], image_params['width'], image_params['resize'])

        inputs = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    ## get the cut image or extend images
    preprocessed_files = glob.glob(data_path+'/image'+'/*.jpg')
    preprocessed_files.sort()
    preprocessed_images = []


    for img in preprocessed_files:
        
        pre_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    
        preprocessed_images.append(pre_img)

    result_path = os.path.join(data_path,'result_test_merge')

    if not os.path.exists(result_path):
        os.makedirs(result_path)


    for idx, input in enumerate(inputs):

        pre_img = preprocessed_images[idx]

        pred_mask, pre_label = trained_model.predict(input)

        np_mask = cv2.resize(pred_mask, (pre_img.shape[0], pre_img.shape[1]), interpolation=cv2.INTER_NEAREST)

        np_mask = merge_class(np_mask, dilate=3, connectivity=2, max_object_num=10)

        masks = []
        for i in range(10):

            if i == 0:
                continue

            temp_binary_image = np.zeros((np_mask.shape[0], np_mask.shape[1]))
            temp_binary_image[np_mask==i]= 1

            mask, _ = find_box(temp_binary_image.astype(np.uint8))

            masks.append(mask)
        

        output_path = os.path.join(result_path, str(idx) + "-mask.png")

        draw_color_masks(pre_img, masks, output_path)
    


if __name__ == "__main__":

    data_folder = '../../data/seg_data/'
    model_path = '../../data/model/epoch=122-step=51290.ckpt'


    predict(model_path, data_folder, type='real')
