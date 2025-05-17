import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import os
import numpy as np
import pandas as pd
import cv2

def read_coco_text_file(filename):
    """
    returns a list of masks and list of corresponding labels
    background is 0
    so there are 5 classes
    """
    ann, label = [], []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            text = f.read()
            ls = text.strip().split('\n')
            for line in ls:
                values = line.strip().split(' ')
                xy = list(map(float, values[1:]))
                xy = np.array(xy).reshape(-1,2)
                cls_id = int(values[0]) + 1

                ann.append(xy)
                label.append(cls_id)
                
            f.close()

    
    # background class if filename does not exist
    
    return (ann, label)
    
class GuitarDataset(Dataset):
    def __init__(self, dataset_dir, n_classes):
        super(Dataset, self).__init__()
        
        #do this for each train, test and val
        self.dataset_dir = None

        #dataset dir
        if os.path.exists(dataset_dir):
            self.dataset_dir = dataset_dir

        # using coco dataset format
        self.masks = [os.path.join(self.dataset_dir,'labels',mask_path) for mask_path in os.listdir(os.path.join(self.dataset_dir, 'labels'))]
        self.images = [os.path.join(self.dataset_dir,'images',img_path) for img_path in os.listdir(os.path.join(self.dataset_dir, 'images'))]

        self.dict = {}
        self.n_classes = n_classes

        for img in self.images:
            img_name = os.path.basename(img)[:-4] 
            self.dict[img_name] = read_coco_text_file(os.path.join(self.dataset_dir, 'labels', img_name + '.txt'))

        # transforms is not needed. because we already augmented.
        

    def __len__(self):
        return len(self.images)

    def get_bbox_from_mask(self, masks):
        # masks is list of N polygons
        areas = torch.zeros(len(masks))
        boxes = torch.zeros((len(masks), 4))
        for i, mask in enumerate(masks):
            #print(mask)
            
            x0 = np.min(mask[:,0])
            y0 = np.min(mask[:,1])
            x1 = np.max(mask[:,0])
            y1 = np.max(mask[:,1])
            boxes[i,:] = torch.tensor([x0, y0, x1, y1])
            areas[i] = (y1-y0)*(x1-x0)
            
        return boxes, areas

    def __getitem__(self, i):
        img_tensor = decode_image(self.images[i])/255
        mask_, labels = self.dict[os.path.basename(self.images[i])[:-4]]
        mask, poly_ls = self.transform_mask(mask_) 
        boxes, areas = self.get_bbox_from_mask(poly_ls)
        #print(masks_to_boxes(torch.Tensor(mask)))
        #mask = torch.tensor(mask)

        # return these:
        """
        image: torchvision.tv_tensors.Image of shape [3, H, W], a pure tensor, or a PIL Image of size (H, W)
        
        target: a dict containing the following fields
        boxes, torchvision.tv_tensors.BoundingBoxes of shape [N, 4]: the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, 
        ranging from 0 to W and 0 to H
        
        labels, integer torch.Tensor of shape [N]: the label for each bounding box. 0 represents always the background class.
        
        image_id, int: an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
        
        area, float torch.Tensor of shape [N]: the area of the bounding box. 
        This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
        
        iscrowd, uint8 torch.Tensor of shape [N]: instances with iscrowd=True will be ignored during evaluation.
        
        (optionally) masks, torchvision.tv_tensors.Mask of shape [N, H, W]: the segmentation masks for each one of the objects

        """
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img_tensor))
        target["masks"] = tv_tensors.Mask(mask) if len(mask.shape) >= 3 else tv_tensors.Mask(np.array([]).reshape(-1, img_tensor.shape[1], img_tensor.shape[2]))
        target["labels"] = torch.Tensor(labels).type(torch.int64)
        target["image_id"] = i
        target["area"] = areas
        target["iscrowd"] = torch.zeros(len(labels), dtype = torch.uint8)
        return img_tensor, target

    def transform_mask(self, mask, img_size = 640):
        """
        mask is of the COCO type
        {1: np.array(Nx2):[[float from 0 to 1, float from 0 to 1], [...], ..], 2: [[x,y],[x,y],...]}
        mask is now:
        ann[]
        converts a polygon to mask
        
        """
        array = []
        #array = np.zeros((num_classes, img_size, img_size))
        poly_ls = []
        for polygon in mask:
            poly = (polygon * img_size).astype(np.int32)
            mask2 = cv2.fillPoly(np.zeros((img_size, img_size), dtype = np.uint8), [poly], 1)
            #array[cls-1,:,:] = mask2
            array.append(mask2)
            poly_ls.append(poly)

        return np.array(array), poly_ls
            
        