from ultralytics import YOLO
from model import *
import torch
from torchvision import tv_tensors
import numpy as np
from utils import maskRCNN_class_names, maskIntersect, get_fret
import cv2


class Predictor:
    def __init__(self, model_name: str = 'yolo', path_to_weights = 'default', device: str = 'cuda', threshold: float = 0.5):
        self.model_name = model_name
        self.path_to_weights = path_to_weights
        self.path_to_weights_ = None
        self.model = None
        self.device = device
        self.num_classes = None
        self.class_names = maskRCNN_class_names
        self.threshold = threshold
        self.get_model()
        
    def get_model(self):
        if self.model_name.lower() == 'yolo':
            if self.path_to_weights == 'default':
                self.path_to_weights_ = ["./runs/segment/train12/weights/best.pt","./runs/segment/train11/weights/best.pt"]
                
            else:
                self.path_to_weights_ = self.path_to_weights
            self.model = [YOLO(self.path_to_weights_[0]), YOLO(self.path_to_weights_[1])]
            self.class_names = self.model[0].names
            self.num_classes = len(self.class_names)
    
        elif self.model_name.lower() == 'vit_maskrcnn':
            if self.path_to_weights == 'default':
                self.path_to_weights_ = './output/ViT_21_class_2025_04_22_16_07_33/model_150.pt'
            else:
                self.path_to_weights_ = self.path_to_weights
            self.class_names = maskRCNN_class_names
            self.num_classes = len(self.class_names)
            self.model = ViT_MaskedRCNN(self.num_classes)
            self.model.load_state_dict(torch.load(self.path_to_weights_, weights_only=True))
            self.model.to(self.device)
            
        elif self.model_name.lower() == 'resnet_maskrcnn':
            if self.path_to_weights == 'default':
                self.path_to_weights_ = './output/MaskRCNN_2025_04_22_11_10_04/model_50.pt'
            else:
                self.path_to_weights_ = self.path_to_weights
            self.class_names = maskRCNN_class_names
            self.num_classes = len(self.class_names)
            self.model = MaskedRCNN(self.num_classes)
            self.model.load_state_dict(torch.load(self.path_to_weights_, weights_only=True))
            self.model.to(self.device)
            
    def predict_masks(self, X, batch_size = 1):
        # X can be a tensor, numpy array, or a filepath
        # returns a dictionary object - boxes, scores, masks, label_idx
        # boxes: [Nx4] tensor
        # scores and label_idx: [N] tensor
        # masks: [NxHxW] array 
        
        if isinstance(X, str):
            if self.model_name.lower() == 'yolo':
                d = {'boxes': None, 'scores': None, 'masks': None, 'label_idx': None}
                for i, model in enumerate(self.model):
                    result = model(X, verbose=False, batch = batch_size, conf = self.threshold, iou = self.threshold)[0]
                    if result:
                        if d['boxes'] is None:
                            if result.boxes:
                                d['boxes'] = result.boxes.xyxy.cpu().detach()
                                d['scores'] = result.boxes.conf.cpu().detach()
                                d['label_idx'] = result.boxes.cls.cpu().detach()
                            if result.masks:
                                d['masks'] = result.masks.data.cpu().detach()
                            
                        else:
                            if result.boxes:
                                d['boxes'] = torch.cat((d['boxes'], result.boxes.xyxy.cpu().detach()),0)
                                d['scores'] = torch.cat((d['scores'], result.boxes.conf.cpu().detach()),0)
                                d['label_idx'] = torch.cat((d['label_idx'], result.boxes.cls.cpu().detach()),0)
                            if result.masks:
                                d['masks'] = torch.cat((d['masks'], result.masks.data.cpu().detach()),0)
                return d
                
            else:
                img_tensor = torchvision.io.decode_image(X)/255
                with torch.no_grad():
                    self.model.eval()
                    result = self.model(img_tensor.unsqueeze(0).to(self.device), None)[0]
                    return dict(zip(['boxes','scores','masks','label_idx'], 
                                [result['boxes'].cpu().detach(), 
                                 result['scores'].cpu().detach(), 
                                 result['masks'].cpu().detach(), 
                                 result['labels'].cpu().detach()]))
        else:
            if self.model_name.lower() == 'yolo':
                d = {'boxes': None, 'scores': None, 'masks': None, 'label_idx': None}
                for i, model in enumerate(self.model):
                    result = model(X, verbose=False, batch = batch_size, conf = self.threshold, iou = self.threshold)[0]
                    if result:
                        if d['boxes'] is None:
                            if result.boxes:
                                d['boxes'] = result.boxes.xyxy.cpu().detach()
                                d['scores'] = result.boxes.conf.cpu().detach()
                                d['label_idx'] = result.boxes.cls.cpu().detach()
                            if result.masks:
                                d['masks'] = result.masks.data.cpu().detach()
                            
                        else:
                            if result.boxes:
                                d['boxes'] = torch.cat((d['boxes'], result.boxes.xyxy.cpu().detach()),0)
                                d['scores'] = torch.cat((d['scores'], result.boxes.conf.cpu().detach()),0)
                                d['label_idx'] = torch.cat((d['label_idx'], result.boxes.cls.cpu().detach()),0)
                            if result.masks:
                                d['masks'] = torch.cat((d['masks'], result.masks.data.cpu().detach()),0)
                            
                        
                return d
            else:
                with torch.no_grad():
                    X = torch.Tensor(X)/255 #3 x W x H
                    X = X.permute(2,0,1)
                    X = tv_tensors.Image(X)
                    self.model.eval()
                    result = self.model(X.unsqueeze(0).to(self.device), None)[0]
                    
                    return dict(zip(['boxes','scores','masks','label_idx'], 
                                    [result['boxes'].cpu().detach(), 
                                     result['scores'].cpu().detach(), 
                                     result['masks'].cpu().detach(), 
                                     result['labels'].cpu().detach()]))

    def get_scaled_and_filtered_masks(self, prediction_output, test_image = None, conf_threshold = 0.4):
        # Filter the output based on the confidence threshold
        # prediction_output: dictionary of scores, labels, boxes and masks
        # boxes: [Nx4] tensor
        # scores and label_idx: [N] tensor
        # masks: [NxHxW] array 
        
        # test image: image passed into the model
        # conf_threshold: score threshold for prediction output, and for the binarization.
        # Returns scaled and filtered masks based on confidence score

        # pick boxes which have scores > conf threshold
        # if there are repeated boxes, choose the one with higher conf

        # get unique key: score dictionary
        
        
        if self.model_name.lower() == 'resnet_maskrcnn' or self.model_name.lower() == 'vit_maskrcnn':
            d = {}
            for i, (label, conf_) in enumerate(zip(prediction_output['label_idx'], prediction_output['scores'])):
                if conf_ > conf_threshold:
                    if label.item() not in list(d.keys()):
                        d[label.item()] = (i,conf_)
                    elif conf_ > d[label.item()][1]:
                        d[label.item()] = (i,conf_)
            
            #filtered_labels = list(d.keys())
            filtered_idx = [val[0] for val in list(d.values())]

            scores_mask = prediction_output['scores'] > conf_threshold
            # loop through d
            #scores_mask = [False for _ in range(len(prediction_output['label_idx']))]
            
            #for key, (i, conf) in d.items():
            #    scores_mask[i] = True
            
            # Scale the predicted bounding boxes
            # 360 x 640
            pred_bboxes = tv_tensors.BoundingBoxes(prediction_output['boxes'][scores_mask], format='xyxy', canvas_size=test_image.shape[0:2])
            
            # Get the class names for the predicted label indices
            pred_labels = [int(label) for label in prediction_output['label_idx'][scores_mask]]
            
            # Extract the confidence scores
            pred_scores = prediction_output['scores'][scores_mask]
            
            # Scale and stack the predicted segmentation masks
            pred_masks = []
            pred_masks2 = []
            if prediction_output['masks'] is not None:
                pred_masks = F2.interpolate(prediction_output['masks'][scores_mask], size=test_image.shape[0:2])
                for mask in pred_masks:
                    m = tv_tensors.Mask(torch.where(mask >= conf_threshold, 1, 0), dtype=torch.bool)
                    pred_masks2.append(m)
                if len(pred_masks2) > 0:
                    pred_masks = torch.concat(pred_masks2)
                    
            return dict(zip(['masks','boxes','label_idx','scores'],[pred_masks, pred_bboxes,pred_labels,pred_scores]))
            
        else:

            return dict(zip(['masks','boxes','label_idx','scores'], 
                            [prediction_output['masks'], 
                             prediction_output['boxes'], 
                             prediction_output['label_idx'].numpy(), 
                             prediction_output['scores']]))


        
    def get_string_fret_region(self, masks, classes):
        """
        masks: must be in the form of NxHxW numpy array
        classes: [N] label_idx array
        """
        # results
        string_masks, fret_masks, hands_mask = {}, {}, {}

        # if its a yolo model
        if self.model_name.lower() == 'yolo':
            for mask, class_ in zip(masks, classes):
                if class_ == 0: 
                    hands_mask[class_] = mask
                elif class_ in range(1,13): #fret
                    if class_ in list(fret_masks.keys()):
                        fret_masks[class_].append(mask)
                    else:
                        fret_masks[class_] = [mask]
                elif class_ in range(14, 20):
                    if class_ in list(string_masks.keys()):
                        string_masks[class_].append(mask)
                    else:
                        string_masks[class_]= [mask]
        else:
            for mask, class_ in zip(masks, classes):
                if class_ == 1:
                    hands_mask[class_] = mask
                elif class_ in range(2,14): #fret
                    if class_ in list(fret_masks.keys()):
                        fret_masks[class_].append(mask)
                    else:
                        fret_masks[class_] = [mask]
                elif class_ in range(15, 21):
                    if class_ in list(string_masks.keys()):
                        string_masks[class_].append(mask)
                    else:
                        string_masks[class_]= [mask]
        
                    
        fret_string_boxes = {}
        for zone_key, fret_mask in fret_masks.items():
            for string_key, string_mask in string_masks.items():
                best_string_mask = string_mask[0] #take the 2D w x h
                best_fret_mask = fret_mask[0]
                #print(best_string_mask)
                #print(best_fret_mask)
                intersection = np.logical_and(best_fret_mask, best_string_mask)
                #intersection = maskIntersect(np.zeros_like(string_mask, dtype=np.int32), best_fret_mask.astype(np.int32), best_string_mask.astype(np.int32))
                fret_string_boxes[(string_key, zone_key)] = intersection

        return fret_string_boxes

def predict_finger_location_of_left_hand(img, hand_landmarks, mp_hands, fret_string_boxes, class_names):
    """
    img: numpy array with 3 channels
    fingertip: 2D array (5,2) (xy coordinates)
    
    initialize: 
    effectivePresses <- []
    for each finger:
        candidates <- dict()
        for each string-fret bounding region:
            if IOU(bounding region, finger) > THRESHOLD:
                if string in candidates.keys():
                    candidates[string] = max((dict[string], fret))
        effectivePresses.append(candidates)

    fret string boxes need to be transformed downwards
    """
    
    fingertip_maps = {4: [], 8: [], 12: [], 16: [], 20: []}

    hand_landmark_indx = [4,8,12,16,20]
    fingertips = np.zeros((len(hand_landmark_indx),2))
                          
    #get the fingertips and index finger
    for j,i in enumerate(hand_landmark_indx):
        fingertips[j,0] = hand_landmarks.landmark[mp_hands.HandLandmark(i)].x * img.shape[1]
        fingertips[j,1] = hand_landmarks.landmark[mp_hands.HandLandmark(i)].y * img.shape[0]

    """
    maps each fingertip to a string and a fret.
    fingertip mask intersect fret mask
    """
    strings = {}
    for key, string_fret_bounding_region in fret_string_boxes.items():
        for k, finger_zone in enumerate(fingertips):
            if finger_zone[1] > string_fret_bounding_region.shape[0] or finger_zone[1] < 0 or finger_zone[0] < 0 or finger_zone[0] > string_fret_bounding_region.shape[1]:
                continue
            else:
                if check_patch(string_fret_bounding_region, finger_zone.astype(int)[1], finger_zone.astype(int)[0], patch_width = 1) > 0:
                    #if len(fingertip_maps[4]) + len(fingertip_maps[8]) + len(fingertip_maps[12]) + len(fingertip_maps[16]) + len(fingertip_maps[20]) == 0:
                        #fingertip_maps[hand_landmark_indx[k]] = [(key[0],key[1])]
                    if key[0] not in strings.keys():
                        strings[key[0]] = get_fret(key[1], class_names)
                    else:
                        strings[key[0]] = max(get_fret(key[1], class_names), strings[key[0]])

    return strings #fingertip_maps

def check_patch(string_fret_bounding_region, x, y, patch_width = 1):
    #checks a 2*patch_width + (x,y) patch for any 1s
    x = string_fret_bounding_region[
    max(0,x-patch_width):min(x+patch_width,string_fret_bounding_region.shape[1]), 
    max(0,y-patch_width):min(y+patch_width,string_fret_bounding_region.shape[0])].sum()
    return x
    
    