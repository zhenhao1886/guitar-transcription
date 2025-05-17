import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import move_targets_to_device, maskRCNN_class_names

class Evaluator:
    def __init__(self, model):
        self.model = model
        self.results = {}
        self.pred_results = None

    def evaluate(self, test_dataloader):

        pred_results = get_predictions(self.model, test_dataloader, 'cuda')
        move_targets_to_device(pred_results, 'cpu')
        
        organised_pred_results = organise_data(pred_results, len(maskRCNN_class_names), len(test_dataloader.dataset), pred=True)
        
        act_targets = []
        for ds in test_dataloader.dataset:
            act_targets.append(ds[1])

        organised_actual_results = organise_data(act_targets, len(maskRCNN_class_names), len(test_dataloader.dataset), pred=False)

        self.pred_results = pred_results

        mARs = []
        mAPs = []
        x_ls, y_ls = [], []
        fig, ax = plt.subplots(figsize=(10,10))

        for t in np.arange(0.5, 1, 0.05):
            detection_df = get_detection_df(organised_pred_results, organised_actual_results, IOU_threshold = t)
            prec_recall_ls = []
            ap_scores = []

            for df in detection_df:
                if len(df) > 0:
                    prec_recall = calculate_prec_recall(df)
                    prec_recall_ls.append(prec_recall)
                    voc_pr = VOC_pr_curve(np.array(list(zip(prec_recall.recall, prec_recall.precision))))
                    ap, x, y = VOC_interp(voc_pr)
                    ap_scores.append(ap)
                    if t == 0.5:
                        ax.plot(x, y, label = maskRCNN_class_names[df.iloc[0]['cls']])
                        x_ls.append(x)
                        y_ls.append(y)

            mARs.append(compute_mAR(prec_recall_ls, organised_actual_results))
            mAPs.append(np.mean(ap_scores))

        ax.plot(x_ls[0], np.mean(y_ls, axis = 0), label = f'mAP@0.5 = {mAPs[0]:.3f}', linewidth=3)
        plt.title('Precision Recall for IOU=0.5')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()

        self.results['mAR@0.5:0.95'] = np.sum(mARs)/10
        self.results['mAP@0.5:0.95'] = np.sum(mAPs)/10
        self.results['mAR@0.5'] = mARs[0]
        self.results['mAP@0.5'] = mAPs[0]


def compute_IOU(pred_mask, gt_mask):
    eps = 1e-7
    if pred_mask is None or gt_mask is None:
        return 0.0

    intersection = torch.logical_and(pred_mask, gt_mask).sum().item()
    union = torch.logical_or(pred_mask, gt_mask).sum().item()
    return intersection / (union + eps)


def get_AP_for_class_i_per_image(pred_masks, gt_masks, pred_scores, iou_threshold):
    detections = []
    matched_gt_indices = set()

    for i, pred_mask in enumerate(pred_masks):
        best_iou = 0
        best_j = None

        for j, gt_mask in enumerate(gt_masks):
            if j in matched_gt_indices:
                continue

            iou = compute_IOU(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j is not None:
            matched_gt_indices.add(best_j)
            detections.append([i, best_j, pred_scores[i].item(), best_iou])  # TP
        else:
            detections.append([i, None, pred_scores[i].item(), best_iou])  # FP

    return detections

def get_predictions(model, test_dataloader, device):
    #model is of a maskedRCNN class
    results = []
    times = []
    with torch.no_grad():
        for k, (batch, targets) in enumerate(test_dataloader):
            batch = batch[0].unsqueeze(0).to(device)
            move_targets_to_device(targets, device)
            model.eval()
            start = time.time()
            result = model(batch, None)
            end = time.time()
            results.append(result[0])
            times.append(end-start)
    print(np.mean(times))
    
    return results

def organise_data(data, num_classes, num_imgs, pred=True):
    # get images predictions by classes
    # data: {boxes: ..., labels: ..., scores: ..., masks: ...}
    # num_classes includes background class as class 1. There are 21 classes
    # {1: [[img_id, bbox_id, pred_masks, pred_bbox, conf], ...], 2: ...}
    collated = {i: {img_id: [] for img_id in range(num_imgs)} for i in range(1, 1+num_classes)}
    
    for img_id, result in enumerate(data):
        boxes = result['boxes']
        labels = result['labels']
        masks = result['masks']
        if pred:
            conf = result['scores']
        
        if not pred:
            for i, (label, mask, box) in enumerate(zip(labels, masks, boxes)):
                collated[label.item()][img_id].append([box, mask])
        else:
            for i, (label, mask, box, conf) in enumerate(zip(labels, masks, boxes, conf)):
                collated[label.item()][img_id].append([box, conf, mask])
                
    return collated

    
def get_detection_df(predicted, actuals, IOU_threshold = 0.5):
    ls2 = []
    for cls in predicted.keys():
        ls = []
        for img_id in predicted[cls].keys():
            # if len(predicted[cls][img_id]) < 2 or len(actuals[cls][img_id]) < 1:
            #     continue

            pred_masks = [data[2] for data in predicted[cls][img_id]]
            actual_masks = [data[1].type(torch.uint8) for data in actuals[cls][img_id]]
            scores = [data[1] for data in predicted[cls][img_id]]
            detections = get_AP_for_class_i_per_image(pred_masks, 
                                                      actual_masks, 
                                                      scores, 
                                                      IOU_threshold)
            
            detections_df = pd.DataFrame(detections, columns = ['pred','actual','conf','iou'])
            detections_df['img_id'] = img_id
            detections_df['cls'] = cls
            ls.append(detections_df)
        detections_df_cls = pd.concat(ls, axis=0)
        detections_df_cls.sort_values('conf', ascending=False, inplace=True)
        ls2.append(detections_df_cls)
    return ls2


def calculate_prec_recall(detections_df, IOU_threshold=0.5):
    """
    detections_df: DataFrame with columns ['pred', 'actual', 'conf', 'iou', 'img_id', 'cls']
    """
    evaluation = []
    matched_gt = set()  # Set of (img_id, gt_index)

    for i, row in detections_df.iterrows():
        img_id = row['img_id']
        gt_idx = row['actual']

        if row['iou'] >= IOU_threshold and gt_idx is not None:
            match_key = (img_id, gt_idx)
            if match_key not in matched_gt:
                evaluation.append('TP')
                matched_gt.add(match_key)
            else:
                evaluation.append('FP')  # Duplicate match to the same GT
        else:
            evaluation.append('FP')  # No match or low IoU

    detections_df['evaluation'] = evaluation

    # Compute cumulative TP and FP
    detections_df['tp'] = detections_df['evaluation'] == 'TP'
    detections_df['fp'] = detections_df['evaluation'] == 'FP'
    detections_df['acc_tp'] = detections_df['tp'].cumsum()
    detections_df['acc_fp'] = detections_df['fp'].cumsum()

    # Compute precision and recall
    detections_df['precision'] = detections_df['acc_tp'] / (detections_df['acc_tp'] + detections_df['acc_fp'] + 1e-7)
    detections_df['recall'] = detections_df['acc_tp'] / (detections_df['tp'].sum() + 1e-7)

    return detections_df

def compute_mAR(detections_df, organised_actual_results):
    total_tps = 0
    total_gt = count_total_gt_boxes(organised_actual_results)
    for df in detections_df:
        total_tps += (df.evaluation == 'TP').sum()
        
    return total_tps/total_gt

def count_total_gt_boxes(organised_actual_results):
    s = 0
    for img_id, masks in organised_actual_results.items():
        for class_, mask in masks.items():
            s += len(mask)
    return s

# VOC 2010
def VOC_pr_curve(pr_points):
    
    new_curve = np.zeros_like(pr_points)
    precision = pr_points[:,1]
    prec = np.array([np.max(precision[i:]) for i in range(len(precision))])
    
    recall = pr_points[:,0]
    return np.stack([recall, prec], axis=1)


def VOC_interp(pr_points):
    #pr_points - 2d array of (recall, precision)
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    new_rec = np.concatenate(([0], pr_points[:,0], [1]))
    new_prec = np.concatenate(([1], pr_points[:,1], [0]))
    values = np.interp(x, new_rec, new_prec)
    AP = np.trapz(np.interp(x, new_rec, new_prec), x)
    return AP, x, values




