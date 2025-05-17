import cv2
import numpy as np
import matplotlib.pyplot as plt
import distinctipy
import torch
import torch.nn.functional as F2
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors

maskRCNN_class_names = {
    0: 'Background',
    1: 'Hand',
  2: 'Zone1',
  3: 'Zone10',
  4: 'Zone11',
  5: 'Zone12',
  6: 'Zone2',
  7: 'Zone3',
  8: 'Zone4',
  9: 'Zone5',
  10: 'Zone6',
  11: 'Zone7',
  12: 'Zone8',
  13: 'Zone9',
  14: 'fretboard',
  15: 'string1E',
  16: 'string2A',
  17: 'string3D',
  18: 'string4G',
  19: 'string5B',
  20: 'string6E'}

def get_colors(num_classes):
    # Generate a list of colors with a length equal to the number of labels
    colors = distinctipy.get_colors(num_classes) 
    # Make a copy of the color map in integer format
    int_colors = [tuple(int(c*255) for c in color) for color in colors]
    
    return int_colors
    
def move_targets_to_device(targets, device):
    for target in targets:
        if 'boxes' in list(target.keys()):
            target['boxes'] = target['boxes'].to(device)
        if 'labels' in list(target.keys()):
            target['labels'] = target['labels'].to(device)
        if 'masks' in list(target.keys()):
            target['masks'] = target['masks'].to(device)
        if 'iscrowd' in list(target.keys()):
            target['iscrowd'] = target['iscrowd'].to(device)
        if 'area' in list(target.keys()):
            target['area'] = target['area'].to(device)
        if 'scores' in list(target.keys()):
            target['scores'] = target['scores'].to(device)
    
def maskIntersect(original_image, fret_mask, string_mask):
    """
    fret_contours: [Fret], 
    string_contours: [String1]
    
    """
    
    # Create image filled with zeros the same size of original image
    blank = np.zeros_like(original_image).astype(np.int32)

    # Copy each mask into its own image and fill it with '1'
    fret = cv2.fillPoly(blank.copy(), [fret_mask], 1)
    string = cv2.fillPoly(blank.copy(), [string_mask], 1)
    
    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to it,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didnt intersect
    intersection = np.logical_and(fret, string)
    
    return intersection


def show_string_fret_regions(image, string_fret_dict, colors, class_names, model_name):
    """
    image: HxW
    string_fret_dict: tuple (string, fret): 3D mask of size 1 x (H+P1) x (WxP2)) 
    colors: distinctipy colours in integer form
    """
    image2 = image.copy()
    alpha = 0.4

    if len(string_fret_dict) == 0:
        image2 = cv2.addWeighted(image2, 1 - alpha, np.zeros_like(image2), alpha, 0)
        return image2

    colored_mask = np.zeros_like(image2)
    for i, region in string_fret_dict.items():
        s, f = i[0], get_fret(i[1], class_names)
        if s % 2==0 and f %2 == 0:
            color = colors[0]
        elif s % 2==1 and f %2 == 0:
            color = colors[1]
        elif s % 2==1 and f %2 == 1:
            color = colors[2]
        else:
            color = colors[3]
        colored_mask[region.squeeze() == 1] = color
            
    image2 = cv2.addWeighted(image2, 1 - alpha, colored_mask, alpha, 0)

    return image2
    
def get_fret(class_idx, class_names):
    zone = class_names[class_idx] #zone1, zone12 etc
    return int(zone[4:])

def transform_string_fret_region(image, string_fret_dict):
    orig_shape_y, orig_shape_x, _ = image.shape
    mask_shape_y, mask_shape_x = list(string_fret_dict.values())[0].squeeze().shape
    if orig_shape_y > orig_shape_x:
        inter_shape_y = 640
        inter_shape_x = (1 + (inter_shape_y * orig_shape_x / orig_shape_y - 1)//32)*32
    else:
        inter_shape_x = 640
        inter_shape_y = (1 + (inter_shape_x * orig_shape_y / orig_shape_x - 1)//32)*32
    
    padding_x = int((mask_shape_x - inter_shape_x)/2)
    padding_y = int((mask_shape_y - inter_shape_y)/2)

    
    for i, region in string_fret_dict.items():
        inter_mask = region[padding_y:mask_shape_y-padding_y, padding_x:mask_shape_x-padding_x]
        #downsize the mask to fit the image.
        new_mask = cv2.resize(inter_mask.astype(np.uint8), (orig_shape_x, orig_shape_y), interpolation=cv2.INTER_NEAREST)
        string_fret_dict[i] = new_mask
    return string_fret_dict
    
def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: tuple = (0, 0),
    font_scale: float = 0.75,
    font_thickness: float = 2,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: tuple = (255, 255, 255),
    bg_color_rgb: tuple | None = None,
    outline_color_rgb: tuple | None = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb

def plot_results(test_image, test_data, pred, threshold = 0.5):
    
    cmap = get_colors(len(maskRCNN_class_names))
    # Filter the output based on the confidence threshold
    scores_mask = pred['scores'] > threshold
    
    # Scale the predicted bounding boxes
    pred_bboxes = tv_tensors.BoundingBoxes(pred['boxes'][scores_mask], format='xyxy', canvas_size=test_image.size())
    act_bboxes = tv_tensors.BoundingBoxes(test_data['boxes'], format='xyxy', canvas_size=test_image.size())
    
    # Get the class names for the predicted label indices
    pred_labels = [maskRCNN_class_names[int(label)] for label in pred['labels'][scores_mask]]
    act_labels = [maskRCNN_class_names[int(label)] for label in test_data['labels']]
    
    # Extract the confidence scores
    pred_scores = pred['scores']
    
    # Scale and stack the predicted segmentation masks
    pred_masks = F2.interpolate(pred['masks'][scores_mask], size=test_image.size()[1:])

    filtered_masks = [tv_tensors.Mask(torch.where(mask >= threshold, 1, 0), dtype=torch.bool) for mask in pred_masks]
    if len(filtered_masks) > 0:
        pred_masks = torch.concat(filtered_masks)
        # Annotate the test image with the predicted segmentation masks
        pred_annotated_tensor = draw_segmentation_masks(image=test_image, masks=pred_masks, alpha=0.9, 
                                                        colors=cmap)
        
        # Annotate the test image with the predicted labels and bounding boxes
        pred_annotated_tensor = draw_bounding_boxes(
            image=pred_annotated_tensor, 
            boxes=pred_bboxes, 
            labels=[f"{label}\n{prob*100:.2f}%" for label, prob in zip(pred_labels, pred_scores)],
            colors=cmap,
            width = 4,
        )
    else:
        pred_annotated_tensor = test_image
    
    act_masks = F2.interpolate(test_data['masks'].unsqueeze(0), size=test_image.size()[1:])

    # Annotate the test image with the target segmentation masks
    actual_annotated_tensor = draw_segmentation_masks(image=test_image, masks=test_data['masks'].type(torch.bool), alpha=0.9, 
                                                      colors=cmap)
    # Annotate the test image with the target bounding boxes
    
    acutal_annotated_tensor = draw_bounding_boxes(image=actual_annotated_tensor, 
                                                  boxes=test_data['boxes'], labels=[maskRCNN_class_names[x.item()] for x in test_data['labels']], 
                                                  colors=cmap,
                                                 width = 4)
    
    
    fig, ax = plt.subplots(1,2,figsize = (12,24))
    
    ax[0].imshow(actual_annotated_tensor.permute(1,2,0).numpy())
    ax[1].imshow(pred_annotated_tensor.permute(1,2,0).numpy())
    ax[0].set_title('actual')
    ax[1].set_title('predicted')
    plt.show()
