import gradio as gr
import cv2
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics.utils.plotting import Annotator
from ultralytics.data.augment import LetterBox

import distinctipy
import torch

from utils import *
from predict import *

hand_landmarker_file = "./mediapipe_hands/hand_landmarker.task"
class_names = maskRCNN_class_names
# Generate a list of colors with a length equal to the number of labels
colors = get_colors(4)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def segment_image(image_filepath, model_name, threshold=0.1):
    predictor = Predictor(model_name, threshold = threshold)
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = predictor.predict_masks(image)
    if results['label_idx'] is not None:
        filtered_results = predictor.get_scaled_and_filtered_masks(results, image, threshold)
        # get the predictions of image
        string_fret_region = predictor.get_string_fret_region(filtered_results['masks'].numpy(), filtered_results['label_idx'])
        if model_name.lower() == 'yolo':
            string_fret_region = transform_string_fret_region(image, string_fret_region)
        #show
        image2 = show_string_fret_regions(image, string_fret_region, colors, predictor.class_names, model_name)
    else:
        image2 = image.copy()
        string_fret_region = {}
        
    text= ""
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.4) as hands:
        image.flags.writeable = False
        hand_results = hands.process(image)
        image.flags.writeable = True

        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if hand_results.multi_handedness[hand_idx].classification[0].label == "Right":
                    # left hand
                    coords = predict_finger_location_of_left_hand(image, hand_landmarks, mp_hands, string_fret_region, predictor.class_names)

                    #draw hands
                    mp_drawing.draw_landmarks(image2, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    for finger, coord in coords.items():
                        for coord_ in coord:
                            text += f"{predictor.class_names[coord_[0]]}, {predictor.class_names[coord_[1]]} \n"
                            

    return image2, text

def segment_video(video_path, model_name, threshold=0.1):
    predictor = Predictor(model_name, threshold = threshold)
    temp_dir = './output'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    output_path = os.path.join(temp_dir, f"segmented_output_{model_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    
    predicted_coords = []
    timestamp = []

    
    with mp_hands.Hands(max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4) as hands:
        while True:
            ret, image = cap.read()
            if not ret:
                break
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = predictor.predict_masks(image) #find a way to accumulate frames
            
            if results:
                filtered_results = predictor.get_scaled_and_filtered_masks(results, image, threshold)
                # get the predictions of image
                string_fret_region = predictor.get_string_fret_region(filtered_results['masks'].numpy(), filtered_results['label_idx'])
                if model_name.lower() == 'yolo':
                    string_fret_region = transform_string_fret_region(image, string_fret_region)
                    #show
                image2 = show_string_fret_regions(image, string_fret_region, colors, predictor.class_names, model_name)
            else:
                image2 = image.copy()

            #get the finger
            image.flags.writeable = False
            hand_results = hands.process(image)
            image.flags.writeable = True

            if hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    if hand_results.multi_handedness[hand_idx].classification[0].label == "Right":
                        # left hand
                        coords = predict_finger_location_of_left_hand(image, hand_landmarks, mp_hands, string_fret_region, predictor.class_names)
    
                        #draw hands
                        mp_drawing.draw_landmarks(image2, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        text= ""

                        for string, fret in coords.items():
                            text += f"{predictor.class_names[int(string)]}, Fret {fret} \n"
                        """
                        for finger, coord in coords.items():
                            print(finger, coord)
                            for coord_ in coord:
                                if isinstance(coord_[0], int):
                                    text += f"{predictor.class_names[coord_[0]]}, {predictor.class_names[coord_[1]]} \n"
                                else:
                                    text += f"{predictor.class_names[coord_[0].item()]}, {predictor.class_names[coord_[1].item()]} \n"
                        """
                        predicted_coords.append((frame_count, text))
                        image2 = add_text_to_image(image2, text)
                        
            out.write(image2)
            
            frame_count += 1
        cap.release()
        out.release()
        return output_path

def main():
    with gr.Blocks() as demo:
        gr.Markdown("Guitar Instance Segmentation")
        
        with gr.Tab("Segment Image"):
            radio_input = gr.Radio(['YOLO','Resnet_MaskRCNN','ViT_MaskRCNN'])
            slider_input = gr.Slider(0,1,0.5,label='IOU and confidence threshold')
            with gr.Row():
                image_input = gr.Image(type="filepath")
                image_output = gr.Image()
                text_output = gr.Textbox(type='text')
            with gr.Row():
                image_btn = gr.Button("Run Segmentation")
                image_btn.click(segment_image, inputs=[image_input, radio_input, slider_input], outputs=[image_output,text_output])
                clear_output = gr.ClearButton([image_input, image_output, text_output])
                
    
        with gr.Tab("Segment Video"):
            radio_input = gr.Radio(['YOLO','Resnet_MaskRCNN','ViT_MaskRCNN'])
            slider_input = gr.Slider(0,1,0.5, label = 'IOU and confidence threshold')
            with gr.Row():
                video_input = gr.Video()
                video_output = gr.Video()
            with gr.Row():
                video_btn = gr.Button("Run Segmentation")
                video_btn.click(segment_video, inputs=[video_input,radio_input,slider_input], outputs=video_output)
                clear_output = gr.ClearButton([video_input, video_output])            
    
    demo.launch()

if __name__ == "__main__":
    main()