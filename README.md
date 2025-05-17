****README****

This project is part of NUS Intelligent Systems Graduate Certificate Programme and is on guitar parts segmentation for guitar tabs transcription.

To install dependencies, ensure that Python 3.10.4 is installed and run:

`pip install -r requirements.txt`

*Pretrained weights*
As MediaPipe Hands is used in this software, please download the weights file `hand_landmarker.task` and store it in the folder `mediapipe_hands`.

Weights for the YOLOv11n, ResNet50-MaskRCNN, and ViT-MaskRCNN are included in your zip file. Otherwise, please request it from me at `zhenhao1886@gmail.com`. YOLOv11n weights are stored in the directory `code/runs/train11` and `code/runs/train12`. The other two are stored in `code/resnet_maskrcnn` and `code/vit_maskrcnn` respectively.

*Demo*
Next, to run the demo, change directory to `/code` and run `python app.py`. 

If you are transcribing with a video format, The output video will be stored in the `output` folder.

*Training*
Classes have to be numbered from 1 to 12 (frets) and 14 to 19 (strings). Class 0 and 13 are hand and fretboard respectively. The YOLOv11n yaml files are in `code/training_yaml`. When training, make sure 2 models are trained, one fore frets and one for string classes. The weights will be saved in the aforementioned weight locations.

To train the model, cd to `code` and run `python train.py --model_name [model name]`. Additional parameters can be accesed via the help menu.

*Done by: Zhen Hao*
