import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import random
import argparse

from dataset import GuitarDataset
from utils import move_targets_to_device, maskRCNN_class_names, plot_results
from model import ViT_MaskedRCNN, MaskedRCNN
from ultralytics import YOLO
from torch.amp import autocast
from predict import Predictor
from evaluate import Evaluator

def train_yolo(training_yaml_filepath, test_yaml_filepath ,imgsz = 640, epochs = 200, class_label_list = [1,2,3,4,5,6,7,8,9,10,11,12], batch_sz = 8, device = 'cuda'):
    model = YOLO("yolo11n-seg.pt")
    model.train(data = training_yaml_filepath, epochs = epochs, imgsz = imgsz, device = device, classes = class_label_list, batch = batch_sz)
    model.val(data = test_yaml_filepath, imgsz=imgsz, batch=1, conf=0.5, iou=0.5, device= device, classes = class_label_list)
    
    return model, model.ckpt_path

def train(model_name, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, n_epochs, scaler, batch_size, save_weights_dir, device = 'cuda:0'):
    if not torch.cuda.is_available():
        device = 'cpu'

        

    model.to(device)
    epoch_loss = []
    val_losses = []


    for i in range(n_epochs):
        model.train()
        training_loss = []
        count_train = 0
        count_val = 0
        for j, (batch, targets) in enumerate(train_dataloader):
            with autocast(torch.device(device).type):
                batch = torch.stack(batch).to(device)
                move_targets_to_device(targets, device)
    
                losses = model(batch, targets)
                sum_losses = sum([loss for loss in losses.values()])

            if scaler:
                scaler.scale(sum_losses).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                sum_losses.backward()
                optimizer.step()
            optimizer.zero_grad()
            
            training_loss.append(sum_losses.item())
            count_train += len(batch)
            print(f'epoch {i+1} run {j+1}/{len(train_dataloader)}: training loss - {sum_losses.item()/len(batch)}')
        epoch_loss.append(np.mean(training_loss))
        lr_scheduler.step()
        
        val_loss = []
        with torch.no_grad():
            for k, (batch, targets) in enumerate(val_dataloader):
                with autocast(torch.device(device).type):
                    batch = torch.stack(batch).to(device)
                    move_targets_to_device(targets, device)
                    vloss = model(batch, targets)
                    val_loss_ = sum([loss for loss in vloss.values()])
                val_loss.append(val_loss_.item())
                count_val += len(batch)
                
                print(f'epoch {i+1} run {k+1}/{len(val_dataloader)}: validation loss - {val_loss_.item()/len(batch)}')

        val_losses.append(np.mean(val_loss))
        print(f'epoch {i+1}: training loss - {np.sum(training_loss)/count_train} | validation loss - {np.sum(val_loss)/count_val}')
        
        if (i+1) % 5 == 0:
            path_to_weights = f'./{save_weights_dir}/model_{i+1}.pt'
            torch.save(model.state_dict(), path_to_weights)

        if n_epochs < 5:
            path_to_weights = f'./{save_weights_dir}/model_{i+1}.pt'
            torch.save(model.state_dict(), path_to_weights)
    
    fig, ax = plt.subplots()
    ax.plot(epoch_loss, label = 'training loss')
    ax.plot(val_losses, label = 'validation loss')
    plt.legend()
    plt.show()

    return model, path_to_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, choices = ['yolo', 'vit_maskrcnn', 'resnet_maskrcnn'], help = 'model name')
    parser.add_argument('--dataset_directory', type = str, default = '../datasets/guitar_dataset_v2', help='directory containing folders named train, test, and valid. Each folder has subfolders named images and labels, and each label is a text file with the same base filename as the image base filename. If the image has no corresponding label text file, it is considered a background class')
    parser.add_argument('--lr', default = 0.01, type = float, help='learning rate for maskrcnn models')
    parser.add_argument('--batch_size', default = 8, type = int, help = 'batch size')
    parser.add_argument('--device', default = 'cuda',type = str, help = 'gpu or cpu')
    parser.add_argument('--epochs', default = 50, type = int, help='number of training and validation epochs')
    parser.add_argument('--yolo_training_yaml_filepath', type=str, default = "training_yaml/yolo11_seg_v2.yaml", help = 'filepath with yolo training yaml config. Only required if model_name is yolo')
    parser.add_argument('--yolo_testing_yaml_filepath', type = str,default = "training_yaml/yolo11_seg_test.yaml", help = 'filepath with yolo testing yaml config. Only required if model_name is yolo')
    parser.add_argument('--yolo_class_label_list', nargs = '+', type=int, default = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19], help = 'selected class indices for training. Only required if model_name is yolo')
    
    args = parser.parse_args()
    
        
    if args.model_name.lower() == 'yolo':
        if len(args.yolo_class_label_list) == 0:
            raise ValueError('Length of label list should be more than 0')
        if args.yolo_training_yaml_filepath is None or args.yolo_testing_yaml_filepath is None:
            raise ValueError('No filepath to yaml file provided')
            
        model, path_to_weights = train_yolo(training_yaml_filepath = args.yolo_training_yaml_filepath, test_yaml_filepath = args.yolo_testing_yaml_filepath, imgsz = 640, epochs = args.epochs, class_label_list = args.yolo_class_label_list, batch_sz = args.batch_size, device = args.device)

    else:
        if args.model_name.lower() == 'vit_maskrcnn':
            model = ViT_MaskedRCNN(len(maskRCNN_class_names))
        elif args.model_name.lower() == 'resnet_maskrcnn':
            model = MaskedRCNN(len(maskRCNN_class_names))

            
        if args.model_name.lower() == 'resnet_maskrcnn':
            dir_name = 'resnet_maskrcnn'
        elif args.model_name.lower() == 'vit_maskrcnn':
            dir_name = 'vit_maskrcnn'
        else:
            raise ValueError(f'accepted values of model name are resnet_maskrcnn or vit_maskrcnn. Got {args.model_name} instead')
        
        train_dataset = GuitarDataset(f'{args.dataset_directory}/train', len(maskRCNN_class_names))
        val_dataset = GuitarDataset(f'{args.dataset_directory}/valid', len(maskRCNN_class_names))
        test_dataset = GuitarDataset(f'{args.dataset_directory}/test', len(maskRCNN_class_names))

        batch_size = args.batch_size
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn = lambda x: tuple(zip(*x)))
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, collate_fn = lambda x: tuple(zip(*x)))
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn = lambda x: tuple(zip(*x)))

        epochs = args.epochs
        scaler = torch.amp.GradScaler()

        now = datetime.datetime.now()
        datetime_string = now.strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.exists(f'./{dir_name}'):
            os.mkdir(f'./{dir_name}')
        if not os.path.exists(f'./{dir_name}/{datetime_string}'):
            os.mkdir(f'./{dir_name}/{datetime_string}')

        if args.model_name.lower() == 'vit_maskrcnn':
            if len(train_dataset) % epochs == 0:
                num_batches = epochs * len(train_dataloader)
            else:
                num_batches = epochs * (len(train_dataloader) + 1)

            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = args.lr)
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                            max_lr=args.lr, 
                                                            total_steps=num_batches)

        elif args.model_name.lower() == 'resnet_maskrcnn':
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(
                params,
                lr=args.lr,
                weight_decay=0.0005
            )

            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=3,
                gamma=1
            )

    
        model, path_to_weights = train(args.model_name, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, epochs, scaler, batch_size = args.batch_size, device = args.device, save_weights_dir = f'./{dir_name}/{datetime_string}')
        print(f'Model weights saved to {path_to_weights}')
        evaluator = Evaluator(model)
        evaluator.evaluate(test_dataloader)
        results = evaluator.results

        with open(f'./{dir_name}/{datetime_string}/test_results.txt', 'w') as f:
            f.write(f'{len(test_dataloader.dataset)} items\n')
            for key, val in results.items():
                f.write(f'{key}: {val:.3f}\n')

        pred_results = evaluator.pred_results
        chosen = random.sample(list(range(len(pred_results))), k = min(5, len(pred_results)))
        
        for i in chosen:
            test_image, test_data = test_dataloader.dataset[i]
            # Annotate the test image with the target segmentation masks
            plot_results(test_image, test_data, pred_results[i])
            