import torch.nn as nn
from torchvision.transforms import v2
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import vit_b_16
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.v2 import functional as F
import torch.nn.functional as F2

from torchvision.ops.feature_pyramid_network import (
    LastLevelMaxPool,
    FeaturePyramidNetwork
)

import collections
import numpy as np
from utils import maskRCNN_class_names

NUM_LAYER = 12
PATCH_SIZE = 16
HIDDEN_DIM = 768
NUM_CLASSES = len(maskRCNN_class_names)
IMG_SIZE = 224

class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers",
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = collections.OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

        self.C = HIDDEN_DIM
        self.H = self.W = IMG_SIZE // PATCH_SIZE

    def forward(self, x):
        out = collections.OrderedDict()
        idx = 0
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                N = x.shape[0]
                out[out_name] = F2.interpolate(
                    F2.instance_norm(
                        x.permute(0, 2, 1).reshape(N, self.C, self.H, self.W)
                    ),
                    scale_factor=4 / (2**idx),
                    mode="bilinear",
                )
                idx += 1
        return out
        
class BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=None,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.backbone = backbone

        self.body = IntermediateLayerGetter(
            self.backbone.encoder.layers,
            return_layers=return_layers,
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.backbone._process_input(x)
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)
        x = self.body(x)
        x = self.fpn(x)
        return x

class ViT_MaskedRCNN(nn.Module):
    def __init__(self, num_classes):
        super(ViT_MaskedRCNN, self).__init__()
        self.mask_rcnn = maskrcnn_resnet50_fpn_v2(weights = 'DEFAULT')
        self.backbone = vit_b_16(weights = 'DEFAULT')
        self.num_classes = num_classes
        
        for child in self.mask_rcnn.children():
            for param in child.parameters():
                param.requires_grad = False

        for child in self.backbone.children():
            for param in child.parameters():
                param.requires_grad = False


        # replace the backbone
        self.backbone.encoder.pos_embedding = nn.Parameter(
            self.backbone.encoder.pos_embedding[:, 1:, :]
        )
        del self.backbone.class_token, self.backbone.encoder.ln
        self.backbone = BackboneWithFPN(
            self.backbone,
            {
                f"encoder_layer_{(NUM_LAYER - 1) - l}": str(
                    ((NUM_LAYER - 1) - l - 2) // 3
                )
                for l in range(9, -1, -3)
            },
            [HIDDEN_DIM] * 4,
            256,
        )

        
        self.mask_rcnn.backbone = self.backbone
        self.mask_rcnn.transform.min_size = [IMG_SIZE]
        self.mask_rcnn.transform.max_size = IMG_SIZE
        self.mask_rcnn.transform.fixed_size = (IMG_SIZE, IMG_SIZE)
        self.mask_rcnn.num_classes = NUM_CLASSES
        self.mask_rcnn.roi_heads.box_predictor.cls_score = nn.Linear(
            1024,
            NUM_CLASSES,
        )
        self.mask_rcnn.roi_heads.box_predictor.bbox_pred = nn.Linear(
            1024,
            4 * NUM_CLASSES,
        )
        self.mask_rcnn.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(
            256, NUM_CLASSES, 1,
        )

        
        # replace the box predictor classifier
        # in_feats = self.masked_rcnn.roi_heads.box_predictor.cls_score.in_features
        # self.masked_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_feats, self.num_classes)
        

        # # replace the mask predictor classifier
        # in_features_mask = self.masked_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        # mask_predictor_hidden_layer_size = 256
        # self.masked_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(
        # in_features_mask,
        # mask_predictor_hidden_layer_size,
        # self.num_classes)

    def forward(self, X, y):
        # X = (batch size, n_channels, width, height)
        # y = (batch size, n_classes, width, height). All 1s and 0s
        
        return self.mask_rcnn(X, y)

class MaskedRCNN(nn.Module):
    def __init__(self, num_classes):
        super(MaskedRCNN, self).__init__()
        self.masked_rcnn = maskrcnn_resnet50_fpn_v2(weights = 'DEFAULT')
        self.num_classes = num_classes
        
        for child in self.masked_rcnn.children():
            for param in child.parameters():
                param.requires_grad = False
        
        
        # replace the box predictor classifier
        in_feats = self.masked_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.masked_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_feats, self.num_classes)
        

        # replace the mask predictor classifier
        in_features_mask = self.masked_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        mask_predictor_hidden_layer_size = 512
        self.masked_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        mask_predictor_hidden_layer_size,
        self.num_classes)

    def forward(self, X, y):
        # X = (batch size, n_channels, width, height)
        # y = (batch size, n_classes, width, height). All 1s and 0s
        
        return self.masked_rcnn(X, y)

