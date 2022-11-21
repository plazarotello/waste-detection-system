# -*- coding: utf-8 -*-

import shared_data as base

from enum import Enum
from torch import load

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

from custom_ultralytics_yolov5 import hubconf


AVAILABLE_MODELS = Enum('Models', 'FASTERRCNN FCOS RETINANET SSD YOLO')


def load_partial_weights(model, weights):
    # load partial weights for warmstarting
    #model.load_state_dict(weights, strict=False)
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model


# TRANSFER LEARNING LEVEL:
# TTL=0 : train from scratch
# TTL=1 : transfer learning, train only the head
# TTL>1 : fine-tuning, trains the head plus the (n-1) number of layers from top to bottom

def get_base_model(num_classes : int, chosen_model : AVAILABLE_MODELS, 
                    transfer_learning_level : int):
    if chosen_model == AVAILABLE_MODELS.FASTERRCNN: return get_fasterrcnn(num_classes, 
                                                                transfer_learning_level)
    elif chosen_model == AVAILABLE_MODELS.FCOS: return get_fcos(num_classes, 
                                                                transfer_learning_level)
    elif chosen_model == AVAILABLE_MODELS.RETINANET: return get_retinanet(num_classes, 
                                                                transfer_learning_level)
    elif chosen_model == AVAILABLE_MODELS.YOLO: return get_yolo(num_classes, 
                                                                transfer_learning_level)
    elif chosen_model == AVAILABLE_MODELS.SSD : return get_ssd(num_classes, 
                                                                transfer_learning_level)
    else: return None


def get_fasterrcnn(num_classes : int, transfer_learning_level : int):
    model = load_partial_weights(
            model=fasterrcnn_resnet50_fpn_v2(num_classes=num_classes+1),
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.get_state_dict(progress=True)
    )
    print([n for n, _ in model.named_children()])
    # if transfer_learning_level == 0:

def get_fcos(num_classes : int, transfer_learning_level : int):
    model = load_partial_weights(
            model=fcos_resnet50_fpn(num_classes=num_classes+1),
            weights=FCOS_ResNet50_FPN_Weights.DEFAULT.get_state_dict(progress=True)
    )
    print([n for n, _ in model.named_children()])


def get_retinanet(num_classes : int, transfer_learning_level : int):
    model = load_partial_weights(
            model=retinanet_resnet50_fpn_v2(num_classes=num_classes+1),
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.get_state_dict(progress=True)
    )
    print([n for n, _ in model.named_children()])
    


def get_ssd(num_classes : int, transfer_learning_level : int):
    model = load_partial_weights(
            model=ssd300_vgg16(num_classes=num_classes+1),
            weights=SSD300_VGG16_Weights.DEFAULT.get_state_dict(progress=True)
    )
    print([n for n, _ in model.named_children()])


def get_yolo(num_classes : int, transfer_learning_level : int):
    model = load_partial_weights(
            model=hubconf.yolov5l6(classes=num_classes, autoshape=False,
                                _verbose=False),
            weights=load(base.MODELS_DIR / 'yolov5l6.pt')
    )
    print([n for n, _ in model.named_children()])


get_yolo(2, 0)