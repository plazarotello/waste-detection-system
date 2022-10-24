# -*- coding: utf-8 -*-

from enum import Enum

import torch
from .custom_torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fcos_resnet50_fpn, \
    retinanet_resnet50_fpn_v2, ssd300_vgg16


AVAILABLE_MODELS = Enum('Models', 'FASTERRCNN FCOS RETINANET SSD YOLO')


def get_base_model(num_classes : int, chosen_model : AVAILABLE_MODELS):
    if chosen_model == AVAILABLE_MODELS.FASTERRCNN: return get_fasterrcnn(num_classes)
    elif chosen_model == AVAILABLE_MODELS.FCOS: return get_fcos(num_classes)
    elif chosen_model == AVAILABLE_MODELS.RETINANET: return get_retinanet(num_classes)
    elif chosen_model == AVAILABLE_MODELS.YOLO: return get_yolo(num_classes)
    elif chosen_model == AVAILABLE_MODELS.SSD : return get_ssd(num_classes)
    else: return None


def get_fasterrcnn(num_classes : int):
    return fasterrcnn_resnet50_fpn_v2(num_classes=num_classes+1)


def get_fcos(num_classes : int):
    return fcos_resnet50_fpn(num_classes=num_classes+1)


def get_retinanet(num_classes : int):
    return retinanet_resnet50_fpn_v2(num_classes=num_classes+1)

def get_ssd(num_classes : int):
    return ssd300_vgg16(num_classes=num_classes+1)

def get_yolo(num_classes : int):
    return torch.hub.load('ultralytics/yolov5', 'yolov5l6', trust_repo=True, 
                        autoshape=False, classes=num_classes, verbose=False)