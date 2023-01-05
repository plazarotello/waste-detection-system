# -*- coding: utf-8 -*-

"""
.. _fasterrcnn: https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
.. _fcos: https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html#torchvision.models.detection.fcos_resnet50_fpn
.. _retinanet: https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html#torchvision.models.detection.retinanet_resnet50_fpn_v2
.. _ssd: https://pytorch.org/vision/main/models/generated/torchvision.models.detection.ssd300_vgg16.html#torchvision.models.detection.ssd300_vgg16
.. _knn: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
.. _svm: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

Collection of allowed object detection models in the Waste Detection System, 
as well as functions to freeze and unfreeze layers given a Transfer Learning Level,
a model creation factory for object detection models and hybrid Deep Learning 
models.

Deep Learning object detection models
    - Faster R-CNN : `Faster R-CNN Pytorch documentation <fasterrcnn_>`_
    - FCOS : `FCOS Pytorch documentation <fcos_>`_
    - RetinaNet : `RetinaNet Pytorch documentation <retinanet_>`_
    - SSD : `SSD Pytorch documentation <ssd_>`_

Hybrid Deep Learning/traditional Machine Learning models
    It takes a Deep Learning object detection model (of the mentioned above) and
    a traditional Machine Learning classifier of the following:
        - k-Nearest Neighbors Classifier: `scikit-learn documentation <knn_>`_
        - SVM Classifier: `scikit-learn documentation <svm_>`_

.. important::
    To train a hybrid Deep Learning model, it is necessary to fully train the object 
    detector and only then, with the best weights saved, use it to create the feature
    extractor of the hybrid model.
"""

from typing import Any, Union
from itertools import chain
from collections import OrderedDict

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights, FCOS
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights, RetinaNet
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.detection.anchor_utils import AnchorGenerator, DefaultBoxGenerator
from torchvision.ops.poolers import MultiScaleRoIAlign
from torch import Tensor, nn

from torchinfo import summary

from waste_detection_system.shared_data import AVAILABLE_CLASSIFIERS, AVAILABLE_MODELS
from waste_detection_system.feature_extractor import FeatureExtractor, HybridDLModel


def load_partial_weights(model: Union[FasterRCNN, FCOS, SSD, RetinaNet], weights : Any) -> Union[FasterRCNN, FCOS, SSD, RetinaNet]:
    """Load the given weights in the model. If a layer is not present in the weights 
    and model, the weights regarding that layer won't be loaded.

    Args:
        model (Union[FasterRCNN, FCOS, SSD, RetinaNet]): model to load the weights in
        weights (Any): weights to load

    Returns:
        Union[FasterRCNN, FCOS, SSD, RetinaNet]: model with the weights loaded
    """
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model


def print_stats(model: Union[FasterRCNN, FCOS, SSD, RetinaNet], transfer_learning_level: int):
    """Prints the trainable/total parameters of the model

    Args:
        model (Union[FasterRCNN, FCOS, SSD, RetinaNet]): model
        transfer_learning_level (int): Transfer Learning Level. For more information, see :ref:`here <tll>`
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'TLL = {transfer_learning_level} | {trainable_params} trainable params '
        f'({round((trainable_params/total_params)*100, 2)}%) | '
        f'{total_params} total params')


def pretty_summary(model: Union[FasterRCNN, FCOS, SSD, RetinaNet]):
    """Prints the summary of the model, showing 3 levels of depth and the trainable/total parameters

    Args:
        model (Union[FasterRCNN, FCOS, SSD, RetinaNet]): model
    """
    print(summary(model, depth=3, 
        col_names=["num_params", "trainable", "input_size", "output_size"], 
        input_size=[1,300,300], batch_dim=0))



class MLPBackbone(nn.Module):
    """
    Lightweighted backbone for use as a feature extractor in an object detector such as
    SSD and Faster R-CNN.

    Consists on a Convolutional layer, a batch normalization, a ReLU and a max pool layer
    """
    def __init__(self):
        super().__init__()
        self.inplanes = 256
        self.out_channels = self.inplanes

        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(self.inplanes)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
    
    def forward(self, x : Tensor) -> Tensor:
        return self.backbone(x)



def get_hybrid_model(num_classes : int, chosen_model : AVAILABLE_MODELS, weights : Any,
    chosen_classifier : AVAILABLE_CLASSIFIERS) -> HybridDLModel:
    """Constructs a hybrid DL+traditional ML model

    Args:
        num_classes (int): number of classes
        chosen_model (AVAILABLE_MODELS): base object detection model
        weights (Any): weights of the base detection model
        chosen_classifier (AVAILABLE_CLASSIFIERS): traditional ML classifier head

    Returns:
        HybridDLModel: constructed model
    """
    model = get_base_model(num_classes=num_classes, chosen_model=chosen_model,
                            transfer_learning_level=0)
    model = load_partial_weights(model, weights)
    feature_extractor = to_feature_extractor(model)
    return HybridDLModel(feature_extractor=feature_extractor, classifier_type=chosen_classifier)




def get_base_model(num_classes : int, chosen_model : AVAILABLE_MODELS, 
                    transfer_learning_level : int) -> Union[FasterRCNN, FCOS, SSD, RetinaNet]:
    """Constructs the given model

    Args:
        num_classes (int): number of classes
        chosen_model (AVAILABLE_MODELS): selected model
        transfer_learning_level (int): Transfer Learning Level. For more information, see :ref:`here <tll>`

    Returns:
        Union[FasterRCNN, FCOS, SSD, RetinaNet]: constructed model
    
    Raises:
        ValueError: if ``chosen_model`` is not amongst the available ones
    """
    if chosen_model == AVAILABLE_MODELS.FASTERRCNN: return get_fasterrcnn(num_classes, 
                                                                transfer_learning_level)
    elif chosen_model == AVAILABLE_MODELS.FCOS: return get_fcos(num_classes, 
                                                                transfer_learning_level)
    elif chosen_model == AVAILABLE_MODELS.RETINANET: return get_retinanet(num_classes, 
                                                                transfer_learning_level)
    elif chosen_model == AVAILABLE_MODELS.SSD : return get_ssd(num_classes, 
                                                                transfer_learning_level)
    elif chosen_model == AVAILABLE_MODELS.MLP_FRCNN: return get_mlp_fasterrcnn(num_classes)
    elif chosen_model == AVAILABLE_MODELS.MLP_SSD: return get_mlp_ssd(num_classes)
    else: raise ValueError('Model not supported')



def to_feature_extractor(model: Union[FasterRCNN, FCOS, RetinaNet, SSD]) -> FeatureExtractor:
    """Wraps the given model in order to obtain the bounding boxes and feature maps associated

    Args:
        model (Union[FasterRCNN, FCOS, RetinaNet, SSD]): model to use as feature extractor

    Raises:
        ValueError: if the wrapper is not implemented for the given model type

    Returns:
        FeatureExtractor: wrapper that returns a dictionary of 'bounding_boxes' and 'features',
        both of them being lists of tensors
    :meta private:
    """
    if type(model) is SSD:
        return FeatureExtractor(model, layer='head.regression_head')
    if type(model) is FasterRCNN:
        return FeatureExtractor(model, layer='roi_heads.box_head')
    raise ValueError('Feature extractor not implemented for this model type')






def apply_tll_to_fasterrcnn(model: FasterRCNN, transfer_learning_level : int) -> FasterRCNN:
    """Applies Transfer Learning Level to the given model, freezing and unfreezing the corresponding layers

    Args:
        model (FasterRCNN): model
        transfer_learning_level (int): Transfer Learning Level. For more information, see :ref:`here <tll>`

    Returns:
        FasterRCNN: model with the correct number of layers frozen/unfrozen
    
    :meta private:
    """
    # TLL=0 : train from scratch
    if transfer_learning_level == 0:
        for param in model.parameters():
            param.requires_grad = True
    # TLL=1 : transfer learning, train only the head
    elif transfer_learning_level == 1:
        for param in chain(model.transform.parameters(),
                           model.backbone.parameters(),
                           model.rpn.parameters()):  # type: ignore
            param.requires_grad = False
        for param in model.roi_heads.parameters():  # type: ignore
            param.requires_grad = True
    # TLL>1 : fine-tuning, trains the head plus the (n-1) number of layers from top to bottom
    elif transfer_learning_level >= 2 and transfer_learning_level <= 5:
        for param in model.transform.parameters():
            param.requires_grad = False
        
        if transfer_learning_level == 2:
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif transfer_learning_level == 3:
            for param in model.backbone.body.parameters():  # type: ignore
                param.requires_grad = False
        elif transfer_learning_level >= 4:
            for child in list(model.backbone.body.children())[:7]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = False
            for child in list(model.backbone.body.children())[7:]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = True
        
        if transfer_learning_level >= 3:
            for param in model.backbone.fpn.parameters():  # type: ignore
                param.requires_grad = True
        
        for param in model.rpn.parameters():
            param.requires_grad = True
        for param in model.roi_heads.parameters():
            param.requires_grad = True
    print_stats(model, transfer_learning_level)
    return model


def apply_tll_to_fcos_retinanet(model: Union[FCOS, RetinaNet], transfer_learning_level : int) -> Union[FCOS, RetinaNet]:
    """Applies Transfer Learning Level to the given model, freezing and unfreezing the corresponding layers

    Args:
        model (Union[FCOS, RetinaNet]): model
        transfer_learning_level (int): Transfer Learning Level. For more information, see :ref:`here <tll>`

    Returns:
        Union[FCOS, RetinaNet]: model with the correct number of layers frozen/unfrozen
    
    :meta private:
    """
    # TLL=0 : train from scratch
    if transfer_learning_level == 0:
        for param in model.parameters():
            param.requires_grad = True
    # TLL=1 : transfer learning, train only the head
    elif transfer_learning_level == 1:
        for param in chain(model.transform.parameters(),
                           model.backbone.parameters(),
                           model.anchor_generator.parameters(),
                           model.head.regression_head.parameters()):  # type: ignore
            param.requires_grad = False
        for param in model.head.classification_head.parameters():  # type: ignore
            param.requires_grad = True
    # TLL>1 : fine-tuning, trains the head plus the (n-1) number of layers from top to bottom
    elif transfer_learning_level >= 2 and transfer_learning_level <= 5:
        for param in model.transform.parameters():
            param.requires_grad = False
        
        if transfer_learning_level == 2:
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif transfer_learning_level == 3:
            for param in model.backbone.body.parameters():  # type: ignore
                param.requires_grad = False
        elif transfer_learning_level == 4:
            for child in list(model.backbone.body.children())[:7]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = False
            for child in list(model.backbone.body.children())[7:]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = True
        elif transfer_learning_level == 5:
            for child in list(model.backbone.body.children())[:6]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = False
            for child in list(model.backbone.body.children())[6:]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = True
        
        if transfer_learning_level >= 3:
            for param in model.backbone.fpn.parameters():  # type: ignore
                param.requires_grad = True
        
        for param in model.anchor_generator.parameters():
            param.requires_grad = True
        for param in model.head.parameters():
            param.requires_grad = True
    print_stats(model, transfer_learning_level)
    return model


def apply_tll_to_ssd(model: SSD, transfer_learning_level : int) -> SSD:
    """Applies Transfer Learning Level to the given model, freezing and unfreezing the corresponding layers

    Args:
        model (SSD): model
        transfer_learning_level (int): Transfer Learning Level. For more information, see :ref:`here <tll>`

    Returns:
        SSD: model with the correct number of layers frozen/unfrozen
        
    :meta private:
    """
    # TLL=0 : train from scratch
    if transfer_learning_level == 0:
        for param in model.parameters():
            param.requires_grad = True
    # TLL=1 : transfer learning, train only the head
    elif transfer_learning_level == 1:
        for param in chain(model.transform.parameters(),
                           model.backbone.parameters(),
                           model.anchor_generator.parameters(),
                           model.head.regression_head.parameters()):  # type: ignore
            param.requires_grad = False
        if hasattr(model.head, "classification_head"):
            for param in model.head.classification_head.parameters():  # type: ignore
                param.requires_grad = True
    # TLL>1 : fine-tuning, trains the head plus the (n-1) number of layers from top to bottom
    elif transfer_learning_level >= 2 and transfer_learning_level <= 5:
        for param in model.transform.parameters():
            param.requires_grad = False
        
        if transfer_learning_level == 2:
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif transfer_learning_level == 3:
            for param in model.backbone.features.parameters():  # type: ignore
                param.requires_grad = False
            for child in list(model.backbone.extra.children())[:2]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = False
            for child in list(model.backbone.extra.children())[2:]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = True
        elif transfer_learning_level == 4:
            for param in model.backbone.features.parameters():  # type: ignore
                param.requires_grad = False
            for child in list(model.backbone.extra.children())[:1]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = False
            for child in list(model.backbone.extra.children())[1:]:  # type: ignore
                for param in child.parameters():
                    param.requires_grad = True
        elif transfer_learning_level == 5:
            for param in model.backbone.features.parameters():  # type: ignore
                param.requires_grad = False
            for param in model.backbone.extra.parameters():  # type: ignore
                param.requires_grad = True
        
        for param in model.anchor_generator.parameters():
            param.requires_grad = True
        for param in model.head.parameters():
            param.requires_grad = True
    print_stats(model, transfer_learning_level)
    return model


def get_fasterrcnn(num_classes : int, transfer_learning_level : int) -> FasterRCNN:
    """Constructs a ``FasterRCNN`` model

    Args:
        num_classes (int): number of classes
        transfer_learning_level (int): Transfer Learning Level. For more information, see :ref:`here <tll>`

    Returns:
        FasterRCNN: constructed model
        
    :meta private:
    """
    FasterRCNN.model_num_classes = 0  # type: ignore
    model = load_partial_weights(
            model=fasterrcnn_resnet50_fpn_v2(num_classes=num_classes+1),
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.get_state_dict(progress=True)
    )
    model.model_num_classes = num_classes+1  # type: ignore
    return apply_tll_to_fasterrcnn(model, transfer_learning_level)  # type: ignore


def get_fcos(num_classes : int, transfer_learning_level : int) -> FCOS:
    """Constructs a ``FCOS`` model

    Args:
        num_classes (int): number of classes
        transfer_learning_level (int): Transfer Learning Level. For more information, see :ref:`here <tll>`

    Returns:
        FCOS: constructed model
        
    :meta private:
    """
    FCOS.model_num_classes = 0  # type: ignore
    model = load_partial_weights(
            model=fcos_resnet50_fpn(num_classes=num_classes+1),
            weights=FCOS_ResNet50_FPN_Weights.DEFAULT.get_state_dict(progress=True)
    )
    model.model_num_classes = num_classes+1  # type: ignore
    return apply_tll_to_fcos_retinanet(model, transfer_learning_level)  # type: ignore


def get_retinanet(num_classes : int, transfer_learning_level : int) -> RetinaNet:
    """Constructs a ``RetinaNet`` model

    Args:
        num_classes (int): number of classes
        transfer_learning_level (int): Transfer Learning Level. For more information, see :ref:`here <tll>`

    Returns:
        RetinaNet: constructed model
        
    :meta private:
    """
    RetinaNet.model_num_classes = 0  # type: ignore
    model = load_partial_weights(
            model=retinanet_resnet50_fpn_v2(num_classes=num_classes+1),
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.get_state_dict(progress=True)
    )
    model.model_num_classes = num_classes+1  # type: ignore
    return apply_tll_to_fcos_retinanet(model=model, transfer_learning_level=transfer_learning_level)  # type: ignore
    

def get_ssd(num_classes : int, transfer_learning_level : int) -> SSD:
    """Constructs a ``SSD`` model

    Args:
        num_classes (int): number of classes
        transfer_learning_level (int): Transfer Learning Level. For more information, see :ref:`here <tll>`
        
    Returns:
        SSD: constructed model
        
    :meta private:
    """
    SSD.model_num_classes = 0  # type: ignore
    model = load_partial_weights(
            model=ssd300_vgg16(num_classes=num_classes+1),
            weights=SSD300_VGG16_Weights.DEFAULT.get_state_dict(progress=True)
    )
    model.model_num_classes = num_classes+1  # type: ignore
    return apply_tll_to_ssd(model, transfer_learning_level) # type: ignore


def get_mlp_fasterrcnn(num_classes : int) -> FasterRCNN:
    """Constructs a ``MLP_FRCNN`` model, a.k.a. a Faster R-CNN detector with a custom
    lightweight backbone.

    Args:
        num_classes (int): number of classes
        
    Returns:
        FasterRCNN: constructed model
        
    :meta private:
    """
    FasterRCNN.model_num_classes = 0  # type: ignore
    backbone = MLPBackbone()

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone, 
        num_classes=num_classes+1,
        rpn_anchor_generator=anchor_generator, 
        box_roi_pool=roi_pooler,
        rpn_fg_iou_thresh=0.25,
        rpn_bg_iou_thresh=0.2,
        box_fg_iou_thresh=0.25,
        box_bg_iou_thresh=0.2
    )
    model = load_partial_weights(
            model=model,
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.get_state_dict(progress=True)
    )
    model.model_num_classes = num_classes+1  # type: ignore

    return model # type: ignore


def get_mlp_ssd(num_classes : int) -> SSD:
    """Constructs a ``MLP_SSD`` model, a.k.a. a SSD detector with a custom
    lightweight backbone.

    Args:
        num_classes (int): number of classes
        
    Returns:
        SSD: constructed model
        
    :meta private:
    """
    SSD.model_num_classes = 0  # type: ignore
    backbone = MLPBackbone()

    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300],
    )
    num_anchors = anchor_generator.num_anchors_per_location()

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    
    kwargs: Any = {**defaults}
    head = SSDHead([backbone.out_channels], num_anchors, num_classes+1)
    model = SSD(backbone, anchor_generator, (300, 300), num_classes+1, head=head,
                iou_thresh=0.25, **kwargs)

    model = load_partial_weights(
            model=model,
            weights=SSD300_VGG16_Weights.DEFAULT.get_state_dict(progress=True)
    )
    model.model_num_classes = num_classes+1  # type: ignore

    return model # type: ignore