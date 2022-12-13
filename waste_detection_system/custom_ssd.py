from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from torchvision.models.vgg import VGG16_Weights, vgg16
from torchvision.models.detection.ssd import SSD, SSDHead, SSD300_VGG16_Weights, _vgg_extractor
from torchvision.models.detection import _utils as det_utils
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops import boxes as box_ops
from torch import Tensor, nn
import torch
import torch.nn.functional as F


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from .shared_data import AVAILABLE_CLASSIFIERS


# mostly taken from torchvision.models.detection.ssd.SSD class


def customssd300_vgg16(
    *,
    classifier : AVAILABLE_CLASSIFIERS = AVAILABLE_CLASSIFIERS.SVM,
    weights: Optional[SSD300_VGG16_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[VGG16_Weights] = VGG16_Weights.IMAGENET1K_FEATURES,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,  # type: ignore
) -> SSD:
    """The SSD300 model is based on the `SSD: Single Shot MultiBox Detector
    <https://arxiv.org/abs/1512.02325>`_ paper.

    .. betastatus:: detection module

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes but they will be resized
    to a fixed size before passing it to the backbone.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection

    Example:

        >>> model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 300), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        weights (:class:`~torchvision.models.detection.SSD300_VGG16_Weights`, optional): The pretrained
                weights to use. See
                :class:`~torchvision.models.detection.SSD300_VGG16_Weights`
                below for more details, and possible values. By default, no
                pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr
            Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.VGG16_Weights`, optional): The pretrained weights for the
            backbone
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 4.
        **kwargs: parameters passed to the ``torchvision.models.detection.SSD``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.SSD300_VGG16_Weights
        :members:
    """

    head_classifier = make_pipeline(StandardScaler(), 
        LinearSVC() if classifier==AVAILABLE_CLASSIFIERS.SVM 
                    else KNeighborsClassifier())

    weights = SSD300_VGG16_Weights.verify(weights)
    weights_backbone = VGG16_Weights.verify(weights_backbone)

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the parameter.")

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 5, 4
    )

    # Use custom backbones more appropriate for SSD
    backbone = vgg16(weights=weights_backbone, progress=progress)
    backbone = _vgg_extractor(backbone, False, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300],
    )

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    kwargs: Any = {**defaults, **kwargs}
    model = CustomSSD(backbone, anchor_generator, (300, 300), num_classes, classifier=head_classifier, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
















class CustomSSD(SSD):

    def __init__(self, backbone: nn.Module, anchor_generator: DefaultBoxGenerator,
                 size: Tuple[int, int], num_classes: int,
                 image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None,
                 head: Optional[nn.Module] = None,
                 score_thresh: float = 0.01,
                 nms_thresh: float = 0.45,
                 detections_per_img: int = 200,
                 iou_thresh: float = 0.5,
                 topk_candidates: int = 400,
                 positive_fraction: float = 0.25,
                 classifier : Union[Pipeline, None] = None):
        
        self.num_classes = num_classes
        self.classifier = classifier

        assert self.classifier is not None

        if head is None:
            if hasattr(backbone, 'out_channels'):
                out_channels = backbone.out_channels
            else:
                out_channels = det_utils.retrieve_out_channels(backbone, size)

            assert len(out_channels) == len(anchor_generator.aspect_ratios)  # type: ignore

            num_anchors = anchor_generator.num_anchors_per_location()
            head = CustomSSDHead(out_channels, num_anchors, num_classes)  # type: ignore
        super().__init__(backbone=backbone, anchor_generator=anchor_generator, size=size,
                            num_classes=num_classes, image_mean=image_mean, image_std=image_std,
                            head=head, score_thresh=score_thresh, nms_thresh=nms_thresh, 
                            detections_per_img=detections_per_img, iou_thresh=iou_thresh, 
                            topk_candidates=topk_candidates, positive_fraction=positive_fraction)


    def compute_loss(self, targets: List[Dict[str, Tensor]], 
        head_outputs: Dict[str, Union[List[Tensor], Tensor]], anchors: List[Tensor], 
        matched_idxs: List[Tensor]) -> Dict[str, Tensor]:

        bbox_regression = head_outputs['bbox_regression']
        features = head_outputs['features']

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (targets_per_image, bbox_regression_per_image, features_per_image, anchors_per_image,
                matched_idxs_per_image) in zip(targets, bbox_regression, features, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image['boxes'][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = super().box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(torch.nn.functional.smooth_l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ))

            feature_list = [x.item() for x in features_per_image]
            targets_list = torch.zeros((bbox_regression_per_image.size(0), ), dtype=targets_per_image['labels'].dtype,
                                            device=targets_per_image['labels'].device)
            self.classifier = self.classifier.fit(feature_list, targets_list.detach().cpu().tolist())   # type: ignore

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros((bbox_regression_per_image.size(0), ), dtype=targets_per_image['labels'].dtype,
                                            device=targets_per_image['labels'].device)
            gt_classes_target[foreground_idxs_per_image] = \
                targets_per_image['labels'][foreground_matched_idxs_per_image]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)
        cls_logits = torch.stack([self.classifier.predict_proba([x.item() for x in   # type: ignore
                        features_per_image]) for features_per_image in features])

        # Calculate classification loss
        num_classes = self.num_classes
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes),
            cls_targets.view(-1),
            reduction='none'
        ).view(cls_targets.size())

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = super().neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float('inf')  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            'bbox_regression': bbox_loss.sum() / N,
            'classification': (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }
    

    def postprocess_detections(self, head_outputs: Dict[str, Union[Tensor, List[Tensor]]], image_anchors: List[Tensor],
                               image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs['bbox_regression']
        cls_logits = torch.stack([self.classifier.predict_proba([x.item() for x in   # type: ignore
                        features_per_image]) for features_per_image in head_outputs['features']])
        pred_scores = F.softmax(cls_logits, dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = super().box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > super().score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(super().topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, super().nms_thresh)
            keep = keep[:super().detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })
        return detections


class CustomSSDHead(SSDHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        super().__init__(in_channels, num_anchors, num_classes)
        del self.classification_head

    def forward(self, x: List[Tensor], ) -> Dict[str, Union[List[Tensor], Tensor]]:
        return {
            "bbox_regression": self.regression_head(x), # (N, HWA, K)
            # "cls_logits": self.classification_head(x),  # (N, HWA, K)
            "features": x,  # (N, features)
        }