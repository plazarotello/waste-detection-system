from itertools import chain
from typing import Union
import lightning as pl
import torch
from torchvision.models.detection import  FasterRCNN, FCOS, RetinaNet
from torchvision.models.detection.ssd import SSD

from .utils import from_dict_to_boundingbox
from .enumerators import MethodAveragePrecision
from .pascal_voc_evaluator import get_pascalvoc_metrics
from . import shared_data as base

class ModelModule(pl.LightningModule):
    def __init__(self, model : Union[FasterRCNN, FCOS, RetinaNet, SSD], 
                config: dict, iou_threshold : float = 0.5) -> None:
        super().__init__()

        # Model
        self.model = model

        # Configuration dictionary
        self.config = config

        # Classes (background inclusive)
        self.num_classes = self.model.model_num_classes

        # IoU threshold
        self.iou_threshold = iou_threshold

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x):
        self.model.eval()
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Batch
        x, y, img_path = batch  # tuple unpacking

        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())

        self.log_dict(loss_dict)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Batch
        x, y, img_path = batch

        # Inference
        preds = self.model(x)

        gt_boxes = [
            from_dict_to_boundingbox(file=target, name=name, groundtruth=True)
            for target, name in zip(y, img_path)
        ]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [
            from_dict_to_boundingbox(file=pred, name=name, groundtruth=False)
            for pred, name in zip(preds, img_path)
        ]
        pred_boxes = list(chain(*pred_boxes))

        return {"pred_boxes": pred_boxes, "gt_boxes": gt_boxes}
    
    def validation_epoch_end(self, outs):
        gt_boxes = [out["gt_boxes"] for out in outs]
        gt_boxes = list(chain(*gt_boxes))
        pred_boxes = [out["pred_boxes"] for out in outs]
        pred_boxes = list(chain(*pred_boxes))

        metric = get_pascalvoc_metrics(
            gt_boxes=gt_boxes,
            det_boxes=pred_boxes,
            iou_threshold=self.iou_threshold,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=True,
        )

        per_class, m_ap = metric["per_class"], metric["m_ap"]
        self.log("Validation_mAP", m_ap)

        for key, value in per_class.items():
            self.log(f"Validation_AP_{key}", value["AP"])

    def test_step(self, batch, batch_idx):
        # Batch
        x, y, img_path = batch

        # Inference
        preds = self.model(x)

        gt_boxes = [
            from_dict_to_boundingbox(file=target, name=name, groundtruth=True)
            for target, name in zip(y, img_path)
        ]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [
            from_dict_to_boundingbox(file=pred, name=name, groundtruth=False)
            for pred, name in zip(preds, img_path)
        ]
        pred_boxes = list(chain(*pred_boxes))

        return {"pred_boxes": pred_boxes, "gt_boxes": gt_boxes}

    def test_epoch_end(self, outs):
        gt_boxes = [out["gt_boxes"] for out in outs]
        gt_boxes = list(chain(*gt_boxes))
        pred_boxes = [out["pred_boxes"] for out in outs]
        pred_boxes = list(chain(*pred_boxes))

        metric = get_pascalvoc_metrics(
            gt_boxes=gt_boxes,
            det_boxes=pred_boxes,
            iou_threshold=self.iou_threshold,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=True,
        )

        per_class, m_ap = metric["per_class"], metric["m_ap"]
        self.log("Test_mAP", m_ap)

        for key, value in per_class.items():
            self.log(f"Test_AP_{key}", value["AP"])

    def configure_optimizers(self):
        momentum = self.config['momentum']
        lr = self.config['lr']
        weight_decay = self.config['weight_decay']
        sch = self.config['scheduler']
        steps = self.config['scheduler_steps']
        opt = self.config['optimizer']

        if opt == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, 
                momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                weight_decay=weight_decay)
        
        if sch == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                step_size=steps)
            return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
            }
        elif sch == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='max', factor=0.75, 
                patience=base.PATIENCE, min_lr=0)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "Validation_mAP"
                }
            }
        else:
            return {
                "optimizer": optimizer
            }
