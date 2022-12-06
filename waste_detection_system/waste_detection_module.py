from itertools import chain
from typing import Union
from pandas import DataFrame
import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import  FasterRCNN, FCOS, RetinaNet
from torchvision.models.detection.ssd import SSD
from torchmetrics.detection.mean_ap import MeanAveragePrecision


from .utils import from_dict_to_boundingbox
from .enumerators import MethodAveragePrecision
from .pascal_voc_evaluator import get_pascalvoc_metrics
from . import shared_data as base
from .waste_detection_dataset import WasteDetectionDataset
from .bounding_box import BoundingBox



class WasteDetectionModule(pl.LightningModule):
    def __init__(self, model : Union[FasterRCNN, FCOS, RetinaNet, SSD], 
                train_dataset : DataFrame, val_dataset : Union[DataFrame, None],
                batch_size, lr, iou_threshold : float = 0.5) -> None:
        super().__init__()

        # Model
        self.model = model

        # Train dataset
        self.train_dataset = train_dataset

        # Validation dataset
        self.val_dataset = val_dataset

        # Configuration dictionary
        self.batch_size = batch_size
        self.lr = lr

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
        predictions = []
        target = []
        for out in outs:
            ground_truth = {
                'boxes' : [],
                'labels' : []
            }
            prediction = {
                'boxes' : [],
                'scores' : [],
                'labels' : []
            }

            for gt, pred in zip(out["gt_boxes"], out["pred_boxes"]):
                ground_truth['boxes'].append(list(gt.get_absolute_bounding_box()))
                ground_truth['labels'].append(gt.get_class_id())

                prediction['boxes'].append(list(pred.get_absolute_bounding_box()))
                prediction['labels'].append(pred.get_class_id())
                prediction['scores'].append(pred.get_confidence())
            
            ground_truth['boxes'] = torch.tensor(ground_truth['boxes'])  # type: ignore
            ground_truth['labels'] = torch.tensor(ground_truth['labels'])  # type: ignore

            prediction['boxes'] = torch.tensor(prediction['boxes'])  # type: ignore
            prediction['scores'] = torch.tensor(prediction['scores'])  # type: ignore
            prediction['labels'] = torch.tensor(prediction['labels'])  # type: ignore

            predictions.append(prediction)
            target.append(ground_truth)
        
        metric = MeanAveragePrecision(iou_thresholds=[self.iou_threshold],
            max_detection_thresholds=[25], class_metrics=True,
            bbox_format='xyxy', iou_type='bbox')
        metric.update(preds=predictions, target=target)
        computed_map = metric.compute()

        self.log("Validation_mAP", computed_map['map'])
        self.log("Validation_AP_PAPEL", computed_map['map_per_class'][0])
        self.log("Validation_AP_PLASTICO", computed_map['map_per_class'][1])

        # gt_boxes = [out["gt_boxes"] for out in outs]
        # gt_boxes = list(chain(*gt_boxes))
        # pred_boxes = [out["pred_boxes"] for out in outs]
        # pred_boxes = list(chain(*pred_boxes))

        # metric = get_pascalvoc_metrics(
        #     gt_boxes=gt_boxes,
        #     det_boxes=pred_boxes,
        #     iou_threshold=self.iou_threshold,
        #     method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
        #     generate_table=True,
        # )

        # per_class, m_ap = metric["per_class"], metric["m_ap"]
        # self.log("Validation_mAP", m_ap)

        # for key, value in per_class.items():
        #     self.log(f"Validation_AP_{key}", value["AP"])

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
        optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.lr or self.hparams.lr)  # type: ignore

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
    

    def train_dataloader(self):
        return get_dataloader(data=self.train_dataset, shuffle=True,
            batch_size=self.batch_size or self.hparams.batch_size)  # type: ignore
    

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return get_dataloader(data=self.val_dataset, shuffle=False,
            batch_size=self.batch_size or self.hparams.batch_size)  # type: ignore




def collate_double(batch):
    x = [sample['x'] for sample in batch]
    y = [sample['y'] for sample in batch]
    img_path = [sample['path'] for sample in batch]
    return x, y, img_path



def get_dataloader(data : DataFrame, batch_size : int, shuffle : bool = True):
    mapping = { base.CATS_PAPEL : 1, base.CATS_PLASTICO : 2}
    dataset = WasteDetectionDataset(data=data, mapping=mapping)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=0, collate_fn=collate_double)
    
    return dataloader