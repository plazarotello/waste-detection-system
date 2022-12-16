# -*- coding: utf-8 -*-

"""
.. _tutorial: https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial
.. _tutorial2: https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70
.. _author: https://github.com/johschmidt42

Class representing the training, validation and testing logic used in the Waste 
Detection System, based on `this tutorial <tutorial_>`_ (available on `medium <tutorial2_>`_).

Original author: `Johannes Schmidt <author_>`_

Further expanded to include Non-Maximum Suppresion, support for different metrics, 
optimizers' configuration and modified validation and test phases.
"""

from typing import Any, Dict, List, Tuple, Union
from statistics import mean
from pandas import DataFrame
import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import  FasterRCNN, FCOS, RetinaNet
from torchvision.models.detection.ssd import SSD
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from waste_detection_system import shared_data as base
from waste_detection_system.waste_detection_dataset import WasteDetectionDataset
from waste_detection_system.transformations import apply_nms, apply_score_threshold


class WasteDetectionModule(pl.LightningModule):
    """The Lightning Module for training, validating, testing and predicting in the
    Waste Detection System.
    """
    def __init__(self, model : Union[FasterRCNN, FCOS, RetinaNet, SSD], 
                train_dataset : DataFrame, val_dataset : Union[DataFrame, None],
                batch_size : int, lr : float, monitor_metric : str, 
                iou_threshold : float = 0.5) -> None:
        """Initializes the ``WasteDetectionModule``

        Args:
            model (Union[FasterRCNN, FCOS, RetinaNet, SSD]): model to use
            train_dataset (DataFrame): dataset used for training
            val_dataset (Union[DataFrame, None]): dataset used for validation. Can be ``None``.
            batch_size (int): batch size
            lr (float): initial learning rate
            monitor_metric (str): metric to use in optimizers
            iou_threshold (float, optional): IoU threshold. Defaults to 0.5.
        """
        super().__init__()

        # Model
        self.model = model

        # Train dataset
        self.train_dataset = train_dataset

        # Validation dataset
        self.val_dataset = val_dataset

        # Configuration dictionary
        if self.val_dataset is not None:
            self.train_batch_size = int(batch_size/2)
            self.val_batch_size = int(batch_size/2)
        else:
            self.train_batch_size = batch_size
            self.val_batch_size = 1
        
        self.batch_size = batch_size
        self.lr = lr

        # Classes (background inclusive)
        self.num_classes = self.model.model_num_classes

        # Metric
        self.metric_to_monitor = monitor_metric

        # IoU threshold
        self.iou_threshold = iou_threshold
        # Score threshold
        self.score_threshold = 0.5

        # Save hyperparameters
        self.hparams['model'] = self.model
        self.hparams['train_batch_size'] = self.train_batch_size
        self.hparams['val_batch_size'] = self.val_batch_size
        self.save_hyperparameters()
    


    def forward(self, x):
        self.model.eval()
        return self.model(x)
    

    def on_train_epoch_start(self) -> None:
        self.epoch_loss = []
        return super().on_train_epoch_start()

    
    def training_step(self, batch, batch_idx):
        # Batch
        x, y, img_path = batch  # tuple unpacking

        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())

        self.epoch_loss.append(loss.item())  # type: ignore

        self.log_dict(loss_dict)
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log("training_loss", mean(self.epoch_loss))
        return super().on_train_epoch_end()


    def on_validation_start(self) -> None:
        super().on_validation_start()

        self.map_metric = MeanAveragePrecision()

    def validation_step(self, batch, batch_idx):
        # Batch
        x, y, img_path = batch

        # Inference
        preds = self.model(x)
        nms_predictions = self.apply_thresholds(preds)

        self.map_metric.update(preds=nms_predictions, target=y)

        return nms_predictions

    def validation_epoch_end(self, outs):
        computed_map = self.map_metric.compute()

        self.log("Validation_mAP", computed_map['map'])
        self.log("Validation_metrics", computed_map)




    def on_test_start(self) -> None:
        super().on_test_start()

        self.map_metric = MeanAveragePrecision()

    def test_step(self, batch, batch_idx):
        # Batch
        x, y, img_path = batch

        # Inference
        preds = self.model(x)
        nms_predictions = self.apply_thresholds(preds)

        self.map_metric.update(preds=nms_predictions, target=y)

        return preds

    def test_epoch_end(self, outs):
        computed_map = self.map_metric.compute()

        self.log("Test_mAP", computed_map['map'])
        self.log("Test_metrics", computed_map)


    def apply_thresholds(self, predictions: List[Dict]):
        """Apply score threshold and IoU threshold to the predictions to filter the irrelevant ones

        Args:
            predictions (List[Dict]): predictions

        Returns:
            List[Dict]: predictions filtered out from NMS and score threshold 
        """
        nms_predictions = []
        for pred in predictions:
            cpu_prediction= {
                'boxes': [box.detach().cpu() for box in pred['boxes']],
                'scores': [score.detach().cpu() for score in pred['scores']],
                'labels': [label.detach().cpu() for label in pred['labels']],
            }
            nms_predictions.append(apply_nms(target=
                    apply_score_threshold(target=cpu_prediction, 
                        score_threshold=self.score_threshold), 
                iou_threshold=self.iou_threshold))
        
        return nms_predictions


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.lr or self.hparams.lr)  # type: ignore

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='max' if self.metric_to_monitor == 
                'Validation_mAP' else 'min', factor=0.9, 
            patience=3, min_lr=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.metric_to_monitor
            }
        }



    def train_dataloader(self):
        real_batch_size = self.train_batch_size or self.batch_size or \
            self.hparams.batch_size  # type: ignore
        return get_dataloader(data=self.train_dataset, shuffle=True,
            batch_size=real_batch_size)  # type: ignore
    

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return get_dataloader(data=self.val_dataset, shuffle=True,
            batch_size=self.val_batch_size or 1)  # type: ignore



def collate_double(batch : Dict) -> Tuple[Any, Any, Any]:
    """Collate function

    Args:
        batch (Dict): contains ``x``, ``y`` and ``path``

    Returns:
        Tuple[Any, Any, Any]: ``x``, ``y``, ``path`` in that order
    """
    x = [sample['x'] for sample in batch]
    y = [sample['y'] for sample in batch]
    img_path = [sample['path'] for sample in batch]
    return x, y, img_path



def get_dataloader(data : DataFrame, batch_size : int, shuffle : bool = True) -> DataLoader[WasteDetectionDataset]:
    """Creates a dataloader

    Args:
        data (DataFrame): data
        batch_size (int): batch size
        shuffle (bool, optional): if shuffle is allowed. Defaults to True.

    Returns:
        DataLoader[WasteDetectionDataset]: dataloader
    """
    mapping = { base.CATS_PAPEL : 1, base.CATS_PLASTICO : 2}
    dataset = WasteDetectionDataset(data=data, mapping=mapping)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=0, collate_fn=collate_double)  # type: ignore
    
    return dataloader