from itertools import chain
from typing import Union
from pandas import DataFrame
import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import  FasterRCNN, FCOS, RetinaNet
from torchvision.models.detection.ssd import SSD
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from . import shared_data as base
from .waste_detection_dataset import WasteDetectionDataset



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
    

    def on_validation_start(self) -> None:
        super().on_validation_start()

        self.map_metric = MeanAveragePrecision()

    def validation_step(self, batch, batch_idx):
        # Batch
        x, y, img_path = batch

        # Inference
        preds = self.model(x)

        self.map_metric.update(preds=preds, target=y)

        return preds

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

        self.map_metric.update(preds=preds, target=y)

        return preds

    def test_epoch_end(self, outs):
        computed_map = self.map_metric.compute()

        self.log("Test_mAP", computed_map['map'])
        self.log_dict("Test_metrics", computed_map)

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
        real_batch_size = self.train_batch_size or self.batch_size or \
            self.hparams.batch_size
        return get_dataloader(data=self.train_dataset, shuffle=True,
            batch_size=real_batch_size)  # type: ignore
    

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return get_dataloader(data=self.val_dataset, shuffle=True,
            batch_size=self.val_batch_size or 1)  # type: ignore




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