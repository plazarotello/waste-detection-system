# -*- coding: utf-8 -*-

import shutil
from lightning import Trainer
from lightning.lite.utilities.seed import seed_everything
from lightning.pytorch.callbacks import (
    # EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint
)
from lightning.pytorch.loggers import NeptuneLogger
from torch import Tensor
from torchvision.models.detection import  FasterRCNN, FCOS, RetinaNet
from torchvision.models.detection.ssd import SSD

from pandas import DataFrame
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from sklearn.model_selection import train_test_split

import importlib_metadata
from neptune.new.types import File
import pathlib

from waste_detection_system import shared_data as base
from waste_detection_system import waste_detection_module
from waste_detection_system.waste_detection_module import WasteDetectionModule
from waste_detection_system.feature_extractor import HybridDLModel



def split_dataset(dataset : DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Splits a dataset in train (85%) and validation (15%)

    Args:
        dataset (DataFrame): dataset to split

    Returns:
        Tuple[DataFrame, DataFrame]: train dataset and validation dataset, in that order
        
    :meta private:
    """
    paths = dataset.path.unique()
    train_paths, val_paths = train_test_split(paths, test_size=0.15)

    return (dataset[dataset['path'].isin(train_paths)],
           dataset[dataset['path'].isin(val_paths)])



def tune(model: Union[FasterRCNN, FCOS, RetinaNet, SSD], train_dataset : DataFrame, 
        monitor_metric : str, find_lr : bool = True, find_batch_size : bool = True):
    """Searches in the hyperparameter space for an optimal initial learning rate and maximum
    batch size

    Args:
        model (_type_): model
        train_dataset (DataFrame): dataset for training
        monitor_metric (str): name of the metric to monitor. Usual values: ``Validation_mAP``, 
                             ``training_loss``, ``bbox_regression`` and ``classification``
        find_lr (bool, optional): if the task must find an optimal learning rate. Defaults to ``True``.
        find_batch_size (bool, optional): if the task must find the maximum batch size. Defaults to ``True``.
    """
    seed_everything(base.SEED)

    epochs = 100

    if base.USE_GPU:
        trainer = Trainer(
            gpus=base.GPU,
            num_sanity_val_steps=0,
            max_epochs=epochs,
            accelerator='gpu', devices=1,
            auto_lr_find=find_lr,
            auto_scale_batch_size=find_batch_size
        )
    else:
        trainer = Trainer(
            accelerator='cpu',
            num_sanity_val_steps=0,
            max_epochs=epochs,
            auto_lr_find=find_lr,
            auto_scale_batch_size=find_batch_size
        )

    lit_model = WasteDetectionModule(model=model, train_dataset=train_dataset,
        val_dataset=None, batch_size=1, lr=0.1, monitor_metric=monitor_metric)

    if find_lr:
        print('Searching for optimal initial learning rate...')
        lr_finder = trainer.tuner.lr_find(model=lit_model, min_lr=1e-9)
        
        if lr_finder: 
            print(f'Suggested initial lr: {lr_finder.suggestion()}')
            lr_finder.plot(suggest=True, show=True)

    if find_batch_size:
        print('Searching for maximum batch size...')
        batch_scaler = trainer.tuner.scale_batch_size(model=lit_model, init_val=1)
        print(f'Largest batch_size allowed: {batch_scaler}')




def train_hybrid(model : HybridDLModel, train_dataset : DataFrame, val_dataset : DataFrame, 
        ) -> HybridDLModel:
    """Trains the model with the given configuration.
    The training includes model checkpointing for the last epoch and best model regarding the
    given metric.
    The configuration dictionary must hold the ``epochs`` key (max epochs to train), 
    ``lr`` key (learning rate), ``bs`` (batch size to use).


    Args:
        model (Union[FasterRCNN, FCOS, RetinaNet, SSD]): model to train
        train_dataset (DataFrame): training dataset
        val_dataset (DataFrame): validation dataset
        config (dict): configuration dictionary
        neptune_project (str): name of the project in which to log
        metric (str): name of the metric to monitor. Usual values: ``Validation_mAP``, 
                    ``training_loss``, ``bbox_regression`` and ``classification``
        limit_validation (Union[bool, float], optional): if validation dataset must be limited (``True`` = 25%). Defaults to ``False``.

    Returns:
        Tuple[Union[Path, str], Trainer]: best model path and the trainer
    """
    seed_everything(base.SEED)

    train_loader = waste_detection_module.get_dataloader(train_dataset,
        batch_size=1)
    val_loader = waste_detection_module.get_dataloader(val_dataset,
        batch_size=1)
    
    train_images : List[Tensor] = []
    train_targets : List[Dict[str, Tensor]] = []
    for image, ground_truth, _ in train_loader:
        train_images.append(image)
        train_targets.append(ground_truth)
    
    model.train(True)
    classification_loss = model.forward(images=train_images, 
        targets=train_targets)['classification_loss'] # type: ignore
    print(f'Train classification loss: {classification_loss}')

    val_images : List[Tensor] = []
    val_targets : List[Dict[str, Tensor]] = []
    for image, ground_truth, _ in val_loader:
        val_images.append(image)
        val_targets.append(ground_truth)

    model.eval()
    val_results : List[Dict[str, Tensor]] = model.forward(images=val_images) # type: ignore
    val_loss = model.validate(x=val_results, y=val_targets)['classification_loss']
    print(f'Validation classification loss: {val_loss}')

    return model



def train(model : Union[FasterRCNN, FCOS, RetinaNet, SSD], train_dataset : DataFrame, 
        val_dataset : DataFrame, config : dict, neptune_project : str, metric : str, 
        limit_validation : Union[bool, float] = False) -> Tuple[Union[Path, None], Trainer]:
    """Trains the model with the given configuration.
    The training includes model checkpointing for the last epoch and best model regarding the
    given metric.
    The configuration dictionary must hold the ``epochs`` key (max epochs to train), 
    ``lr`` key (learning rate), ``bs`` (batch size to use).


    Args:
        model (Union[FasterRCNN, FCOS, RetinaNet, SSD]): model to train
        train_dataset (DataFrame): training dataset
        val_dataset (DataFrame): validation dataset
        config (dict): configuration dictionary
        neptune_project (str): name of the project in which to log
        metric (str): name of the metric to monitor. Usual values: ``Validation_mAP``, 
                    ``training_loss``, ``bbox_regression`` and ``classification``
        limit_validation (Union[bool, float], optional): if validation dataset must be limited (``True`` = 25%). Defaults to ``False``.

    Returns:
        Tuple[Union[Path, str], Trainer]: best model path and the trainer
    """
    seed_everything(base.SEED)

    epochs = config['epochs']
    lr = config['lr']
    bs = config['bs']
    output_dir = config['checkpoint_dir']

    if type(limit_validation) is bool:
        if limit_validation: limit_validation = base.LIMIT_VAL_BATCHES
        else: limit_validation = 1.0

    neptune_logger = NeptuneLogger(
        api_key=base.NEPTUNE_API_KEY,
        project=neptune_project,
        mode='offline'
    )

    lit_model = WasteDetectionModule(model=model, train_dataset=train_dataset,
        val_dataset=val_dataset, batch_size=bs, lr=lr, monitor_metric=metric)
    
    checkpoint_callback = ModelCheckpoint(
        monitor=metric, 
        mode='max' if metric == 'Validation_mAP' else 'min',
        save_top_k=1,
        verbose=False,
        save_last=True
        )
    learningrate_callback = LearningRateMonitor(
        logging_interval='step',
        log_momentum=False
    )
    # earlystopping_callback = EarlyStopping(
    #     monitor='Validation_mAP',
    #     patience=int(base.PATIENCE/10),
    #     mode='max'
    # )

    if base.USE_GPU:
        trainer = Trainer(
            gpus=base.GPU,
            callbacks=[checkpoint_callback, learningrate_callback],
            default_root_dir=Path(output_dir),
            max_epochs=epochs,
            logger=neptune_logger,
            accelerator='gpu',
            num_sanity_val_steps=0,
            auto_lr_find=False,
            auto_scale_batch_size=False,
            limit_val_batches=limit_validation,
            check_val_every_n_epoch=10
        )
    else:
        trainer = Trainer(
            gpus=base.GPU,
            callbacks=[checkpoint_callback, learningrate_callback],
            default_root_dir=Path(output_dir),
            max_epochs=epochs,
            logger=neptune_logger,
            num_sanity_val_steps=0,
            auto_lr_find=False,
            auto_scale_batch_size=False,
            limit_val_batches=limit_validation,
            check_val_every_n_epoch=10
        )

    trainer.fit(model=lit_model)
    
    log_packages_neptune(neptune_logger=neptune_logger)
    neptune_logger.experiment.stop()
    
    best_k_models = checkpoint_callback.best_k_models
    last_model_path = Path(checkpoint_callback.last_model_path)
    best_model_path = None
    if best_k_models:
        for path, _ in best_k_models.items():
            path = Path(path)
            if path.exists() and path.is_file():
                if not best_model_path:
                    best_model_path = path
                save_best_model(
                    checkpoint_path=path, 
                    save_directory=Path(output_dir)
                )
    elif last_model_path.exists() and last_model_path.is_file():
        best_model_path = last_model_path
        save_best_model(
            checkpoint_path=best_model_path, 
            save_directory=Path(output_dir)
        )
                
    if best_model_path is not None:
        best_model_path = Path(output_dir) / Path(best_model_path).name

    return best_model_path, trainer


def test(module : WasteDetectionModule, dataset : DataFrame) -> Any:
    """Tests the dataset in the given trainer

    Args:
        module (WasteDetectionModule): trainer to test
        dataset (DataFrame): test dataset

    Returns:
        Any: test metrics
    """
    dataloader = waste_detection_module.get_dataloader(
        data=dataset,
        batch_size=1,
        shuffle=False
        )

    if base.USE_GPU:
        trainer = Trainer(
            gpus=base.GPU,
            accelerator='gpu',
            auto_lr_find=False,
            auto_scale_batch_size=False
        )
    else:
        trainer = Trainer(
            gpus=base.GPU,
            auto_lr_find=False,
            auto_scale_batch_size=False
        )
    return trainer.test(model=module, dataloaders=dataloader)


def log_packages_neptune(neptune_logger : NeptuneLogger):
    """Flushes the logs of the run

    Args:
        neptune_logger (NeptuneLogger): neptune.ai logger
        
    :meta private:
    """
    dists = importlib_metadata.distributions()
    packages = {
        idx: (dist.metadata["Name"], dist.version) for idx, dist in enumerate(dists)
    }

    packages_df = DataFrame.from_dict(
        packages, orient="index", columns=["package", "version"]
    )

    neptune_logger.experiment['packages'].upload(File.as_html(packages_df))


def save_best_model(checkpoint_path: pathlib.Path, save_directory: pathlib.Path):
    """Copies the best model to the save directory

    Args:
        checkpoint_path (pathlib.Path): best model path
        save_directory (pathlib.Path): save directory
        
    :meta private:
    """
    shutil.copy2(src=checkpoint_path, dst=save_directory)