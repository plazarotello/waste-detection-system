# -*- coding: utf-8 -*-

"""Waste Detection System

Collection of entry points to the module, namely the hyperparameter seach 
and trainer. 
"""

from pathlib import Path
import json
import os
from typing import Any, Union, Dict
from codecarbon import EmissionsTracker
import pandas as pd
import torch


from waste_detection_system import shared_data as base
from waste_detection_system import models
from waste_detection_system import trainer
from waste_detection_system.waste_detection_module import WasteDetectionModule
from waste_detection_system.feature_extractor import HybridDLModel



def configure(name: str, config: Union[Path, str]) -> Dict[str, Any]:
    """Obtains the configuration key-values from the specified JSON file.

    Args:
        name (str): name of the configuration, used for creating the 
                    checkpoint and results directory
        config (Union[Path, str]): path to the configuration JSON file

    Returns:
        Dict[str, Any]: a dictionary with the configuration from the JSON file
    """
    configuration = Path(config)
    chk_dir = configuration.parent.resolve()
    with open(configuration, 'r') as f:
        configuration = json.load(f)
    
    epochs = configuration['epochs'] if 'epochs' in configuration.keys() else 100
    momentum = configuration['momentum'] if 'momentum' in configuration.keys() else 0.9
    optimizer = configuration['optimizer'] if 'optimizer' in configuration.keys()\
        else 'SGD'
    scheduler = configuration['scheduler'] if 'scheduler' in configuration.keys()\
        else None
    scheduler_steps = configuration['scheduler_steps'] if 'scheduler_steps' in\
        configuration.keys() else None

    lr = configuration['lr'] if 'lr' in configuration.keys() else 0.1

    weight_decay = configuration['weight_decay'] if 'weight_decay' in configuration.keys() else 0.001

    bs = configuration['batch_size'] if 'batch_size' in configuration.keys() else 1
    checkpoint_dir = chk_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(checkpoint_dir)
    
    results_dir = Path(configuration['results_dir']) / name if 'results_dir' in configuration.keys()\
        else checkpoint_dir

    return {
        'epochs' : epochs, 'momentum' : momentum,  'optimizer' : optimizer,
        'scheduler' : scheduler, 'scheduler_steps' : scheduler_steps,
        'lr' : lr, 'bs' : bs, 'weight_decay' : weight_decay,
        'checkpoint_dir' : checkpoint_dir, 'results_dir' : results_dir
    }



def hyperparameter_search(name: str, dataset : pd.DataFrame, config: Union[Path, str],
    selected_model : base.AVAILABLE_MODELS, num_classes : int, tll : int,
    metric : str, find_lr : bool = True, find_batch_size : bool = True,
    weights: Union[os.PathLike, str, Any, None] = None):
    """Searches optimal hyperparameters (namely maximum batch size an optimal 
    initial learning rate)

    Args:
        name (str): name of the task
        dataset (pd.DataFrame): training dataset to optimize for
        config (Union[Path, str]): path to the configuration file
        selected_model (base.AVAILABLE_MODELS): model for the task
        num_classes (int): number of classes in the dataset
        tll (int): Transfer Learning Level, coded as:
        
                        - TLL = 0 : train from scratch (all layers)
                        - TLL = 1 : use transfer learning and train only the classification and regression heads
                        - TLL > 1 : use fine-tuning and train the heads as well as some more layers. MINIMUM = 2, MAXIMUM = 5
        metric (str): metric to monitor
        find_lr (bool, optional): if the task must find an optimal initial learning rate. 
                                Defaults to ``True``.
        find_batch_size (bool, optional): if the task must find the maximum batch size. 
                                Defaults to ``True``.
        weights (Union[os.PathLike, str, Any, None], optional): weights to apply to the model.
                                Defaults to None.
    """
    configuration = configure(name, config)

    if weights:
        if type(weights) is os.PathLike or type(weights) is str:
            weights = torch.load(weights)
    
    base_model = models.get_base_model(num_classes, selected_model, tll)
    assert base_model is not None
    if weights:
        base_model = models.load_partial_weights(base_model, weights)

    model = base_model

    gpu_ids = [base.GPU] if torch.cuda.is_available() and base.USE_GPU else None
    tracker = EmissionsTracker(project_name=name, experiment_id=f'hypersearch-{name}', 
        gpu_ids=gpu_ids, log_level='error', tracking_mode='process', 
        measure_power_secs=30, output_file=Path(configuration['results_dir']) / f'{name}-emissions.csv')  # type: ignore
    

    tracker.start()
    trainer.tune(model=model, train_dataset=dataset, monitor_metric = metric, find_lr=find_lr, 
                find_batch_size=find_batch_size)
    tracker.stop()




def train(train_dataset: pd.DataFrame, val_dataset: pd.DataFrame, name: str, 
            config: Union[Path, str], resortit_zw : int, metric : str,
            selected_model : base.AVAILABLE_MODELS, num_classes : int, tll : int,
            limit_validation : Union[bool, float] = False,
            weights: Union[os.PathLike, str, Any,  None] = None):
    """Trains a selected model

    Args:
        train_dataset (pd.DataFrame): dataset used for training
        val_dataset (pd.DataFrame): dataset used for validation
        name (str): name of the task
        config (Union[Path, str]): path to the configuration JSON file
        resortit_zw (int): ``0`` if ResortIT dataset, ``1`` if ZeroWaste
                            Used for neptune.ai logger
        metric (str): metric to optimize
        selected_model (base.AVAILABLE_MODELS): model to train
        num_classes (int): number of classes in the dataset
        tll (int): Transfer Learning Level.
                    TLL = 0 : train from scratch (all layers)
                    TLL = 1 : use transfer learning and train only the 
                    classification and regression heads
                    TLL > 1 : use fine-tuning and train the heads as well as
                    some more layers. MINIMUM = 2, MAXIMUM = 5
        limit_validation (Union[bool, float], optional): if validation dataset must be limited. 
                                                        Accepts a percentage or a fllag, in which case 
                                                        validation is limited to 25%. Defaults to ``False``.
        weights (Union[os.PathLike, str, Any,  None], optional): weights to initialize the model with. 
                                                                Defaults to ``None``.
    """
    configuration = configure(name, config)
    
    model = models.get_base_model(num_classes, selected_model, tll)
    assert model is not None

    if weights:
        if type(weights) is os.PathLike or type(weights) is str:
            weights = torch.load(weights)
        model = models.load_partial_weights(model, weights)
    
    gpu_ids = [base.GPU] if torch.cuda.is_available() and base.USE_GPU else None
    
    tracker = EmissionsTracker(project_name=name, experiment_id='train', gpu_ids=gpu_ids, 
        log_level='error', tracking_mode='process', measure_power_secs=30)  # type: ignore

    tracker.start()
    best_model_path, model_trainer = trainer.train(model=model, train_dataset=train_dataset, 
        val_dataset=val_dataset, config=configuration, limit_validation=limit_validation,
        neptune_project=base.NEPTUNE_PROJECTS[selected_model][resortit_zw], metric=metric)
    tracker.stop()


def train_hybrid(train_dataset: pd.DataFrame, val_dataset: pd.DataFrame, name: str, 
                config: Union[Path, str], resortit_zw : int, num_classes : int,
                selected_model : base.AVAILABLE_MODELS, 
                selected_classifier : base.AVAILABLE_CLASSIFIERS,
                weights: Union[os.PathLike, str, Any,  None] = None) -> HybridDLModel:
    """Trains a selected hybrid model

    Args:
        train_dataset (pd.DataFrame): dataset used for training
        val_dataset (pd.DataFrame): dataset used for validation
        name (str): name of the task
        config (Union[Path, str]): path to the configuration JSON file
        selected_model (base.AVAILABLE_MODELS): model to train
        num_classes (int): number of classes in the dataset
        weights (Union[os.PathLike, str, Any,  None], optional): weights to initialize the model with. 
                                                                Defaults to ``None``.
    """
    assert weights is not None

    configuration = configure(name, config)
    configuration['epochs'] = 1
    if weights:
        if type(weights) is os.PathLike or type(weights) is str:
            weights = torch.load(weights)
    model = models.get_hybrid_model(num_classes=num_classes, chosen_model=selected_model,
            chosen_classifier=selected_classifier, weights=weights)
    assert model is not None
    
    gpu_ids = [base.GPU] if torch.cuda.is_available() and base.USE_GPU else None
    
    tracker = EmissionsTracker(project_name=name, experiment_id='train', gpu_ids=gpu_ids, 
        log_level='error', tracking_mode='process', measure_power_secs=30)  # type: ignore

    tracker.start()
    model = trainer.train_hybrid(model, train_dataset, val_dataset)
    tracker.stop()
    return model






def save_weights(weights : Any, save_path : Union[str, Path]):
    """Saves the weights to disk

    Args:
        weights (Any): weights to save
        save_path (Union[str, Path]): file saving path
    """
    torch.save(weights, save_path)



def load_weights_from_checkpoint(checkpoint_path : Union[str, Path], 
            selected_model : base.AVAILABLE_MODELS, num_classes : int) -> Dict[str, Any]:
    """Loads the selected weights in the selected model

    Args:
        checkpoint_path (Union[str, Path]): path to the checkpoint holding the weights or the weights directly
        selected_model (base.AVAILABLE_MODELS): model in which to load the weights
        num_classes (int): number of classes of the model

    Returns:
        Dict[str, Any]: weights of the model
    """
    try:
        module = WasteDetectionModule.load_from_checkpoint(checkpoint_path=checkpoint_path)
    except Exception:
        fake_df = pd.DataFrame({})
        module = WasteDetectionModule(
            model=models.get_base_model(num_classes, selected_model, 1), 
            train_dataset=fake_df, val_dataset=None, batch_size=128, lr=1, 
            monitor_metric='training_loss')
        checkpoint = torch.load(checkpoint_path)
        module.load_state_dict(checkpoint['state_dict'])

    return module.model.state_dict()