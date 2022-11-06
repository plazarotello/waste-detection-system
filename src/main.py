# -*- coding: utf-8 -*-
from pathlib import Path

import json
from typing import Union

from codecarbon import EmissionsTracker

import pandas as pd

import torch

from . import models
from . import trainer
from . import hyperparameter_search as hyper


def configure(name: str, config: Union[Path, str]):
    config = Path(config)
    chk_dir = config.parent.resolve()
    with open(config, 'r') as f:
        config = json.load(f)
    
    epochs = config['epochs']
    momentum = config['momentum']
    evolutions = config['evolutions'] if 'evolutions' in config.keys()\
        else 10
    optimizer = config['optimizer'] if 'optimizer' in config.keys()\
        else 'SGD'
    scheduler = config['scheduler'] if 'scheduler' in config.keys()\
        else None
    scheduler_steps = config['scheduler_steps'] if 'scheduler_steps' in\
        config.keys() else None

    lr = config['lr']

    weight_decay = config['weight_decay']

    bs = config['batch_size']
    checkpoint_dir = chk_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(checkpoint_dir)

    data_augmentation = config['data_augmentation'] if 'data_augmentation' in config.keys()\
        else False
    
    results_dir = Path(config['results_dir']) / name if 'results_dir' in config.keys()\
        else checkpoint_dir

    return {
        'epochs' : epochs, 'momentum' : momentum, 
        'evolutions' : evolutions,
        'optimizer' : optimizer,
        'scheduler' : scheduler,
        'scheduler_steps' : scheduler_steps,
        'lr' : lr, 'bs' : bs,
        'weight_decay' : weight_decay,
        'checkpoint_dir' : checkpoint_dir,
        'data_augmentation' : data_augmentation,
        'results_dir' : results_dir
    }


def hyperparameter_search(labels: pd.DataFrame, name: str, config: Union[Path, str],
    selected_model : models.AVAILABLE_MODELS, num_classes : int):
    config = configure(name, config)
    hyper.hyperparameter_search(labels, name, config, selected_model, num_classes)

def train(labels: pd.DataFrame, name: str, config: Union[Path, str],
            selected_model : models.AVAILABLE_MODELS, num_classes : int, 
            resume: bool = False):
    device = torch.device('cuda') if torch.cuda.is_available()\
        else torch.device('cpu')
    # device = torch.device('cpu')

    config = configure(name, config)
    epochs = config['epochs']
    momentum = config['momentum']
    lr = config['lr']
    bs = config['bs']
    weight_decay = config['weight_decay']

    sch = config['scheduler']
    steps = config['scheduler_steps']

    opt = config['optimizer']

    checkpoint_dir = config['checkpoint_dir']
    data_augmentation = config['data_augmentation']

    model = models.get_base_model(num_classes, selected_model)
    assert model is not None
    
    if opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
            momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
            weight_decay=weight_decay)
    
    if sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
            step_size=steps)
    elif sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='max')
    else:
        scheduler = None
    
    gpu_ids = [0] if torch.cuda.is_available() else None
    
    tracker = EmissionsTracker(project_name=name, experiment_id='train', gpu_ids=gpu_ids, 
        log_level='error', tracking_mode='process', measure_power_secs=30)

    tracker.start()
    model, _, _ = trainer.train(model, labels, bs, optimizer, scheduler, 
        epochs, checkpoint_dir, device, data_augmentation, 
        binary_classification=(num_classes==1), resume=resume, save=True)
    tracker.stop()
    model = trainer.choose_best_model(checkpoint_dir, model)
    return model