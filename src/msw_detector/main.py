# -*- coding: utf-8 -*-

import collections
from pathlib import Path
from itertools import product
import json
from typing import Union

from codecarbon import EmissionsTracker

import pandas as pd
import numpy as np

import torch

from . import models
from . import trainer


def configure(name: str, config: Union[Path, str]):
    config = Path(config)
    chk_dir = config.parent.resolve()
    with open(config, 'r') as f:
        config = json.load(f)
    
    epochs = config['epochs']
    momentum = config['momentum']

    lr = config['lr']
    lr_steps = config['lr_steps']
    lr_gamma = config['lr_gamma']

    weight_decay = config['weight_decay']

    bs = config['batch_size']
    checkpoint_dir = chk_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(checkpoint_dir)

    data_augmentation = config['data_augmentation'] if 'data_augmentation' in config.keys()\
        else False
    
    results_dir = Path(config['results_dir']) / name if 'results_dir' in config.keys()\
        else None
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        results_dir = str(results_dir)

    return {
        'epochs' : epochs, 'momentum' : momentum, 
        'lr' : lr, 'lr_steps' : lr_steps, 
        'lr_gamma' : lr_gamma, 'bs' : bs,
        'weight_decay' : weight_decay,
        'checkpoint_dir' : checkpoint_dir,
        'data_augmentation' : data_augmentation,
        'results_dir' : results_dir
    }


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
    lr_steps = config['lr_steps']
    lr_gamma = config['lr_gamma']
    bs = config['bs']
    weight_decay = config['weight_decay']
    checkpoint_dir = config['checkpoint_dir']
    data_augmentation = config['data_augmentation']

    model = models.get_base_model(num_classes, selected_model)
    assert model is not None

    # param_groups = split_normalization_params(model)
    # wd_groups = [weight_decay]
    # parameters = [{'params': p, 'weight_decay': w}
    #               for p, w in zip(param_groups, wd_groups) if p]

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
        weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
        milestones=lr_steps, gamma=lr_gamma)
    
    gpu_ids = [0] if torch.cuda.is_available()\
        else None
    
    tracker = EmissionsTracker(project_name=name, experiment_id='train', gpu_ids=gpu_ids, 
        log_level='error', tracking_mode='process', measure_power_secs=30)

    tracker.start()
    model, _, _ = trainer.train(model, labels, bs, optimizer, lr_scheduler,
        epochs, checkpoint_dir, device, data_augmentation, 
        binary_classification=(num_classes==1), resume=resume, save=True)
    tracker.stop()
    model = trainer.choose_best_model(checkpoint_dir, model)
    return model


def hyperparameter_search(labels: pd.DataFrame, name: str, config: Union[Path, str],
            selected_model : models.AVAILABLE_MODELS, num_classes : int):
    device = torch.device('cuda') if torch.cuda.is_available()\
        else torch.device('cpu')

    config = configure(name, config)
    epochs = config['epochs']                       # fixed
    momentum = config['momentum']                   # may be list
    lr = config['lr']                               # may be list
    lr_steps = config['lr_steps']                   # fixed
    lr_gamma = config['lr_gamma']                   # may be list
    bs = config['bs']                               # may be list
    weight_decay = config['weight_decay']           # may be list
    checkpoint_dir = config['checkpoint_dir']       # fixed
    results_dir = config['results_dir']             # fixed
    data_augmentation = config['data_augmentation'] # fixed

    model = models.get_base_model(num_classes, selected_model)
    assert model is not None

    # param_groups = split_normalization_params(model)

    momentum_list = momentum if isinstance(collections.Iterable)\
        else list(momentum)
    lr_list = lr if isinstance(collections.Iterable)\
        else list(lr)
    lr_gamma_list = lr_gamma if isinstance(collections.Iterable)\
        else list(lr_gamma)
    bs_list = bs if isinstance(collections.Iterable)\
        else list(bs)
    weight_decay_list = weight_decay if isinstance(collections.Iterable)\
        else list(weight_decay)

    hyperparameter_search_space = product(momentum_list, lr_list, lr_gamma_list, 
        bs_list, weight_decay_list)

    gpu_ids = [0] if torch.cuda.is_available() else None
    tracker = EmissionsTracker(project_name=name, experiment_id=f'hypersearch-{name}', 
        gpu_ids=gpu_ids, log_level='error', tracking_mode='process', 
        measure_power_secs=30)
    tracker.start()

    hyperparameter_results = []
    for momentum, lr, lr_gamma, bs, weight_decay in hyperparameter_search_space:

        # wd_groups = [weight_decay]
        # parameters = [{'params': p, 'weight_decay': w}
        #             for p, w in zip(param_groups, wd_groups) if p]

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
            weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
            milestones=lr_steps, gamma=lr_gamma)
    
        model, loss_train, val_acc = trainer.train(model, labels, bs, optimizer, 
            lr_scheduler, epochs, checkpoint_dir, device, data_augmentation, 
            binary_classification=(num_classes==1), resume=False, save=False)
        
        min_train_loss = min(loss_train)
        train_loss_idx = np.argmin(loss_train)

        min_val_loss = min([item[0].item() for item in val_acc])
        val_loss_idx = np.argmin([item[0].item() for item in val_acc])+1
        max_val_map = max([item[1].item() for item in val_acc])+1
        val_map_idx = np.argmax([item[1].item() for item in val_acc])+1
        max_val_mar = max([item[2].item() for item in val_acc])+1
        val_mar_idx = np.argmax([item[2].item() for item in val_acc])+1
        
        hyperparameter_results.append( (momentum, lr, lr_gamma, bs, weight_decay,
           min_train_loss, train_loss_idx, min_val_loss, val_loss_idx, 
           max_val_map, val_map_idx, max_val_mar, val_mar_idx) )
        tracker.flush()
    
    tracker.stop()

    hyperparameter_results_df = pd.DataFrame(hyperparameter_results,
        columns=['momentum', 'lr', 'lr_gamma', 'bs', 'weight_decay', 
            'train_loss', 'train_loss_epoch', 'val_loss', 'val_loss_epoch', 
            'val_map', 'val_map_epoch', 'val_mar', 'val_mar_epoch'])
    hyperparameter_results_df.to_csv(results_dir / f'{name}.csv', 
        encoding='utf-8', index=False)