import random
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from codecarbon import EmissionsTracker

from . import models, trainer, shared_data as base


def hyperparameter_search(labels: pd.DataFrame, name: str, config: dict,
        selected_model : models.AVAILABLE_MODELS, num_classes : int, 
        tll : int, resortit_zw : int, weights = None):

    epochs = config['epochs']                       # fixed
    optimizer_name = config['optimizer']            # may be list
    momentum = config['momentum']                   # may be list
    lr = config['lr']                               # may be list
    bs = config['bs']                               # may be list
    weight_decay = config['weight_decay']           # may be list

    base_model = models.get_base_model(num_classes, selected_model, tll)
    assert base_model is not None
    if weights:
        base_model = models.load_partial_weights(base_model, weights)

    momentum_list = momentum if type(momentum) is list else list(momentum)
    lr_list = lr if type(lr) is list else [lr]
    bs_list = bs if type(bs) is list else [bs]
    weight_decay_list = weight_decay if type(weight_decay) is list\
        else [weight_decay]
    optimizer_list = optimizer_name if type(optimizer_name) is list\
        else [optimizer_name]

    hyperparameter_search_space = create_initial_search_space(momentum_list, lr_list,
            bs_list, weight_decay_list, optimizer_list)

    gpu_ids = [base.GPU] if torch.cuda.is_available() and base.USE_GPU else None
    tracker = EmissionsTracker(project_name=name, experiment_id=f'hypersearch-{name}', 
        gpu_ids=gpu_ids, log_level='error', tracking_mode='process', 
        measure_power_secs=30, output_file=Path(results_dir) / f'{name}-emissions.csv')  # type: ignore
    tracker.start()

    for id, momentum, lr, bs, weight_decay, optimizer, scheduler_steps,\
        scheduler in hyperparameter_search_space:

        print(f'ID: {id} | {optimizer}({momentum}) | '+
        f'lr: {lr} | bs: {bs} | wd: {weight_decay} | '+
        f'{scheduler}({scheduler_steps})')
        
        model = base_model

        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                weight_decay=weight_decay)
        #if optimizer_name == 'SGD':
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
                momentum=momentum, weight_decay=weight_decay)
        
        if scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            step_size=scheduler_steps)
        elif scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode='min')
        else: scheduler = None

        configuration = {
            'epochs' : epochs, 'momentum' : momentum,
            'optimizer' : optimizer,
            'scheduler' : scheduler,
            'scheduler_steps' : scheduler_steps,
            'lr' : lr, 'bs' : bs,
            'weight_decay' : weight_decay,
            'checkpoint_dir' : config['checkpoint_dir'],
            'results_dir' : config['results_dir']
        }
    
        _, _ = trainer.train(model=model, 
            dataset=labels, config=configuration, 
            neptune_project=base.NEPTUNE_PROJECTS[selected_model][resortit_zw])
        
    tracker.stop()


def create_initial_search_space(momentum_list : Iterable, lr_list : Iterable, 
        bs_list : Iterable, weight_decay_list : Iterable,
        optimizer_list : Iterable) -> Iterable:

    full_search_space = product(momentum_list, lr_list, bs_list,
        weight_decay_list, optimizer_list)
    
    hyperparameter_search_space = []
    for momentum, lr, bs, weight_decay, optimizer \
        in full_search_space:

        if optimizer != 'SGD':
            momentum = -1
        hyperparameter_option = tuple(map(lambda x : tuple(x) if type(x) is list else x,
        [momentum, lr, bs, weight_decay, optimizer]))
        hyperparameter_search_space.append(hyperparameter_option)

    hyperparameter_search_space = list(set(hyperparameter_search_space))

    _tmp = list()
    for id, (momentum, lr, bs, weight_decay, optimizer)\
        in enumerate(hyperparameter_search_space):
        _tmp.append((id, momentum, lr, bs, weight_decay, optimizer))

    hyperparameter_search_space = _tmp
    return hyperparameter_search_space