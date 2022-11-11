import random
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from codecarbon import EmissionsTracker
from numpy import dtype, int16, linspace
from math import inf

from . import models, trainer


def hyperparameter_search(labels: pd.DataFrame, name: str, config: dict,
            selected_model : models.AVAILABLE_MODELS, num_classes : int):
    
    device = torch.device('cuda') if torch.cuda.is_available()\
        else torch.device('cpu')

    epochs = config['epochs']                       # fixed
    evolutions = config['evolutions']               # fixed
    optimizer_name = config['optimizer']            # may be list
    scheduler = config['scheduler']                 # may be list
    scheduler_steps = config['scheduler_steps']     # may be list
    momentum = config['momentum']                   # may be list
    lr = config['lr']                               # may be list
    bs = config['bs']                               # may be list
    weight_decay = config['weight_decay']           # may be list
    checkpoint_dir = config['checkpoint_dir']       # fixed
    results_dir = config['results_dir']             # fixed
    data_augmentation = config['data_augmentation'] # fixed

    base_model = models.get_base_model(num_classes, selected_model)
    assert base_model is not None

    momentum_list = momentum if type(momentum) is list else list(momentum)
    lr_list = lr if type(lr) is list else [lr]
    bs_list = bs if type(bs) is list else [bs]
    weight_decay_list = weight_decay if type(weight_decay) is list\
        else [weight_decay]
    optimizer_list = optimizer_name if type(optimizer_name) is list\
        else [optimizer_name]
    scheduler_list = scheduler if type(scheduler) is list\
        else [scheduler]
    scheduler_steps_list = scheduler_steps if type(scheduler_steps) is list\
        else [scheduler_steps]
    
    bag_of_mutations = create_bag_of_mutations(momentum_list, lr_list, bs_list,
            weight_decay_list, scheduler_list, scheduler_steps_list, optimizer_list)
    hyperparameter_search_space = create_initial_search_space(momentum_list, lr_list,
            bs_list, weight_decay_list, scheduler_list, scheduler_steps_list, 
            optimizer_list)

    gpu_ids = [0] if torch.cuda.is_available() else None
    tracker = EmissionsTracker(project_name=name, experiment_id=f'hypersearch-{name}', 
        gpu_ids=gpu_ids, log_level='error', tracking_mode='process', 
        measure_power_secs=30, output_file=Path(results_dir) / f'{name}-emissions.csv')
    tracker.start()

    hyperparameter_results = []
    hyperparameter_models = {}

    for evo in range(evolutions):
        print('-'*80)
        print(' '*30, f'EVOLUTION {evo+1}')
        print('-'*80)
        for id, momentum, lr, bs, weight_decay, optimizer, scheduler_steps,\
            scheduler in hyperparameter_search_space:

            print(f'ID: {id}, optimizer: {optimizer}, momentum: {momentum}\n'+
            f'lr: {lr}, bs: {bs}, weight_decay: {weight_decay}\n'+
            f'scheduler: {scheduler}, scheduler_steps: {scheduler_steps}')

            if id in hyperparameter_models:
                continue    # already trained this configuration (elite top 1)
            
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
        
            model, loss_train, val_acc = trainer.train(model, labels, bs, optimizer, 
                scheduler, epochs, checkpoint_dir, device, data_augmentation, 
                binary_classification=(num_classes==1), resume=False, save=False)
            
            min_train_loss = min(loss_train) if loss_train else inf

            min_val_loss = min([item[0] for item in val_acc]) if val_acc else inf
            max_val_map = max([item[1] for item in val_acc]) if val_acc else inf
            max_val_mar = max([item[2] for item in val_acc]) if val_acc else inf
            
            hyperparameter_models[id] = model

            hyperparameter_results.append( (evo, id, momentum, lr, bs, weight_decay,
            optimizer.__class__.__name__, scheduler.__class__.__name__, scheduler_steps,
            min_train_loss, min_val_loss, max_val_map, max_val_mar) )
            tracker.flush()
        
        hyperparameter_results_df = pd.DataFrame(hyperparameter_results,
            columns=['evolution', 'id', 'momentum', 'lr', 'bs', 'weight_decay', 
                'optimizer', 'scheduler', 'scheduler_steps',
                'train_loss', 'val_loss', 'val_map', 'val_mar'])
        if can_evolve(hyperparameter_results_df, evo):
            evolution_result = evolve(evo, hyperparameter_results_df, 
                hyperparameter_models, bag_of_mutations)
            hyperparameter_search_space = evolution_result[0]
            hyperparameter_models = evolution_result[1]
        print(hyperparameter_search_space)
        
    tracker.stop()

    hyperparameter_results_df = pd.DataFrame(hyperparameter_results,
        columns=['evolution', 'id', 'momentum', 'lr', 'bs', 'weight_decay', 
                 'optimizer', 'scheduler', 'scheduler_steps',
                 'train_loss', 'val_loss', 'val_map', 'val_mar'])
    hyperparameter_results_df.to_csv(Path(results_dir) / f'{name}.csv', 
        encoding='utf-8', index=False)


def create_initial_search_space(momentum_list : Iterable, lr_list : Iterable, 
        bs_list : Iterable, weight_decay_list : Iterable,
        scheduler_list : Iterable, scheduler_steps_list : Iterable, 
        optimizer_list : Iterable) -> Iterable:

    full_search_space = product(momentum_list, lr_list, bs_list,
        weight_decay_list, scheduler_list, scheduler_steps_list, optimizer_list)
    
    hyperparameter_search_space = []
    for momentum, lr, bs, weight_decay, scheduler, scheduler_steps, optimizer \
        in full_search_space:

        if optimizer != 'SGD':
            momentum = -1
        if scheduler != 'StepLR':
            scheduler_steps = -1
        hyperparameter_option = tuple(map(lambda x : tuple(x) if type(x) is list else x,
        [momentum, lr, bs, weight_decay, optimizer, scheduler_steps, scheduler]))
        hyperparameter_search_space.append(hyperparameter_option)

    hyperparameter_search_space = list(set(hyperparameter_search_space))

    _tmp = list()
    for id, (momentum, lr, bs, weight_decay, optimizer, scheduler_steps, scheduler)\
        in enumerate(hyperparameter_search_space):
        _tmp.append((id, momentum, lr, bs, weight_decay, optimizer, 
            scheduler_steps, scheduler))

    hyperparameter_search_space = _tmp
    return hyperparameter_search_space


def create_bag_of_mutations(momentum_list : list, lr_list : list, 
        bs_list : list, weight_decay_list : list,
        scheduler_list : list, scheduler_steps_list : list, 
        optimizer_list : list) -> dict:
    momentum_list = linspace(min(momentum_list), max(momentum_list), num=20)
    weight_decay_list = linspace(min(weight_decay_list), max(weight_decay_list), num=10)
    scheduler_steps_list = linspace(min(scheduler_steps_list), max(scheduler_steps_list), 
                                    num=10, dtype=dtype(int16))
    lr_list = linspace(min(lr_list), max(lr_list), num=25)

    return {
        'momentum' : momentum_list, 'lr' : lr_list, 'bs' : bs_list,
        'weight_decay' : weight_decay_list, 'scheduler' : scheduler_list,
        'scheduler_steps' : scheduler_steps_list, 'optimizer' : optimizer_list
    }


def can_evolve(hyperparameter_results_df : pd.DataFrame, evolution : int, 
                threshold : float = 0.8):
    hyper_evo = hyperparameter_results_df[hyperparameter_results_df.evolution == evolution]
    hyper_evo = hyper_evo.sort_values(by='val_map', ascending=False)

    current_population = len(hyper_evo.index)

    if current_population <= 2: return False
    
    best_model = hyper_evo.iloc[0]
    if best_model['val_map'] >= threshold: return False

    return True


def evolve(evolution : int, hyperparameter_results_df : pd.DataFrame, 
            hyperparameter_models : dict, bag_of_mutations : dict):
    
    def crossover(one, two, id):
        threshold = 0.5
        
        lr = one['lr'] if random.uniform(0,1) > threshold else two['lr']
        bs = one['bs'] if random.uniform(0,1) > threshold else two['bs']
        weight_decay = one['weight_decay'] if random.uniform(0,1) > threshold\
            else two['weight_decay']
        
        optimizer = one['optimizer'] if random.uniform(0,1) > threshold\
            else two['optimizer']
        
        momentum = -1
        if optimizer == 'SGD':
            if one['optimizer'] == 'SGD' and two['optimizer'] == 'SGD':
                momentum = one['momentum'] if random.uniform(0,1) > threshold\
                    else two['momentum']
            elif one['optimizer'] == 'SGD':
                momentum = one['momentum']
            else:
                momentum = two['momentum']
        
        scheduler = one['scheduler'] if random.uniform(0,1) > threshold\
            else two['scheduler']
        
        scheduler_steps = -1
        if scheduler == 'StepLR':
            if one['scheduler'] == 'StepLR' and two['scheduler'] == 'StepLR':
                scheduler_steps = one['scheduler_steps'] if random.uniform(0,1) > threshold\
                    else two['scheduler_steps']
            elif one['scheduler'] == 'StepLR':
                scheduler_steps = one['scheduler_steps']
            else:
                scheduler_steps = two['scheduler_steps']

        return pd.Series([id, momentum, lr, bs, weight_decay, optimizer, 
            scheduler, scheduler_steps], index=['id', 'momentum', 'lr', 'bs', 
            'weight_decay', 'optimizer', 'scheduler', 'scheduler_steps'])
    
    def mutate(one, bag_of_mutations):
        threshold = 0.1

        candidates = ['lr', 'bs', 'weight_decay']
        if one.optimizer == 'SGD': candidates.append('momentum')
        if one.scheduler == 'StepLR': candidates.append('scheduler_steps')

        random.shuffle(candidates)

        for el in candidates:
            if random.uniform(0,1) > threshold:
                one[el] = random.choice(bag_of_mutations[el])
                break
        
        return one
        
    hyper_evo = hyperparameter_results_df[hyperparameter_results_df.evolution == evolution]
    hyper_evo = hyper_evo.sort_values(by='val_map', ascending=False)
    
    current_population = len(hyper_evo.index)
    elitism_top1 = hyper_evo.iloc[0].squeeze()

    population = hyper_evo
    crossovered = []
    identifier = hyperparameter_results_df['id'].max()+1
    for _ in range(int((current_population-1)/2)):
        crossovered.append(crossover(
            population.sample(n=1).squeeze(),
            population.sample(n=1).squeeze(),
            identifier
        ))
        identifier+=1
    if int(len(crossovered)/4) > 0:
        for _ in random.randrange(0, int(len(crossovered)/4)):
            to_mutate = random.shuffle(crossovered).pop(0)
            crossovered.append(mutate(to_mutate, bag_of_mutations))
    
    population_list = crossovered
    population_list.append(elitism_top1)
    population_df = pd.DataFrame(population_list)[['id', 'momentum', 'lr', 'bs', 
        'weight_decay', 'optimizer', 'scheduler_steps', 'scheduler']]
    search_space = list(population_df.itertuples(index=False, name=None))
    
    elite_model = hyperparameter_models[elitism_top1.id]
    hyperparameter_models = {elitism_top1.id: elite_model}

    return search_space, hyperparameter_models