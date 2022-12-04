# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import os
import json
from typing import Union

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from .models import AVAILABLE_MODELS

from . import shared_data as base
from . import utils as ann_utils
from . import trainer
from . import main

# =============================================================================

def weakly_annotate(new_labels, model, config):
    if base.USE_GPU:
        device = torch.device('cuda') if torch.cuda.is_available()\
            else torch.device('cpu')
    else:
        device = torch.device('cpu')

    config = Path(config)
    with open(config, 'r') as f:
        config = json.load(f)

    labelled_data = trainer.test(model, new_labels, device)
    formatted_data = []

    for path, detection in labelled_data:
        image = Image.open(path)

        boxes = detection['boxes'].tolist()
        labels = detection['labels'].tolist()
        scores = detection['scores'].tolist()

        for box, label, score in zip(boxes, labels, scores):
            if score < config['valid_score_threshold']:
                # print(f'Discarding annotation with score={score}')
                pass
            else:
                row = {'name': Path(path).name, 'path': path,
                        'width': image.width, 'height': image.height,
                        'type': '', 'label': base.CATS_IDX[label],
                        'bbox-x': box[0], 'bbox-y': box[1],
                        'bbox-w': box[2]-box[1], 'bbox-h': box[3]-box[1]}
                formatted_data.append(row)
        image.close()
    
    if len(formatted_data) == 0:
        print(f'Annotations not relevant for {len(new_labels)} images')
        return pd.DataFrame(columns=['name', 'path', 'width', 'height',
            'type', 'label', 'bbox-x', 'bbox-y', 'bbox-w', 'bbox-h'])
    
    formatted_data = pd.DataFrame(formatted_data)
    train, test = train_test_split(formatted_data, test_size=0.2)
    train, val = train_test_split(train, test_size=0.15)
    train['type'] = 'train'
    val['type'] = 'val'
    test['type'] = 'test'
    return pd.concat([train, val, test])


def plot_results(config: Union[Path, str], name : str):
    configuration = main.configure(name, config)
    output_dir = configuration['results_dir']
    
    assert output_dir is not None

    trainer.plot_training_results(output_dir)

# =============================================================================

def pseudolabel(labels_csv: Path, new_labels: list, name: str, config: Union[Path, str],
                resume: bool = False, binary_classification : bool = False):
    with open(labels_csv, 'r') as f:
        labels = pd.read_csv(f)
    pseudolabel_df(labels, new_labels, name, config, resume, binary_classification)


def pseudolabel_df(labels: pd.DataFrame, new_labels: list, name: str, config: Union[Path, str],
                   tll: int, resume: bool = False, binary_classification : bool = False):

    model = main.train(labels, name, config, AVAILABLE_MODELS.FASTERRCNN, 
        (1 if binary_classification else 6), tll, resume)
    
    new_data = weakly_annotate(new_labels, model, config)
    if len(new_data.index) > 0:
        ann_utils.plot_data_sample(new_data.sample(n=5), new_data)

    return new_data
