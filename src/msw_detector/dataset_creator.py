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
from msw_detector.models import AVAILABLE_MODELS

from msw_detector import shared_data as base
from msw_detector import utils as ann_utils
from msw_detector import trainer
from msw_detector import main

# -----------------------------------------------------------------------------
# YOLOv5 dataset formatting with YOLO annotations
# (class_idx norm_bb_cx norm_bb_cy norm_bb_w norm_bb_h)
# =============================================================================
#
# current folder
# ├── models
# |   └── ultralytics_yolov5_master
# └── dataset
#     ├── train
#     ├── val
#     └── test
#
# -----------------------------------------------------------------------------


def get_path_train() -> Path:
    return base.YOLO_DATA_FOLDER/base.YOLO_DATA_TRAIN


def get_path_val() -> Path:
    return base.YOLO_DATA_FOLDER/base.YOLO_DATA_VAL


def get_path_test() -> Path:
    return base.YOLO_DATA_FOLDER/base.YOLO_DATA_TEST


def create_yolo_structure(df_path: Path = base.DATA_CSV):
    with open(df_path, 'r', encoding='utf-8-sig') as f:
        df = pd.read_csv(f)

    # delete directories from previous runs
    shutil.rmtree(str(base.YOLO_DATA_FOLDER/base.YOLO_DATA_TRAIN),
                  ignore_errors=True)
    shutil.rmtree(str(base.YOLO_DATA_FOLDER/base.YOLO_DATA_VAL),
                  ignore_errors=True)
    shutil.rmtree(str(base.YOLO_DATA_FOLDER/base.YOLO_DATA_TEST),
                  ignore_errors=True)

    # create the directories
    os.makedirs(str(base.YOLO_DATA_FOLDER/base.YOLO_DATA_TRAIN), exist_ok=True)
    os.makedirs(str(base.YOLO_DATA_FOLDER/base.YOLO_DATA_VAL), exist_ok=True)
    os.makedirs(str(base.YOLO_DATA_FOLDER/base.YOLO_DATA_TEST), exist_ok=True)

    # copy images and generate annotations in YOLO format
    df.reset_index()
    images = df['path'].unique()

    for idx, img in enumerate(images):
        annotations = df[df['path'] == img]
        sample_row = annotations.iloc[0]

        img_name = f'{idx}.jpg'
        ann_name = f'{idx}.txt'

        if sample_row.type == base.YOLO_DATA_TRAIN:
            _path = get_path_train()
        elif sample_row.type == base.YOLO_DATA_VAL:
            _path = get_path_val()
        elif sample_row.type == base.YOLO_DATA_TEST:
            _path = get_path_test()
        else:
            print(f'Incorrect type in row: {sample_row}')
            return
        
        _img_path = _path/img_name
        _label_path = _path/ann_name

        anns_dict = {'label_idx' : [], 'x' : [],
        'y' : [], 'w' : [], 'h' : []}

        src = Image.open(sample_row.path)
        src.verify()
        src.close()
        src = Image.open(sample_row.path)
        src.save(_img_path)

        for _, row in annotations.iterrows():
            _bb = [row['bbox-x'], row['bbox-y'], row['bbox-w'], row['bbox-h']]
            _x, _y, _w, _h = ann_utils.coco2yolo(_bb[0], _bb[1], _bb[2], _bb[3],
                                                row.width, row.height)
            _label_idx = base.IDX_CATS[row.label]
            anns_dict['label_idx'] = anns_dict['label_idx'] + [_label_idx]
            anns_dict['x'] = anns_dict['x'] + [_x]
            anns_dict['y'] = anns_dict['y'] + [_y]
            anns_dict['w'] = anns_dict['w'] + [_w]
            anns_dict['h'] = anns_dict['h'] + [_h]

        _ann = pd.DataFrame(data=anns_dict)
        # write or append the annotation
        with open(_label_path, 'w') as f:
            _ann.to_csv(f, index=False, sep=' ', header=False)

# =============================================================================

def weakly_annotate(new_labels, model, config):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

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
                   resume: bool = False, binary_classification : bool = False):

    model = main.train(labels, name, config, AVAILABLE_MODELS.FASTERRCNN, 
        (1 if binary_classification else 6), resume)
    
    new_data = weakly_annotate(new_labels, model, config)
    if len(new_data.index) > 0:
        ann_utils.plot_data_sample(new_data.sample(n=5), new_data)

    return new_data
