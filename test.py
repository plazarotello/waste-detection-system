
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import cv2

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import random
import itertools
import json

from src.msw_detector import shared_data as base, utils, dataset_creator


# plot style
# ==============================================================================
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 16




with open(base.TRASHBOX_METAL / 'metals.csv', 'r') as f:
    metals = pd.read_csv(f)


metals.head(n=10)


len(metals)


metals['name'] = metals['image_name']
metals['path'] = metals['name'].map(lambda n: str(base.TRASHBOX_METAL / n))
metals['width'] = metals['image_width']
metals['height'] = metals['image_height']
metals['label'] = metals['label_name']
metals['bbox-x'] = metals['bbox_x']
metals['bbox-y'] = metals['bbox_y']
metals['bbox-w'] = metals['bbox_width']
metals['bbox-h'] = metals['bbox_height']


metals = metals.drop(['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 
    'image_name', 'image_width', 'image_height'], axis=1)


train, test = train_test_split(metals, test_size=0.2)
train, val = train_test_split(train, test_size=0.15)
train['type'] = 'train'
val['type'] = 'val'
test['type'] = 'test'
metals = pd.concat([train, val, test])


metals.head(n=10)


metals.info()


metals['type'].value_counts()


metals['type'].value_counts(normalize=True)

metals['label'].value_counts(normalize=True)

metals = metals.apply(utils.batch_conversion_to_jpg, axis=1)

sample_imgs = metals[(metals.type == 'train')].sample(n=5)
utils.plot_data_sample(sample_imgs, metals)

with open(base.TRASHBOX_METAL_CSV, 'w', encoding='utf-8-sig') as f:
  metals.to_csv(f, index=False)

unlabelled_metal_images = [os.path.join(base.TRASHBOX_METAL, file) 
    for file in os.listdir(base.TRASHBOX_METAL) if file.endswith('.jpg') and
    file not in metals.path.tolist()]

with open(base.TRASHBOX_METAL_CSV, 'w', encoding='utf-8-sig') as f:
  metals.to_csv(f, index=False)

labelled_metals = dataset_creator.pseudolabel_df(metals, unlabelled_metal_images,
                        'metal-pseudolabeller', 
                        config=base.PSEUDOLABELLING_DIR/'trashbox-metal_config.json', 
                        resume=False, binary_classification=True)

dataset_creator.plot_results('metal-pseudolabeller')