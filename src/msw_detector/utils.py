# -*- coding: utf-8 -*-

from typing import Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np

import torch
import custom_torchvision.transforms.functional as F
from custom_torchvision.io import read_image
from custom_torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from msw_detector import shared_data as base

# -----------------------------------------------------------------------------


def yolo2pascal(x, y, w, h, img_w, img_h):
    x1, y1 = int((x-w/2)*img_w), int((y-h/2)*img_h)
    x2, y2 = int((x+w/2)*img_w), int((y+h/2)*img_h)

    if x1 < 0:
        x1 = 0
    if x2 > img_w-1:
        x2 = img_w-1
    if y1 < 0:
        y1 = 0
    if y2 > img_h-1:
        y2 = img_h-1

    return x1, y1, x2, y2


def pascal2yolo(x1, y1, x2, y2, img_w, img_h):
    x, y = ((x2+x1)/(2*img_w)), ((y2+y1)/(2*img_h))
    w, h = (x2-x1)/img_w, (y2-y1)/img_h

    return x, y, w, h

# -----------------------------------------------------------------------------


def coco2pascal(x, y, w, h):
    return x, y, x+w, y+h


def pascal2coco(x1, y1, x2, y2):
    return x1, y1, x2-x1, y2-y1

# -----------------------------------------------------------------------------


def yolo2coco(x, y, w, h, img_w, img_h):
    x1, x2, y1, y2 = yolo2pascal(x, y, w, h, img_w, img_h)
    return pascal2coco(x1, x2, y1, y2)


def coco2yolo(x, y, w, h, img_w, img_h):
    x1, y1, x2, y2 = coco2pascal(x, y, w, h)
    return pascal2yolo(x1, y1, x2, y2, img_w, img_h)

# -----------------------------------------------------------------------------

# https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html

def show(imgs, figsize=(5,5), title=''):
    if not isinstance(imgs, list):
        imgs = [imgs]
    (fs_x, fs_y) = figsize
    figsize = (fs_x*len(imgs), fs_y)
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, 
                          figsize=figsize, dpi=300)
    fix.suptitle(title)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.tight_layout()
    plt.show()


def plot_image_with_annotations(image : Union[Path,str], 
                                annotations : pd.DataFrame,
                                plot : bool = True):
    img = read_image(image)
    
    bounding_boxes = []
    labels = []
    colors = []

    for (_, ann) in annotations.iterrows():
        labels.append(ann.label)
        
        bbox_x = int(ann['bbox-x'])
        bbox_y = int(ann['bbox-y'])
        bbox_w = int(ann['bbox-w'])
        bbox_h = int(ann['bbox-h'])

        x1, y1, x2, y2 = coco2pascal(bbox_x, bbox_y, bbox_w, bbox_h)
        bounding_boxes.append([x1, y1, x2, y2])
        colors.append(base.COLOR_CATS[ann['label']])
    
    boxes = torch.tensor(bounding_boxes)
    result = draw_bounding_boxes(img, boxes, labels, colors)

    if plot: show(result)
    else: return result


def plot_data_sample(sample_imgs: pd.DataFrame, images_df: pd.DataFrame):
    images = []
    for (_, img) in sample_imgs.iterrows():
        anns = images_df[images_df.path == img.path]
        images.append(plot_image_with_annotations(str(img.path), anns, False))
    
    show(images, title='Muestra de los datos')
        

# -----------------------------------------------------------------------------

def batch_conversion_to_jpg(row : pd.Series, resize: bool = True, labelled : bool = True) -> pd.Series:
    base.MOD_DATASET.mkdir(parents=True, exist_ok=True)
    current_path = Path(row.path)
    prefix = 'undefined'
    for path, _prefix in base.PREFIXES_CATS.items():
        if str(path) in str(current_path):
            prefix = _prefix
            break
    new_path = base.MOD_DATASET/Path(f'{prefix}-{current_path.stem}.jpg')
    img = Image.open(current_path)
    img = img.convert('RGB')
    if resize:
        img = resize_with_pad(img, (base.IMG_HEIGHT, base.IMG_WIDTH))
    img.save(new_path, 'jpeg')
    if labelled:
        bbox = coco2yolo(row['bbox-x'], row['bbox-y'], row['bbox-w'], row['bbox-h'], 
                            row['width'], row['height'])
    row['width'] = base.IMG_WIDTH
    row['height'] = base.IMG_HEIGHT
    if labelled:
        bbox = yolo2coco(bbox[0], bbox[1], bbox[2], bbox[3], row['width'], row['height'])
        row['bbox-x'] = bbox[0]
        row['bbox-y'] = bbox[1]
        row['bbox-w'] = bbox[2]
        row['bbox-h'] = bbox[3]
    row.name = new_path.name
    row.path = new_path
    return row

# https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
def resize_with_pad(image: Image, new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (0, 0, 0)) -> Image:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    image = np.array(image)
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return Image.fromarray(image)