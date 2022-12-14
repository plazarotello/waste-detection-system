# -*- coding: utf-8 -*-

"""Waste Detection System: Utilities

Collection of general utility functions
"""
from typing import List, Tuple, Union, Any
from pathlib import Path
from enum import Enum
from shutil import rmtree

import pandas as pd
import numpy as np

import torch
from torchvision.transforms import functional as F
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from PIL import Image
import cv2


from waste_detection_system import shared_data as base


def yolo2pascal(x : int, y : int, w : int, h : int, 
    img_w : int, img_h : int) -> Tuple[int, int, int, int]:
    """Converts the bounding box from YOLO format to Pascal format

    Args:
        x (int): x
        y (int): y
        w (int): width
        h (int): height
        img_w (int): image width
        img_h (int): image height

    Returns:
        Tuple[int, int, int, int]: xyxy bounding box
    """
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


def pascal2yolo(x1 : int, y1 : int, x2 : int, y2 : int, 
    img_w : int, img_h : int) -> Tuple[int, int, int, int]:
    """Converts from Pascal to YOLO format

    Args:
        x1 (int): x1
        y1 (int): y1
        x2 (int): x2
        y2 (int): y2
        img_w (int): image width
        img_h (int): image height

    Returns:
        Tuple[int, int, int, int]: xywh bounding box
    """
    x, y = int((x2+x1)/(2*img_w)), int((y2+y1)/(2*img_h))
    w, h = int((x2-x1)/img_w), int((y2-y1)/img_h)

    return x, y, w, h


def coco2pascal(x : int, y : int, w : int, 
    h : int) -> Tuple[int, int, int, int]:
    """Converts from COCO to Pascal format

    Args:
        x (int): x
        y (int): y
        w (int): width of the bounding box
        h (int): height of the bounding box

    Returns:
        Tuple[int, int, int, int]: xyxy bounding box
    """
    return x, y, x+w, y+h


def pascal2coco(x1 : int, y1 : int, x2 : int, 
    y2 : int) -> Tuple[int, int, int, int]:
    """Converts from Pascal to COCO format

    Args:
        x1 (int): x1
        y1 (int): y1
        x2 (int): x2
        y2 (int): y2

    Returns:
        Tuple[int, int, int, int]: xywh bounding box
    """
    return x1, y1, x2-x1, y2-y1


def yolo2coco(x : int, y : int, w : int, h : int, 
    img_w : int, img_h : int) -> Tuple[int, int, int, int]:
    """Converts from YOLO to COCO format

    Args:
        x (int): x
        y (int): y
        w (int): width of the bounding box
        h (int): height of the bounding box
        img_w (int): image width
        img_h (int): image height

    Returns:
        Tuple[int, int, int, int]: xywh bounding box
    """
    x1, x2, y1, y2 = yolo2pascal(x, y, w, h, img_w, img_h)
    return pascal2coco(x1, x2, y1, y2)


def coco2yolo(x : int, y : int, w : int, h : int, 
    img_w : int, img_h : int) -> Tuple[int, int, int, int]:
    """Converts from COCO to YOLO format

    Args:
        x (int): x
        y (int): y
        w (int): width of the bounding box
        h (int): height of the bounding box
        img_w (int): image width
        img_h (int): image height

    Returns:
        Tuple[int, int, int, int]: xywh bounding box
    """
    x1, y1, x2, y2 = coco2pascal(x, y, w, h)
    return pascal2yolo(x1, y1, x2, y2, img_w, img_h)



def show(imgs : Union[List[torch.Tensor], torch.Tensor], figsize : Tuple[int, int]=(5,5), title : str='') -> None:
    """Plots the given images side by side in a row. Taken from https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html

    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): list of images to plot
        figsize (Tuple[int, int], optional): size of the overall figure. Defaults to ``(5,5)``.
        title (str, optional): title of the overall figure. Defaults to ``''``.
    """
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
        axs[0, i].imshow(np.asarray(img))  # type: ignore
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])  # type: ignore
    fix.tight_layout()
    plt.show()


def plot_image_with_annotations(image : Union[Path,str], 
                                annotations : pd.DataFrame,
                                plot : bool = True) -> Union[torch.Tensor, None]:
    """Plots the given image with all its annotations. Taken from https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html

    Args:
        image (Union[Path,str]): path of the image
        annotations (pd.DataFrame): bounding boxes and labels of each annotation
        plot (bool, optional): if must plot or return. Defaults to ``True``.

    Returns:
        Union[torch.Tensor, None]: ``None`` when plot=``True``, ``torch.Tensor`` when plot=``False``
    """
    img = read_image(str(image))
    
    bounding_boxes = []
    colors = []

    for (_, ann) in annotations.iterrows():
        
        bbox_x = int(ann['bbox-x'])
        bbox_y = int(ann['bbox-y'])
        bbox_w = int(ann['bbox-w'])
        bbox_h = int(ann['bbox-h'])

        x1, y1, x2, y2 = coco2pascal(bbox_x, bbox_y, bbox_w, bbox_h)
        bounding_boxes.append([x1, y1, x2, y2])
        colors.append(base.COLOR_CATS[ann['label']])
    
    boxes = torch.tensor(bounding_boxes)
    result = draw_bounding_boxes(img, boxes, None, colors, width=5, fill=True)

    if plot: show(result)
    else: return result


def plot_data_sample(sample_imgs: pd.DataFrame, images_df: pd.DataFrame) -> None:
    """Plots some sample images with its corresponding annotations. Taken from https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html

    Args:
        sample_imgs (pd.DataFrame): sample images (only the path property is taken into account)
        images_df (pd.DataFrame): dataset from which to extract the annotations
    """
    images = []
    for (_, img) in sample_imgs.iterrows():
        anns : pd.DataFrame = images_df[images_df.path == img.path]  # type: ignore
        images.append(plot_image_with_annotations(str(img.path), anns, False))
    
    show(images, title='Muestra de los datos')
        


DATASET_TYPES = Enum('DatasetTypes', 'CANDIDATE FINAL COMPLEMENTARY')
"""enum: type of dataset, a choice between ``CANDIDATE``, ``FINAL`` (ZeroWaste) and ``COMPLEMENTARY`` (ResortIT)
"""

def batch_conversion_to_jpg(row : pd.Series, resize: bool = True, labelled : bool = True, 
                            dataset_type : DATASET_TYPES = DATASET_TYPES.CANDIDATE) -> pd.Series:
    """Converts an image of the dataset to JPEG images and stores it in the corresponding folder. Can be resized as well.

    Args:
        row (pd.Series): row corresponding to an annotation
        resize (bool, optional): if the image must be resized. Defaults to ``True``.
        labelled (bool, optional): if the image is labelled. Important for the resizing command. Defaults to ``True``.
        dataset_type (DATASET_TYPES, optional): type of dataset. Defaults to ``DATASET_TYPES.CANDIDATE``.

    Returns:
        pd.Series: row with properties updated
    
    Raises:
        AttributeError: if dataset_type is not amongst the ones defined.
    """
    base.CANDIDATE_DATASET.mkdir(parents=True, exist_ok=True)
    base.COMP_DATASET.mkdir(parents=True, exist_ok=True)
    base.FINAL_DATASET.mkdir(parents=True, exist_ok=True)

    path_part = None
    if dataset_type == DATASET_TYPES.CANDIDATE:
        path_part = base.CANDIDATE_DATASET
    elif dataset_type == DATASET_TYPES.COMPLEMENTARY:
        path_part = base.COMP_DATASET
    elif dataset_type == DATASET_TYPES.FINAL:
        path_part = base.FINAL_DATASET
    else:
        raise AttributeError('No dataset type')

    current_path = Path(row.path)
    prefix = 'undefined'
    for path, _prefix in base.PREFIXES_CATS.items():
        if str(path) in str(current_path):
            prefix = _prefix
            break
    new_path = path_part/Path(f'{prefix}-{current_path.stem}.jpg')
    img = Image.open(current_path)
    img = img.convert('RGB')
    if resize:
        img = resize_with_pad(img, (base.IMG_HEIGHT, base.IMG_WIDTH))
    img.save(new_path, 'jpeg')
    bbox = (0, 0, 0, 0)
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

def resize_with_pad(image : Any, new_shape: Tuple[int, int], 
                    padding_color: Tuple[int, int, int] = (0, 0, 0)):
    """Maintains aspect ratio and resizes with padding.
    Taken from: https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec

    Params:
        image (Any): Image to be resized.
        new_shape (Tuple[int, int]): Expected (width, height) of new image.
        padding_color (Tuple[int, int, int]): Tuple in BGR of padding color

    Returns:
        image (Image): Resized image with padding
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

def clean_datasets():
    """Removes the different folders for storing the different types of datasets
    """
    rmtree(base.CANDIDATE_DATASET, ignore_errors=True)
    rmtree(base.FINAL_DATASET, ignore_errors=True)
    rmtree(base.COMP_DATASET, ignore_errors=True)