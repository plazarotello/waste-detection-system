# -*- coding: utf-8 -*-

"""
.. _tutorial: https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial
.. _tutorial2: https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70
.. _author: https://github.com/johschmidt42

Class representing the dataset used in the Waste Detection System, created following 
`this tutorial <tutorial_>`_ (available on `medium <tutorial2_>`_).

Original author: `Johannes Schmidt <author_>`_
"""

from typing import List, Dict
from pathlib import Path

from pandas import DataFrame
from PIL import Image

from .transformations import ComposeDouble, Clip, normalize_01
from .transformations import FunctionWrapperDouble
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from waste_detection_system import shared_data as base


class WasteDetectionDataset(Dataset):
    """The Waste Detection Dataset for use with Pytorch Lightning
    """
    def __init__(self, data: DataFrame, mapping: Dict) -> None:
        """Parses the data from the dataset and performs sanity operations
        on the bounding boxes

        Args:
            data (DataFrame): dataset
            mapping (Dict): maps the class numbers to the string representations
        """
        self.transforms = ComposeDouble([
            Clip()
        ])
        self.mapping = mapping

        self.inputs : List[Path] = []
        self.targets : Dict[Path, List] = {}

        for img_path in data.path.unique():
            idx = base.ROOT / img_path
            self.inputs.append(idx)
            self.targets[idx] = data[data.path == img_path].apply(  # type: ignore
                lambda row: {
                    'label' : row['label'],
                    'bounding-box': [row['bbox-x'], row['bbox-y'],
                                    row['bbox-x']+row['bbox-w'], 
                                    row['bbox-y']+row['bbox-h']]
                }, 
                axis=1
            )
    
    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.inputs)
    
    def __getitem__(self, index : int) -> Dict:
        """
        Args:
            index (int): index

        Returns:
            Dict: item on the given index
        """
        input_path = Path(self.inputs[index])
        targets = self.targets[input_path]

        x = transforms.ToTensor()(Image.open(input_path).convert('RGB'))
        
        boxes = [row['bounding-box'] for row in targets]
        boxes = torch.tensor(boxes).to(torch.float32)

        labels = [self.mapping[row['label']] for row in targets]
        labels = torch.tensor(labels).to(torch.int64)

        y = {'boxes' : boxes, 'labels' : labels}

        return {'x' : x, 'y' : y, 'path' : input_path}