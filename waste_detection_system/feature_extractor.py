# -*- coding: utf-8 -*-

"""
.. note::
    For the Hybrid Deep Learning models, the system requires that the user 
    previously trained the object detection module and applied the 
    corresponding weights.

    The Hybrid Deep Learning model will *not* train the object detection module;
    insted, when training the Machine Learning classifier the object detection
    module will be put in ``eval()`` mode.
"""

from typing import Callable, Dict, List, Union
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted
from torch import Tensor, nn
from torch.utils.data import DataLoader
import numpy as np
import torch

from waste_detection_system.shared_data import AVAILABLE_CLASSIFIERS
from waste_detection_system.bbox_iou_evaluation import match_bboxes
from waste_detection_system.waste_detection_dataset import WasteDetectionDataset

class FeatureExtractor(nn.Module):
    """
    Intermediate module that transforms the object detection model into
    a feature extractor that returns feature maps and bounding boxes.
    """
    def __init__(self, model : nn.Module, layer: str) -> None:
        """
        Args:
            model (nn.Module): object detection module (see :ref:`here <object_detection>_` 
                                for the list of allowed models)
            layer (str): name of the bounding box regression head
        """
        super().__init__()
        self.model = model
        self.layer_name = layer
        self.bounding_boxes = []
        self.features = []

        self.layer = dict([*self.model.named_modules()])[layer]
        self.layer.register_forward_hook(self.save_features())
    
    def save_features(self) -> Callable:
        """Hook that takes the inputs and outputs of a layer and
        registers them in the module

        Returns:
            Callable: a forward hook for the given layer
        
        :meta private:
        """
        def fn(_, input : List[Tensor], output : Tensor):
            self.bounding_boxes = output.tolist()
            self.features = input
        return fn
    
    def forward(self, x : Tensor) -> Dict[str, List[Tensor]]:
        """Applies the input tensor to the model and retrieves the
        results from the forward hook

        Args:
            x (Tensor): input tensor

        Returns:
            Dict[str, List[Tensor]]: dictionary with ``bounding_boxes`` and 
                                    ``features`` keys
        """
        _ = self.model(x)
        return {
            'bounding_boxes' : self.bounding_boxes,
            'features' : self.features
        }




class HybridDLModel(nn.Module):
    """
    Hybrid Deep Learning/traditional Machine Learning model that uses the
    Deep Learning object detector model to obtain the bounding boxes and
    the feature maps associated, and a traditional Machine Learning 
    classifier algorithm to predict the bounding box class given its 
    feature map.
    """
    def __init__(self, feature_extractor : FeatureExtractor, 
                classifier_type : AVAILABLE_CLASSIFIERS) -> None:
        """
        Args:
            feature_extractor (FeatureExtractor): feature extractor
            classifier_type (AVAILABLE_CLASSIFIERS): type of traditional ML
                                                    classifier

        Raises:
            ValueError: if the classifier type is not supported.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        if classifier_type == AVAILABLE_CLASSIFIERS.SVM:
            self.classifier = Pipeline([('scaler', StandardScaler()), ('svc', LinearSVC())])
        elif classifier_type == AVAILABLE_CLASSIFIERS.KNN:
            self.classifier = Pipeline([('scaler', StandardScaler()), ('knnc', KNeighborsClassifier())])
        else: raise ValueError('Classifier type not supported')

        assert self.classifier is not None

    
    def forward(self, train_loader : Union[DataLoader[WasteDetectionDataset], None] = None,
                val_loader : Union[DataLoader[WasteDetectionDataset], None] = None
        ) -> Union[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        """
        Args:
            train_loader (Union[DataLoader[WasteDetectionDataset], None]): dataloader for
                training. It won't be used if the model is not in training mode. Defaults to ``None``
            val_loader (Union[DataLoader[WasteDetectionDataset], None]): dataloader for
                validating. It won't be used if the model is in training mode. Defaultos to ``None``

        Returns:
            result (List[Dict[str, Tensor]]): the output from the model.
                During training, it returns the classifier loss.
                During testing, it returns a list of dictionaries (one for each image) that contains
                ``boxes``, ``labels``, ``scores``.

        Raises: 
            Exception: if the model is training but train_loader is None or the model is not training
                        but val_loader is None.
                        The function will also raise an exception if the feature extractor can't make
                        any valid predictions.
            NotFittedError: if the model is in eval() mode but hasn't been fitted beforehand.

        """
        if self.training and train_loader is not None:
            self.feature_extractor.eval()

            features = []
            print('Extracting features...', end='')
            for image, target, _ in train_loader:
                feature_extraction = self.feature_extractor(image)
                
                pred_boxes = np.asarray(feature_extraction['bounding_boxes']).squeeze()
                feature_map = feature_extraction['features'][0][0].detach().cpu().numpy().squeeze()

                target_boxes = np.asarray([t['boxes'].detach().cpu().numpy() for t in target]).squeeze()
                target_labels = np.asarray([t['labels'].detach().cpu().numpy() for t in target]).squeeze()

                feature = {
                    'ground_truth_boxes' : [],
                    'ground_truth_labels' : [],
                    'prediction_boxes' : [],
                    'feature_map' : []
                }

                if pred_boxes.ndim != 2 or target_boxes.ndim != 2:
                    continue

                for result in match_bboxes(target_boxes, pred_boxes):
                    if len(result) == 4:
                        gt, pred, _, valid = result
                    else: continue
                    if valid < 1:
                        continue
                    
                    feature['ground_truth_boxes'].append(target_boxes[gt])
                    feature['ground_truth_labels'].append(target_labels[gt])
                    feature['prediction_boxes'].append(pred_boxes[pred])
                    feature['feature_map'].append(feature_map[pred])
                
                features.append(feature)
            
            print(' Done!')
            if not features:
                print('Couldn\'t find any predictions.')
                raise Exception('Cant\'t find any predictions')

            X = [feature['feature_map'] for feature in features if feature['feature_map']]
            y = [feature['ground_truth_labels'] for feature in features if feature['ground_truth_labels']]
            if not X or not y:
                print('Couldn\'t find any predictions.')
                raise Exception('Cant\'t find any predictions')
            
            print('Classifying...', end='')
            self.classifier = self.classifier.fit(X, y)
            y_hat_prob = self.classifier.predict_proba(X)
            print(' Done')

            return {'classification_loss' : torch.tensor(log_loss(y_true=y, y_pred=y_hat_prob))}
        
        elif not self.training and val_loader is not None:
            self.feature_extractor.eval()

            results = []
            for image, _, _ in val_loader:
                feature_extraction = self.feature_extractor(image)
                
                pred_boxes = np.asarray(feature_extraction['bounding_boxes']).squeeze()
                feature_map = feature_extraction['features'][0][0].detach().cpu().numpy().squeeze()

                if pred_boxes.ndim != 2:
                    continue
                check_is_fitted(self.classifier)
                
                y_hat_prob = self.classifier.predict_proba(feature_map)
                y_hat = self.classifier.predict(feature_map)

                results.append({
                    'boxes' : pred_boxes,
                    'scores' : torch.tensor(y_hat_prob),
                    'labels' : torch.tensor(y_hat)
                })

            return results
        
        raise Exception(f'Incorrect forward call, training={self.training}, '
                        f'train_loader={train_loader is not None} and '
                        f'val_loader={val_loader is not None}')
    

    def validate(self, x : List[Dict[str, Tensor]], y : List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Custom validation step to be used in conjunction with the model in ``eval()`` mode

        Args:
            x (List[Dict[str, Tensor]]): results from the ``forward()`` function in ``eval()`` mode
            y (List[Dict[str, Tensor]]): targets, as passed to the ``forward()`` function

        Returns:
            Dict[str, Tensor]: dictionary with only one key ``classification_loss`` that holds the
                                classification loss
        """
        features = []
        for image, target in zip(x, y):
            pred_boxes = image['boxes'].detach().cpu().numpy()
            feature_map = image['labels'].detach().cpu().numpy()

            target_boxes = target['boxes'].detach().cpu().numpy()
            target_labels = target['labels'].detach().cpu().numpy()

            feature = {
                'ground_truth_boxes' : [],
                'ground_truth_labels' : [],
                'prediction_boxes' : [],
                'feature_map' : []
            }

            for gt, pred, iou, valid in match_bboxes(target_boxes, pred_boxes):
                if valid < 1:
                    continue
                
                feature['ground_truth_boxes'].append(target_boxes[gt])
                feature['ground_truth_labels'].append(target_labels[gt])
                feature['prediction_boxes'].append(pred_boxes[pred])
                feature['feature_map'].append(feature_map[pred])
            
            features.append(feature)
        
        X = [feature['feature_map'] for feature in features]
        y = [feature['ground_truth_labels'] for feature in features]
        y_hat_prob = self.classifier.predict_proba(X)

        return {'classification_loss' : torch.tensor(log_loss(y_true=y, y_pred=y_hat_prob))}