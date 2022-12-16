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

from typing import Callable, Dict, List, Optional, Union
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import Tensor, nn
import torch

from waste_detection_system.shared_data import AVAILABLE_CLASSIFIERS
from waste_detection_system.bbox_iou_evaluation import match_bboxes

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
            classifier_module = LinearSVC()
        elif classifier_type == AVAILABLE_CLASSIFIERS.KNN:
            classifier_module = KNeighborsClassifier()
        else: raise ValueError('Classifier type not supported')

        assert classifier_module is not None

        self.classifier = Pipeline([StandardScaler(), classifier_module])
    
    def forward(self, images : List[Tensor], targets : Optional[List[Dict[str, Tensor]]] = None
        ) -> Union[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (List[Dict[str, Tensor]]): the output from the model.
                During training, it returns the classifier loss.
                During testing, it returns a list of dictionaries (one for each image) that contains
                ``boxes``, ``labels``, ``scores``.

        Raises: 
            Exception: if the model is training but doesn't have targets

        """
        if self.training:
            assert targets is not None
            self.feature_extractor.eval()

            features = []
            for image, target in zip(images, targets):
                feature_extraction = self.feature_extractor(image)
                
                pred_boxes = feature_extraction['bounding_boxes'].detach().cpu().numpy()
                feature_map = feature_extraction['features'].detach().cpu().numpy()

                target_boxes = target['boxes'].detach().cpu().numpy()
                target_labels = target['labels'].detach().cpu().numpy()

                feature = {
                    'ground_truth_boxes' : [],
                    'ground_truth_labels' : [],
                    'prediction_boxes' : [],
                    'feature_map' : []
                }

                for gt, pred, _, valid in match_bboxes(target_boxes, pred_boxes):
                    if valid < 1:
                        continue
                    
                    feature['ground_truth_boxes'].append(target_boxes[gt])
                    feature['ground_truth_labels'].append(target_labels[gt])
                    feature['prediction_boxes'].append(pred_boxes[pred])
                    feature['feature_map'].append(feature_map[pred])
                
                features.append(feature)
            
            X = [feature['feature_map'] for feature in features]
            y = [feature['ground_truth_labels'] for feature in features]
            self.classifier = self.classifier.fit(X, y)
            y_hat_prob = self.classifier.predict_proba(X)

            return {'classification_loss' : torch.tensor(log_loss(y_true=y, y_pred=y_hat_prob))}
        
        else:
            self.feature_extractor.eval()

            results = []
            for image in images:
                feature_extraction = self.feature_extractor(image)
                
                pred_boxes = feature_extraction['bounding_boxes']
                feature_map = feature_extraction['features'].detach().cpu().numpy()

                y_hat_prob = self.classifier.predict_proba(feature_map)
                y_hat = self.classifier.predict(feature_map)

                results.append({
                    'boxes' : pred_boxes,
                    'scores' : torch.tensor(y_hat_prob),
                    'labels' : torch.tensor(y_hat)
                })

            return results
    

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