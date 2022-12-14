# -*- coding: utf-8 -*-

from typing import Callable, Dict, List, Optional, Tuple, Union
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
    def __init__(self, model : nn.Module, layer: str) -> None:
        super().__init__()
        self.model = model
        self.layer_name = layer
        self.bounding_boxes = []
        self.features = []

        self.layer = dict([*self.model.named_modules()])[layer]
        self.layer.register_forward_hook(self.save_features())
    
    def save_features(self) -> Callable:
        def fn(_, input : List[Tensor], output : Tensor):
            self.bounding_boxes = output.tolist()
            self.features = input
        return fn
    
    def forward(self, x : Tensor) -> Dict[str, List[Tensor]]:
        _ = self.model(x)
        return {
            'bounding_boxes' : self.bounding_boxes,
            'features' : self.features
        }





class HybridDLModel(nn.Module):
    def __init__(self, feature_extractor : FeatureExtractor, 
                classifier_type : AVAILABLE_CLASSIFIERS) -> None:
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
                'boxes', 'labels', 'scores'.

        Raises:

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