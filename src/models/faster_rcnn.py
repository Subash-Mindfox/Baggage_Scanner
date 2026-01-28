"""
Faster R-CNN model implementation.
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, Any, Optional, List

from src.models.base_model import BaseDetectionModel


class FasterRCNNModel(BaseDetectionModel):
    """
    Faster R-CNN object detection model.
    
    This class wraps torchvision's Faster R-CNN implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Faster R-CNN model.
        
        Parameters
        ----------
        config : dict
            Model configuration dictionary
        """
        super().__init__(config)
        
        # Get model-specific config
        model_config = config.get('faster_rcnn', {})
        
        # Load pretrained model
        self.model = fasterrcnn_resnet50_fpn(
            pretrained=config.get('use_pretrained', True)
        )
        
        # Replace the classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            self.num_classes
        )
        
        # Store backbone reference for freezing/unfreezing
        self.backbone = self.model.backbone
        
        # Configure model parameters
        self._configure_model(model_config)
    
    def _configure_model(self, model_config: Dict[str, Any]) -> None:
        """
        Configure model parameters from config.
        
        Parameters
        ----------
        model_config : dict
            Faster R-CNN specific configuration
        """
        # Set image size constraints
        if 'min_size' in model_config:
            self.model.transform.min_size = (model_config['min_size'],)
        if 'max_size' in model_config:
            self.model.transform.max_size = model_config['max_size']
        
        # Configure RPN parameters
        if 'rpn_pre_nms_top_n_train' in model_config:
            self.model.rpn.pre_nms_top_n['training'] = model_config['rpn_pre_nms_top_n_train']
        if 'rpn_post_nms_top_n_train' in model_config:
            self.model.rpn.post_nms_top_n['training'] = model_config['rpn_post_nms_top_n_train']
        if 'rpn_nms_thresh' in model_config:
            self.model.rpn.nms_thresh = model_config['rpn_nms_thresh']
        
        # Configure detection parameters
        if 'box_detections_per_img' in model_config:
            self.model.roi_heads.detections_per_img = model_config['box_detections_per_img']
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Faster R-CNN.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images
        targets : list of dict, optional
            List of target dictionaries with 'boxes' and 'labels'
            
        Returns
        -------
        dict
            Loss dict during training, detections during inference
        """
        return self.model(images, targets)
    
    def predict(self, images: torch.Tensor) -> List[Dict]:
        """
        Run inference on images.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images
            
        Returns
        -------
        list of dict
            Predictions with 'boxes', 'labels', and 'scores'
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        return predictions
    
    def set_score_threshold(self, threshold: float) -> None:
        """
        Set the confidence score threshold for predictions.
        
        Parameters
        ----------
        threshold : float
            Confidence threshold (0-1)
        """
        self.model.roi_heads.score_thresh = threshold
    
    def set_nms_threshold(self, threshold: float) -> None:
        """
        Set the NMS IoU threshold.
        
        Parameters
        ----------
        threshold : float
            NMS IoU threshold (0-1)
        """
        self.model.roi_heads.nms_thresh = threshold
