"""
Abstract base class for all detection models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List


class BaseDetectionModel(ABC, nn.Module):
    """
    Abstract base class for all detection models.
    
    This provides a common interface for different object detection architectures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detection model.
        
        Parameters
        ----------
        config : dict
            Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.num_classes = config.get('num_classes', 2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def forward(self, images: torch.Tensor, targets: Optional[List[Dict]] = None):
        """
        Forward pass of the model.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images
        targets : list of dict, optional
            List of target dictionaries (for training)
            
        Returns
        -------
        dict or list
            Model outputs (loss dict during training, detections during inference)
        """
        pass
    
    @abstractmethod
    def predict(self, images: torch.Tensor) -> List[Dict]:
        """
        Inference mode prediction.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images
            
        Returns
        -------
        list of dict
            List of predictions, one per image
        """
        pass
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Save model checkpoint.
        
        Parameters
        ----------
        path : str
            Path to save the checkpoint
        epoch : int
            Current epoch number
        optimizer : torch.optim.Optimizer
            Optimizer state
        metrics : dict
            Training/validation metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = False
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Parameters
        ----------
        path : str
            Path to the checkpoint file
        load_optimizer : bool
            Whether to load optimizer state
            
        Returns
        -------
        dict
            Checkpoint dictionary containing epoch, metrics, etc.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint
    
    def to_device(self, device: Optional[torch.device] = None) -> 'BaseDetectionModel':
        """
        Move model to specified device.
        
        Parameters
        ----------
        device : torch.device, optional
            Device to move model to. If None, uses self.device
            
        Returns
        -------
        BaseDetectionModel
            Self for method chaining
        """
        if device is not None:
            self.device = device
        self.to(self.device)
        return self
    
    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters.
        
        Returns
        -------
        int
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self) -> None:
        """
        Freeze backbone parameters (prevent training).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze_backbone(self) -> None:
        """
        Unfreeze backbone parameters (allow training).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
