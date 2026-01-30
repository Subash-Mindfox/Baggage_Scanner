"""
Faster R-CNN with Parallel Spatial Transformer Networks.

This model combines Faster R-CNN object detection with STNs that
learn to apply geometric transformations to improve detection accuracy.

Think of it as: Before detecting objects, the network learns to rotate/zoom/warp
the image in the way that makes objects easiest to detect.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from src.models.stn_transforms import DynamicSTN


class FasterRCNNWithParallelSTN(nn.Module):
    """
    Faster R-CNN with Parallel STN branches.
    
    Architecture:
    - Input Image
      ├→ STN (learns geometric transform)
      ├→ Transformed Image
      └→ Faster R-CNN (object detection)
    
    The STN and Faster R-CNN are trained together, so the STN learns
    transformations that specifically help the detector find objects.
    """
    
    def __init__(
        self,
        num_classes,
        anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        rpn_nms_thresh=0.7,
        backbone_name='resnet50',
        pretrained=True,
        stn_type='affine',
        device=None
    ):
        """
        Initialize Faster R-CNN with STN.
        
        Parameters
        ----------
        num_classes : int
            Number of object classes (including background)
        anchor_sizes : tuple
            Anchor box sizes for different feature map levels
        aspect_ratios : tuple
            Aspect ratios for anchor boxes
        rpn_nms_thresh : float
            NMS threshold for Region Proposal Network
        backbone_name : str
            Backbone architecture (e.g., 'resnet50')
        pretrained : bool
            Whether to use ImageNet pretrained weights
        stn_type : str
            Type of STN: 'affine', 'projective', or 'tps'
        device : torch.device
            Device to run model on
        """
        super(FasterRCNNWithParallelSTN, self).__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.stn_type = stn_type
        
        # Create Spatial Transformer Network
        self.stn = DynamicSTN(
            transformation_type=stn_type,
            control_pts=9,  # For TPS (3x3 grid)
            out_height=584,
            out_width=688
        ).to(self.device)
        
        # Create Faster R-CNN model
        self.faster_rcnn = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(self.device)
        
        # Configure anchor generator
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        self.faster_rcnn.rpn.anchor_generator = anchor_generator
        
        # Configure RPN NMS threshold
        self.faster_rcnn.rpn.nms_thresh = rpn_nms_thresh
    
    def forward(self, images, targets=None):
        """
        Forward pass through STN and Faster R-CNN.
        
        Parameters
        ----------
        images : list of torch.Tensor
            Input images, each of shape (3, H, W)
        targets : list of dict, optional
            Ground truth boxes and labels for training
            Each dict contains:
            - 'boxes': torch.Tensor of shape (N, 4)
            - 'labels': torch.Tensor of shape (N,)
            
        Returns
        -------
        dict or list
            During training: dict of losses
            During inference: list of predictions
        
        Process:
        1. Stack images into batch
        2. Apply STN to transform images
        3. Pass transformed images to Faster R-CNN
        4. Return detection results or losses
        """
        # Step 1: Stack images into a batch tensor
        images_stacked = torch.stack(images).to(self.device)
        
        # Step 2: Apply STN transformation
        # The STN learns how to transform the image to make detection easier
        transformed_images = self.stn(images_stacked)
        
        # Step 3: Convert back to list for Faster R-CNN
        transformed_images_list = [img for img in transformed_images]
        
        # Step 4: Pass through Faster R-CNN
        if self.training and targets is not None:
            # Training mode: return losses
            loss_dict = self.faster_rcnn(transformed_images_list, targets)
            return loss_dict
        else:
            # Inference mode: return predictions
            self.faster_rcnn.eval()
            with torch.no_grad():
                predictions = self.faster_rcnn(transformed_images_list)
            return predictions
    
    def get_transformed_images(self, images):
        """
        Get the transformed images without detection.
        
        Useful for visualizing what transformations the STN is learning.
        
        Parameters
        ----------
        images : list of torch.Tensor
            Input images
            
        Returns
        -------
        torch.Tensor
            Transformed images
        """
        images_stacked = torch.stack(images).to(self.device)
        with torch.no_grad():
            transformed_images = self.stn(images_stacked)
        return transformed_images


def create_model(
    num_classes,
    anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    rpn_nms_thresh=0.7,
    backbone_name='resnet50',
    pretrained=True,
    stn_type='affine',
    device=None
):
    """
    Factory function to create Faster R-CNN with parallel STNs.
    
    Parameters
    ----------
    num_classes : int
        Number of classes for detection (including background)
    anchor_sizes : tuple
        Sizes for anchor boxes
    aspect_ratios : tuple
        Aspect ratios for anchor boxes
    rpn_nms_thresh : float
        NMS threshold for Region Proposal Network
    backbone_name : str
        Name of the backbone model
    pretrained : bool
        Whether to use ImageNet pretrained weights
    stn_type : str
        Type of STN: 'affine', 'projective', or 'tps'
    device : torch.device
        Device to run model on
        
    Returns
    -------
    FasterRCNNWithParallelSTN
        Complete model with STN and detection
    
    Example
    -------
    >>> model = create_model(
    ...     num_classes=3,
    ...     stn_type='affine',
    ...     device=torch.device('cuda')
    ... )
    >>> # Model will learn affine transformations to improve detection
    """
    return FasterRCNNWithParallelSTN(
        num_classes=num_classes,
        anchor_sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
        rpn_nms_thresh=rpn_nms_thresh,
        backbone_name=backbone_name,
        pretrained=pretrained,
        stn_type=stn_type,
        device=device
    )
