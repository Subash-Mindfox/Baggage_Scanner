"""
Custom Dataset classes for baggage scanner X-ray images.
"""

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os


class BaggageXRayDataset(Dataset):
    """
    Dataset class for X-ray baggage scanner images with bounding box annotations.
    
    This dataset loads images lazily (on demand) to avoid memory issues.
    """
    
    def __init__(
        self,
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame,
        transforms: Optional[object] = None
    ):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        df_images : pd.DataFrame
            DataFrame with columns: filename, path, width, height, mode
        df_annotations : pd.DataFrame
            DataFrame with columns: filename, class, xmin, ymin, xmax, ymax
        transforms : callable, optional
            Optional transforms to apply to images and targets
        """
        self.df_images = df_images.reset_index(drop=True)
        self.df_annotations = df_annotations.reset_index(drop=True)
        self.transforms = transforms
        
        # Get unique image filenames
        self.image_filenames = self.df_images['filename'].unique().tolist()
        
        # Create a mapping for quick annotation lookup
        self._create_annotation_lookup()
    
    def _create_annotation_lookup(self):
        """Create a dictionary mapping filenames to their annotations."""
        self.annotations_dict = {}
        
        for filename in self.image_filenames:
            # Get all annotations for this image
            image_annotations = self.df_annotations[
                self.df_annotations['filename'] == filename
            ]
            
            boxes = []
            labels = []
            
            for _, row in image_annotations.iterrows():
                # Bounding box: [xmin, ymin, xmax, ymax]
                boxes.append([
                    float(row['xmin']),
                    float(row['ymin']),
                    float(row['xmax']),
                    float(row['ymax'])
                ])
                
                # Label (convert class name to integer if needed)
                # For now, assume 'class' column contains string labels
                # You may need to create a label mapping
                labels.append(self._class_to_label(row['class']))
            
            self.annotations_dict[filename] = {
                'boxes': boxes,
                'labels': labels
            }
    
    def _class_to_label(self, class_name: str) -> int:
        """
        Convert class name to integer label.
        
        Parameters
        ----------
        class_name : str
            Name of the class
            
        Returns
        -------
        int
            Integer label
        """
        # TODO: Implement proper class mapping
        # For now, return a dummy value
        # You should create a class_to_idx mapping based on your dataset
        
        # Example class mapping:
        class_mapping = {
            'gun': 1,
            'knife': 2,
            'weapon': 3,
            # Add more classes as needed
        }
        
        return class_mapping.get(class_name.lower(), 1)  # Default to 1 if unknown
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get an image and its annotations.
        
        Parameters
        ----------
        idx : int
            Index of the image
            
        Returns
        -------
        tuple
            (image_tensor, target_dict) where target_dict contains:
            - boxes: torch.Tensor of shape (N, 4)
            - labels: torch.Tensor of shape (N,)
            - image_id: torch.Tensor
        """
        # Get filename
        filename = self.image_filenames[idx]
        
        # Get image path
        image_row = self.df_images[self.df_images['filename'] == filename].iloc[0]
        image_path = image_row['path']
        
        # Load image
        image = read_image(image_path)
        
        # Get annotations
        annotations = self.annotations_dict[filename]
        boxes = torch.as_tensor(annotations['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annotations['labels'], dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        # Apply transforms if provided
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def get_image_info(self, idx: int) -> Dict:
        """
        Get metadata about an image.
        
        Parameters
        ----------
        idx : int
            Index of the image
            
        Returns
        -------
        dict
            Dictionary with image metadata
        """
        filename = self.image_filenames[idx]
        image_row = self.df_images[self.df_images['filename'] == filename].iloc[0]
        
        return {
            'filename': filename,
            'path': image_row['path'],
            'width': image_row['width'],
            'height': image_row['height'],
            'mode': image_row['mode'],
            'num_objects': len(self.annotations_dict[filename]['boxes'])
        }


class BaggageXRayDatasetSimple(Dataset):
    """
    Simplified dataset class that just returns images and filenames.
    Useful for inference or visualization.
    """
    
    def __init__(self, df_images: pd.DataFrame):
        """
        Initialize simple dataset.
        
        Parameters
        ----------
        df_images : pd.DataFrame
            DataFrame with image metadata
        """
        self.df_images = df_images.reset_index(drop=True)
    
    def __len__(self) -> int:
        """Return number of images."""
        return len(self.df_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get image and filename.
        
        Parameters
        ----------
        idx : int
            Index
            
        Returns
        -------
        tuple
            (image_tensor, filename)
        """
        row = self.df_images.iloc[idx]
        image = read_image(row['path'])
        filename = row['filename']
        
        return image, filename


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function for DataLoader.
    
    This is needed because images may have different sizes and
    different numbers of objects.
    
    Parameters
    ----------
    batch : list
        List of (image, target) tuples
        
    Returns
    -------
    tuple
        (list of images, list of targets)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


def create_dataloaders(
    df_train_images: pd.DataFrame,
    df_train_annotations: pd.DataFrame,
    df_val_images: Optional[pd.DataFrame] = None,
    df_val_annotations: Optional[pd.DataFrame] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    train_transforms: Optional[object] = None,
    val_transforms: Optional[object] = None
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create DataLoaders for training and validation.
    
    Parameters
    ----------
    df_train_images : pd.DataFrame
        Training images DataFrame
    df_train_annotations : pd.DataFrame
        Training annotations DataFrame
    df_val_images : pd.DataFrame, optional
        Validation images DataFrame
    df_val_annotations : pd.DataFrame, optional
        Validation annotations DataFrame
    batch_size : int
        Batch size
    num_workers : int
        Number of workers for data loading
    train_transforms : callable, optional
        Transforms for training data
    val_transforms : callable, optional
        Transforms for validation data
        
    Returns
    -------
    tuple
        (train_loader, val_loader)
    """
    # Create training dataset
    train_dataset = BaggageXRayDataset(
        df_train_images,
        df_train_annotations,
        transforms=train_transforms
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create validation dataset if provided
    val_loader = None
    if df_val_images is not None and df_val_annotations is not None:
        val_dataset = BaggageXRayDataset(
            df_val_images,
            df_val_annotations,
            transforms=val_transforms
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, val_loader
