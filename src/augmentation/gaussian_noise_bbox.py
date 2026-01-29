"""
Gaussian Noise to Bounding Boxes augmentation implementation.
"""

import pandas as pd
import numpy as np
import os
import torch
import random
from typing import Tuple

from src.augmentation.base_augmentor import BaseAugmentor
from src.augmentation.registry import AugmentationRegistry
from src.common.image_utils import save_tensor_as_image, load_image_as_tensor


@AugmentationRegistry.register('gaussian_noise_bbox')
class GaussianNoiseBBoxAugmentor(BaseAugmentor):
    """
    Apply Gaussian Noise only within bounding boxes.
    
    This augmentation adds Gaussian noise only to the regions inside
    bounding boxes, leaving the rest of the image unchanged.
    """
    
    def _get_suffix(self) -> str:
        """Return filename suffix for Gaussian Noise BBox augmentation."""
        return '_noisebb.jpg'
    
    def apply_transform(self, image_tensor, annotations, **kwargs):
        """
        Apply Gaussian Noise to bounding box regions.
        
        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor
        annotations : pd.DataFrame
            Annotations with bounding box coordinates
        **kwargs : dict
            Additional parameters (mean, sigma)
            
        Returns
        -------
        torch.Tensor
            Image tensor with noise added to bounding boxes
        """
        # Clone to avoid modifying original
        img_tensor = image_tensor.clone()
        
        # Get mean and sigma ranges from config
        mean_range = self.params.get('mean_range', [0.0, 1.0])
        sigma_range = self.params.get('sigma_range', [5.0, 15.0])
        
        # Randomly select mean and sigma
        mean = random.uniform(*mean_range)
        sigma = random.uniform(*sigma_range)
        
        # Add noise to each bounding box
        for _, ann_row in annotations.iterrows():
            xmin = int(ann_row['xmin'])
            xmax = int(ann_row['xmax'])
            ymin = int(ann_row['ymin'])
            ymax = int(ann_row['ymax'])
            
            # Ensure coordinates are within image dimensions
            xmin = max(0, xmin)
            xmax = min(img_tensor.shape[2] - 1, xmax)
            ymin = max(0, ymin)
            ymax = min(img_tensor.shape[1] - 1, ymax)
            
            # Generate noise for the bounding box area
            noise = torch.randn(img_tensor[:, ymin:ymax+1, xmin:xmax+1].size()) * sigma + mean
            noise = noise.to(img_tensor.dtype)
            
            # Add noise to the bounding box area
            img_tensor[:, ymin:ymax+1, xmin:xmax+1] += noise
            
            # Clamp the values to valid pixel range
            img_tensor = torch.clamp(img_tensor, 0, 255)
        
        return img_tensor
    
    def adjust_annotations(
        self,
        annotations: pd.DataFrame,
        img_width: int,
        img_height: int,
        **kwargs
    ) -> pd.DataFrame:
        """
        Gaussian Noise to BBox doesn't change bounding boxes, just copy annotations.
        
        Parameters
        ----------
        annotations : pd.DataFrame
            Original annotations
        img_width : int
            Image width (unused)
        img_height : int
            Image height (unused)
            
        Returns
        -------
        pd.DataFrame
            Annotations (unchanged)
        """
        # This augmentation doesn't affect bounding boxes
        return annotations.copy()
    
    def _perform_augmentation(
        self,
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame,
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform Gaussian Noise BBox augmentation on selected images.
        
        Parameters
        ----------
        df_images : pd.DataFrame
            DataFrame containing image metadata
        df_annotations : pd.DataFrame
            DataFrame containing annotations
        output_dir : str
            Directory to save augmented images
            
        Returns
        -------
        tuple
            (updated df_images, updated df_annotations)
        """
        print(f"Starting Gaussian Noise BBox augmentation...")
        
        # Set random seed if provided
        if self.random_seed is not None:
            random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
        
        # Select random samples
        selected_images_df = self._select_random_samples(df_images)
        
        augmented_images = []
        augmented_annotations = []
        
        for idx, row in selected_images_df.iterrows():
            original_filename = row['filename']
            image_path = row['path']
            
            try:
                # Load image tensor
                tensor = load_image_as_tensor(image_path)
                
                # Get annotations for this image
                annotations = df_annotations[
                    df_annotations['filename'] == original_filename
                ]
                
                if len(annotations) == 0:
                    print(f"No annotations found for {original_filename}. Skipping.")
                    continue
                
                # Apply transformation
                transformed_tensor = self.apply_transform(tensor, annotations)
                
                # Create new filename
                new_filename = original_filename.replace('.jpg', self.suffix)
                
                # Check if augmented image already exists
                if new_filename in df_images['filename'].values:
                    print(f"Augmented image {new_filename} already exists. Skipping.")
                    continue
                
                # Save augmented image
                output_path = os.path.join(output_dir, new_filename)
                shape, mode = save_tensor_as_image(transformed_tensor, output_path)
                
                # Create new image row
                new_row = row.copy()
                new_row['filename'] = new_filename
                new_row['path'] = output_path
                new_row['width'] = shape[0]
                new_row['height'] = shape[1]
                new_row['mode'] = mode
                augmented_images.append(new_row)
                
                # Copy annotations with new filename
                new_annotations = annotations.copy()
                new_annotations['filename'] = new_filename
                augmented_annotations.append(new_annotations)
                
            except Exception as e:
                print(f"Error processing image {original_filename}: {e}")
                continue
        
        # Convert to DataFrames
        if augmented_images:
            augmented_images_df = pd.DataFrame(augmented_images)
            augmented_annotations_df = pd.concat(
                augmented_annotations,
                ignore_index=True
            )
            
            # Add to original DataFrames
            df_images = pd.concat(
                [df_images, augmented_images_df],
                ignore_index=True
            )
            df_annotations = pd.concat(
                [df_annotations, augmented_annotations_df],
                ignore_index=True
            )
            
            print(f"Gaussian Noise BBox augmentation completed. Added {len(augmented_images)} images.")
        else:
            print("No images were augmented.")
        
        return df_images, df_annotations
