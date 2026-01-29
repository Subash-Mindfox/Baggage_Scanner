"""
Gaussian Noise augmentation implementation.
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


@AugmentationRegistry.register('gaussian_noise')
class GaussianNoiseAugmentor(BaseAugmentor):
    """
    Apply Gaussian Noise augmentation to entire images.
    
    This augmentation adds Gaussian noise to the entire image with random
    mean and sigma values.
    """
    
    def _get_suffix(self) -> str:
        """Return filename suffix for Gaussian Noise augmentation."""
        return '_noise.jpg'
    
    def apply_transform(self, image_tensor, **kwargs):
        """
        Apply Gaussian Noise transformation.
        
        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor
        **kwargs : dict
            Additional parameters (mean, sigma)
            
        Returns
        -------
        torch.Tensor
            Noisy image tensor
        """
        # Get mean and sigma ranges from config
        mean_range = self.params.get('mean_range', [0.5, 1.5])
        sigma_range = self.params.get('sigma_range', [20.0, 70.0])
        
        # Randomly select mean and sigma
        mean = random.uniform(*mean_range)
        sigma = random.uniform(*sigma_range)
        
        # Generate noise
        noise = torch.randn(image_tensor.size(), device=image_tensor.device) * sigma + mean
        
        # Add noise to image
        noised_tensor = image_tensor.float() + noise
        noised_tensor = torch.clamp(noised_tensor, 0, 255).type(image_tensor.dtype)
        
        return noised_tensor
    
    def adjust_annotations(
        self,
        annotations: pd.DataFrame,
        img_width: int,
        img_height: int,
        **kwargs
    ) -> pd.DataFrame:
        """
        Gaussian Noise doesn't change bounding boxes, just copy annotations.
        
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
            Annotations (unchanged for Gaussian Noise)
        """
        # Gaussian Noise doesn't affect bounding boxes
        return annotations.copy()
    
    def _perform_augmentation(
        self,
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame,
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform Gaussian Noise augmentation on selected images.
        
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
        print(f"Starting Gaussian Noise augmentation...")
        
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
                
                # Apply transformation
                transformed_tensor = self.apply_transform(tensor)
                
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
                annotations = df_annotations[
                    df_annotations['filename'] == original_filename
                ].copy()
                annotations['filename'] = new_filename
                augmented_annotations.append(annotations)
                
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
            
            print(f"Gaussian Noise augmentation completed. Added {len(augmented_images)} images.")
        else:
            print("No images were augmented.")
        
        return df_images, df_annotations
