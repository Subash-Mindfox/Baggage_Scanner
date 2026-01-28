"""
ColorJitter augmentation implementation.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple
from torchvision.transforms import ColorJitter as TorchColorJitter

from src.augmentation.base_augmentor import BaseAugmentor
from src.augmentation.registry import AugmentationRegistry
from src.common.image_utils import save_tensor_as_image, load_image_as_tensor


@AugmentationRegistry.register('color_jitter')
class ColorJitterAugmentor(BaseAugmentor):
    """
    Apply ColorJitter augmentation to images.
    
    This augmentation randomly changes brightness, contrast, saturation, and hue.
    """
    
    def _get_suffix(self) -> str:
        """Return filename suffix for ColorJitter augmentation."""
        return '_jit.jpg'
    
    def apply_transform(self, image_tensor, **kwargs):
        """
        Apply ColorJitter transformation.
        
        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor
            
        Returns
        -------
        torch.Tensor
            Transformed image tensor
        """
        # Create ColorJitter transform
        color_jitter = TorchColorJitter(
            brightness=tuple(self.params.get('brightness', [0.8, 2.0])),
            contrast=tuple(self.params.get('contrast', [0.8, 2.0])),
            saturation=tuple(self.params.get('saturation', [0.8, 2.0])),
            hue=tuple(self.params.get('hue', [-0.1, 0.1]))
        )
        
        return color_jitter(image_tensor)
    
    def adjust_annotations(
        self,
        annotations: pd.DataFrame,
        img_width: int,
        img_height: int,
        **kwargs
    ) -> pd.DataFrame:
        """
        ColorJitter doesn't change bounding boxes, just copy annotations.
        
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
            Annotations (unchanged for ColorJitter)
        """
        # ColorJitter doesn't affect bounding boxes
        return annotations.copy()
    
    def _perform_augmentation(
        self,
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame,
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform ColorJitter augmentation on selected images.
        
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
        print(f"Starting ColorJitter augmentation...")
        
        # Select random samples
        selected_images_df = self._select_random_samples(df_images)
        
        augmented_images = []
        augmented_annotations = []
        
        for idx, row in selected_images_df.iterrows():
            original_filename = row['filename']
            image_path = row['path']
            
            # Load image tensor
            tensor = load_image_as_tensor(image_path)
            
            # Apply transformation
            transformed_tensor = self.apply_transform(tensor)
            
            # Create new filename
            new_filename = original_filename.replace('.jpg', self.suffix)
            
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
            
            print(f"ColorJitter augmentation completed. Added {len(augmented_images)} images.")
        else:
            print("No images were augmented.")
        
        return df_images, df_annotations
