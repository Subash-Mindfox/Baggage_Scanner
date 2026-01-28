"""
Flip augmentation implementation.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple
import torchvision.transforms.functional as TF

from src.augmentation.base_augmentor import BaseAugmentor
from src.augmentation.registry import AugmentationRegistry
from src.common.image_utils import save_tensor_as_image, load_image_as_tensor
from src.common.bbox_utils import transform_bbox_after_flip


@AugmentationRegistry.register('flip')
class FlipAugmentor(BaseAugmentor):
    """
    Apply flip augmentations to images (horizontal, vertical, both).
    """
    
    def _get_suffix(self) -> str:
        """Return filename suffix for flip augmentation."""
        # This will be overridden based on flip type
        return '_flip.jpg'
    
    def apply_transform(self, image_tensor, flip_type='hflip', **kwargs):
        """
        Apply flip transformation.
        
        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor
        flip_type : str
            Type of flip: 'hflip', 'vflip', or 'hvflip'
            
        Returns
        -------
        torch.Tensor
            Flipped image tensor
        """
        if flip_type == 'hflip':
            return TF.hflip(image_tensor)
        elif flip_type == 'vflip':
            return TF.vflip(image_tensor)
        elif flip_type == 'hvflip':
            return TF.hflip(TF.vflip(image_tensor))
        else:
            raise ValueError(f"Unknown flip type: {flip_type}")
    
    def adjust_annotations(
        self,
        annotations: pd.DataFrame,
        img_width: int,
        img_height: int,
        flip_type: str = 'hflip',
        **kwargs
    ) -> pd.DataFrame:
        """
        Adjust bounding boxes after flip.
        
        Parameters
        ----------
        annotations : pd.DataFrame
            Original annotations
        img_width : int
            Image width
        img_height : int
            Image height
        flip_type : str
            Type of flip applied
            
        Returns
        -------
        pd.DataFrame
            Adjusted annotations
        """
        adjusted_annotations = []
        
        for _, row in annotations.iterrows():
            xmin, ymin = row['xmin'], row['ymin']
            xmax, ymax = row['xmax'], row['ymax']
            
            # Transform bbox
            new_bbox = transform_bbox_after_flip(
                [xmin, ymin, xmax, ymax],
                flip_type,
                img_width,
                img_height
            )
            
            # Create new row
            new_row = row.copy()
            new_row['xmin'] = new_bbox[0]
            new_row['ymin'] = new_bbox[1]
            new_row['xmax'] = new_bbox[2]
            new_row['ymax'] = new_bbox[3]
            
            adjusted_annotations.append(new_row)
        
        return pd.DataFrame(adjusted_annotations)
    
    def _perform_augmentation(
        self,
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame,
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform flip augmentation on selected images.
        
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
        print(f"Starting Flip augmentation...")
        
        # Get flip types from config
        flip_types = self.params.get('types', ['hflip', 'vflip', 'hvflip'])
        
        # Select random samples
        selected_images_df = self._select_random_samples(df_images)
        
        augmented_images = []
        augmented_annotations = []
        
        for idx, row in selected_images_df.iterrows():
            original_filename = row['filename']
            image_path = row['path']
            img_width = row['width']
            img_height = row['height']
            
            # Load image tensor
            tensor = load_image_as_tensor(image_path)
            
            # Apply each flip type
            for flip_type in flip_types:
                # Apply transformation
                flipped_tensor = self.apply_transform(tensor, flip_type=flip_type)
                
                # Create new filename
                new_filename = original_filename.replace('.jpg', f'_{flip_type}.jpg')
                
                # Save augmented image
                output_path = os.path.join(output_dir, new_filename)
                shape, mode = save_tensor_as_image(flipped_tensor, output_path)
                
                # Create new image row
                new_row = row.copy()
                new_row['filename'] = new_filename
                new_row['path'] = output_path
                new_row['width'] = shape[0]
                new_row['height'] = shape[1]
                new_row['mode'] = mode
                augmented_images.append(new_row)
                
                # Adjust annotations
                annotations = df_annotations[
                    df_annotations['filename'] == original_filename
                ].copy()
                
                adjusted_annotations = self.adjust_annotations(
                    annotations,
                    img_width,
                    img_height,
                    flip_type=flip_type
                )
                adjusted_annotations['filename'] = new_filename
                augmented_annotations.append(adjusted_annotations)
        
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
            
            print(f"Flip augmentation completed. Added {len(augmented_images)} images.")
        else:
            print("No images were augmented.")
        
        return df_images, df_annotations
