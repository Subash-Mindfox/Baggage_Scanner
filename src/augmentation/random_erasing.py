"""
Random Erasing augmentation implementation.
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


@AugmentationRegistry.register('random_erasing')
class RandomErasingAugmentor(BaseAugmentor):
    """
    Apply Random Erasing augmentation to images.
    
    This augmentation randomly erases rectangular regions while ensuring
    that no more than a specified percentage of any object is erased.
    """
    
    def _get_suffix(self) -> str:
        """Return filename suffix for Random Erasing augmentation."""
        return '_erase.jpg'
    
    def apply_transform(self, image_tensor, annotations, **kwargs):
        """
        Apply Random Erasing transformation.
        
        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor
        annotations : pd.DataFrame
            Annotations with bounding box coordinates
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        torch.Tensor
            Image tensor with random erasing applied
        """
        # Get parameters from config
        sl = self.params.get('sl', 0.02)  # Min proportion of erased area
        sh = self.params.get('sh', 0.3)   # Max proportion of erased area
        r1 = self.params.get('r1', 0.3)   # Min aspect ratio
        max_erasing_area_per_bbox = self.params.get('max_erasing_area_per_bbox', 0.7)
        attempts = self.params.get('attempts', 100)
        
        return self._apply_random_erasing(
            image_tensor,
            annotations,
            sl=sl,
            sh=sh,
            r1=r1,
            max_erasing_area_per_bbox=max_erasing_area_per_bbox,
            attempts=attempts
        )
    
    def _apply_random_erasing(
        self,
        img_tensor,
        annotations,
        sl=0.02,
        sh=0.3,
        r1=0.3,
        max_erasing_area_per_bbox=0.7,
        attempts=100
    ):
        """
        Apply Random Erasing with constraints on bounding box overlap.
        
        Parameters
        ----------
        img_tensor : torch.Tensor
            Image tensor of shape (C, H, W)
        annotations : pd.DataFrame
            DataFrame containing annotations
        sl : float
            Minimum proportion of erased area against input image
        sh : float
            Maximum proportion of erased area against input image
        r1 : float
            Minimum aspect ratio of erased area
        max_erasing_area_per_bbox : float
            Maximum proportion of any bounding box that can be erased
        attempts : int
            Number of attempts to find a valid erasing rectangle
            
        Returns
        -------
        torch.Tensor
            Image tensor with random erasing applied
        """
        # Clone to avoid modifying original
        img_tensor = img_tensor.clone()
        
        # Convert to float32 if needed
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float()
        
        # Get image dimensions
        C, H, W = img_tensor.shape
        
        # Extract bounding boxes from annotations
        bboxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, W)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, H)
        
        # Compute areas of bounding boxes
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        
        for attempt in range(attempts):
            # Area of the image
            area = H * W
            
            # Target area for erasing
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)
            
            # Compute dimensions of the erasing rectangle
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            
            if h < H and w < W:
                # Randomly choose top-left corner
                x1 = random.randint(0, W - w)
                y1 = random.randint(0, H - h)
                x2 = x1 + w
                y2 = y1 + h
                
                # Check overlap with each bounding box
                erase_rect = np.array([x1, y1, x2, y2])
                exceeds_threshold = False
                
                for bbox, bbox_area in zip(bboxes, bbox_areas):
                    # Compute intersection
                    xi1 = max(erase_rect[0], bbox[0])
                    yi1 = max(erase_rect[1], bbox[1])
                    xi2 = min(erase_rect[2], bbox[2])
                    yi2 = min(erase_rect[3], bbox[3])
                    
                    inter_width = max(0, xi2 - xi1)
                    inter_height = max(0, yi2 - yi1)
                    intersection_area = inter_width * inter_height
                    
                    # Compute overlap ratio
                    overlap_ratio = intersection_area / bbox_area if bbox_area > 0 else 0
                    
                    if overlap_ratio > max_erasing_area_per_bbox:
                        exceeds_threshold = True
                        break
                
                if not exceeds_threshold:
                    # Replace pixels with random values
                    img_tensor[:, y1:y2, x1:x2] = torch.randn(
                        (C, y2 - y1, x2 - x1),
                        dtype=img_tensor.dtype,
                        device=img_tensor.device
                    ) * img_tensor.std() + img_tensor.mean()
                    
                    # Clamp values
                    img_tensor = torch.clamp(img_tensor, 0, 255)
                    img_tensor = img_tensor.type(torch.uint8)
                    
                    return img_tensor
        
        # If no valid rectangle found, return original
        print(f"No valid erasing found after {attempts} attempts.")
        return img_tensor.type(torch.uint8)
    
    def adjust_annotations(
        self,
        annotations: pd.DataFrame,
        img_width: int,
        img_height: int,
        **kwargs
    ) -> pd.DataFrame:
        """
        Random Erasing doesn't change bounding boxes, just copy annotations.
        
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
        return annotations.copy()
    
    def _perform_augmentation(
        self,
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame,
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform Random Erasing augmentation on selected images.
        
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
        print(f"Starting Random Erasing augmentation...")
        
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
            
            print(f"Random Erasing augmentation completed. Added {len(augmented_images)} images.")
        else:
            print("No images were augmented.")
        
        return df_images, df_annotations
