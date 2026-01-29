"""
Random Blocking augmentation implementation.

This augmentation masks random patches of the image with mean color values
and applies Gaussian blur to blend the patch seamlessly.
"""

import pandas as pd
import numpy as np
import os
import torch
import random
import cv2
from typing import Tuple

from src.augmentation.base_augmentor import BaseAugmentor
from src.augmentation.registry import AugmentationRegistry
from src.common.image_utils import save_tensor_as_image, load_image_as_tensor, torch_to_np


@AugmentationRegistry.register('random_blocking')
class RandomBlockingAugmentor(BaseAugmentor):
    """
    Apply Random Blocking augmentation to images.
    
    This augmentation creates random rectangular patches filled with the image's
    mean color values and blended with Gaussian blur. The patch size is constrained
    to not exceed the smallest bounding box in the image.
    """
    
    def _get_suffix(self) -> str:
        """Return filename suffix for Random Blocking augmentation."""
        return '_block.jpg'
    
    def calculate_channel_means(self, image_tensor):
        """
        Calculate mean values for each color channel.
        
        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor
            
        Returns
        -------
        tuple
            (mean_r, mean_g, mean_b)
        """
        image_tensor = image_tensor.float()
        mean_r = torch.mean(image_tensor[0, :, :]).item()
        mean_g = torch.mean(image_tensor[1, :, :]).item()
        mean_b = torch.mean(image_tensor[2, :, :]).item()
        
        return mean_r, mean_g, mean_b
    
    def hide_patch(
        self,
        image_tensor,
        mean_r,
        mean_g,
        mean_b,
        probability=1.0,
        max_patch=0.5
    ):
        """
        Create a masked patch on the image filled with mean values.
        
        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor
        mean_r : float
            Mean value for red channel
        mean_g : float
            Mean value for green channel
        mean_b : float
            Mean value for blue channel
        probability : float
            Probability of applying the mask (0-1)
        max_patch : float
            Maximum proportion of image to mask
            
        Returns
        -------
        tuple
            (masked_image_tensor, final_image_numpy)
        """
        # Clone original image
        masked_image_tensor = image_tensor.clone()
        
        # Determine whether to apply the mask
        apply_mask = np.random.rand() <= probability
        
        if apply_mask:
            # Get image dimensions
            height, width = image_tensor.shape[1], image_tensor.shape[2]
            
            # Calculate random mask size (between 0.1% and max_patch)
            mask_size = int(height * width * np.random.uniform(0.001, max_patch))
            mask_side = int(np.sqrt(mask_size))
            
            # Ensure mask doesn't exceed image dimensions
            mask_side = min(mask_side, height, width)
            
            # Random starting coordinates
            start_y = np.random.randint(0, height - mask_side + 1)
            start_x = np.random.randint(0, width - mask_side + 1)
            
            # Create tensor with mean values
            mean_value_tensor = torch.tensor(
                [mean_r, mean_g, mean_b],
                dtype=torch.float32
            ).view(3, 1, 1)
            
            # Fill the masking region with mean values
            masked_image_tensor[
                :,
                start_y:start_y + mask_side,
                start_x:start_x + mask_side
            ] = mean_value_tensor
            
            # Convert to numpy for OpenCV processing
            masked_image_np = masked_image_tensor.permute(1, 2, 0).numpy()
            
            # Create mask for the region that needs smoothing
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[start_y:start_y + mask_side, start_x:start_x + mask_side] = 255
            
            # Apply Gaussian blur to the masked region
            blurred_mask = cv2.GaussianBlur(masked_image_np, (15, 15), 0)
            
            # Blend the original image with the blurred masked image
            final_image = np.where(
                mask[:, :, np.newaxis] == 255,
                blurred_mask,
                masked_image_np
            )
        else:
            # If no mask applied, return original image
            final_image = image_tensor.permute(1, 2, 0).numpy()
        
        return masked_image_tensor, final_image
    
    def apply_transform(self, image_tensor, annotations, **kwargs):
        """
        Apply Random Blocking transformation.
        
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
            Image tensor with random blocking applied
        """
        # Calculate channel means
        mean_r, mean_g, mean_b = self.calculate_channel_means(image_tensor)
        
        # Calculate max_patch based on smallest bounding box
        if len(annotations) > 0:
            # Calculate bounding box areas
            annotations = annotations.copy()
            annotations['bb_width'] = annotations['xmax'] - annotations['xmin']
            annotations['bb_height'] = annotations['ymax'] - annotations['ymin']
            annotations['area'] = annotations['bb_width'] * annotations['bb_height']
            
            # Get smallest bbox
            smallest_bbox = annotations['area'].min()
            
            # Calculate image size
            img_size = annotations['width'].iloc[0] * annotations['height'].iloc[0]
            
            # Calculate max_patch
            max_patch = smallest_bbox / img_size
        else:
            # If no annotations, use default max_patch from config
            max_patch = self.params.get('max_patch', 0.5)
        
        # Get probability from config
        probability = self.params.get('probability', 1.0)
        
        # Apply the blocking
        masked_image_tensor, final_image = self.hide_patch(
            image_tensor,
            mean_r,
            mean_g,
            mean_b,
            probability=probability,
            max_patch=max_patch
        )
        
        return masked_image_tensor
    
    def adjust_annotations(
        self,
        annotations: pd.DataFrame,
        img_width: int,
        img_height: int,
        **kwargs
    ) -> pd.DataFrame:
        """
        Random Blocking doesn't change bounding boxes, just copy annotations.
        
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
        Perform Random Blocking augmentation on selected images.
        
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
        print(f"Starting Random Blocking augmentation...")
        
        # Set random seed if provided
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        # Select random samples (exclude already blocked images)
        original_images = df_images[
            ~df_images['filename'].str.contains('_block', regex=False)
        ]
        
        # Calculate sample size based on percentage
        blocked_percentage = self.params.get('blocked_percentage', 0.1)
        sample_size = int(len(original_images) * blocked_percentage)
        sample_size = max(sample_size, 1)  # At least 1 image
        
        # Override with num_samples if specified
        if self.num_samples > 0:
            sample_size = min(self.num_samples, len(original_images))
        
        # Randomly select images
        selected_images_df = original_images.sample(n=sample_size, random_state=self.random_seed)
        
        augmented_images = []
        augmented_annotations = []
        processed_count = 0
        
        for idx, row in selected_images_df.iterrows():
            original_filename = row['filename']
            image_path = row['path']
            
            try:
                # Get annotations for this image
                annotations = df_annotations[
                    df_annotations['filename'] == original_filename
                ]
                
                if len(annotations) == 0:
                    print(f"No annotations found for {original_filename}. Skipping.")
                    continue
                
                # Load image tensor
                tensor = load_image_as_tensor(image_path)
                
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
                
                processed_count += 1
                
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
            
            print(f"Random Blocking augmentation completed. Added {len(augmented_images)} images.")
        else:
            print("No images were augmented.")
        
        return df_images, df_annotations
