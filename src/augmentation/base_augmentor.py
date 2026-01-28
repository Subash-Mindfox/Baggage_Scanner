"""
Abstract base class for all augmentation operations.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import os


class BaseAugmentor(ABC):
    """
    Abstract base class for all augmentation operations.
    
    This class provides the common interface and functionality for all augmentations.
    Subclasses must implement the abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the augmentor with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary for this augmentation
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.num_samples = config.get('num_samples', 0)
        self.random_seed = config.get('random_seed', None)
        self.params = config.get('params', {})
        self.suffix = self._get_suffix()
    
    @abstractmethod
    def _get_suffix(self) -> str:
        """
        Return filename suffix for this augmentation.
        
        Returns
        -------
        str
            Suffix to add to filenames (e.g., '_jit.jpg')
        """
        pass
    
    @abstractmethod
    def apply_transform(self, image_tensor, **kwargs):
        """
        Apply the augmentation transformation to an image tensor.
        
        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor
        **kwargs : dict
            Additional parameters for the transformation
            
        Returns
        -------
        torch.Tensor
            Transformed image tensor
        """
        pass
    
    @abstractmethod
    def adjust_annotations(
        self,
        annotations: pd.DataFrame,
        img_width: int,
        img_height: int,
        **kwargs
    ) -> pd.DataFrame:
        """
        Adjust bounding box annotations after transformation.
        
        Parameters
        ----------
        annotations : pd.DataFrame
            Original annotations for the image
        img_width : int
            Image width
        img_height : int
            Image height
        **kwargs : dict
            Additional parameters for annotation adjustment
            
        Returns
        -------
        pd.DataFrame
            Adjusted annotations
        """
        pass
    
    def augment_dataset(
        self,
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame,
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method to augment dataset.
        
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
        if not self.enabled:
            print(f"{self.__class__.__name__} is disabled. Skipping.")
            return df_images, df_annotations
        
        # Check if already augmented
        if self._is_already_augmented(df_images):
            print(f"{self.__class__.__name__} already applied. Skipping.")
            return df_images, df_annotations
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform augmentation
        return self._perform_augmentation(df_images, df_annotations, output_dir)
    
    @abstractmethod
    def _perform_augmentation(
        self,
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame,
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Implement specific augmentation logic.
        
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
        pass
    
    def _is_already_augmented(self, df_images: pd.DataFrame) -> bool:
        """
        Check if augmentation has already been applied.
        
        Parameters
        ----------
        df_images : pd.DataFrame
            DataFrame containing image metadata
            
        Returns
        -------
        bool
            True if augmentation already applied, False otherwise
        """
        return df_images['filename'].str.contains(self.suffix).any()
    
    def _select_random_samples(
        self,
        df_images: pd.DataFrame,
        num_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Select random samples from the dataset.
        
        Parameters
        ----------
        df_images : pd.DataFrame
            DataFrame containing image metadata
        num_samples : int, optional
            Number of samples to select. If None, uses self.num_samples
            
        Returns
        -------
        pd.DataFrame
            Selected samples
        """
        import numpy as np
        
        if num_samples is None:
            num_samples = self.num_samples
        
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Get original images (exclude already augmented ones)
        original_images = df_images[
            ~df_images['filename'].str.contains('_', regex=False)
        ]
        
        unique_filenames = original_images['filename'].unique()
        
        # Adjust num_samples if necessary
        if len(unique_filenames) < num_samples:
            num_samples = len(unique_filenames)
            print(f"Only {num_samples} unique images available. Selecting all of them.")
        
        # Select random filenames
        selected_filenames = np.random.choice(
            unique_filenames,
            size=num_samples,
            replace=False
        )
        
        # Filter DataFrame
        return df_images[df_images['filename'].isin(selected_filenames)].copy()
