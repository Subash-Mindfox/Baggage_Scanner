"""
Augmentation pipeline for orchestrating multiple augmentations.
"""

import os
from typing import Dict, List, Tuple
import pandas as pd

from src.utils.config_loader import load_config
from src.augmentation.registry import AugmentationRegistry


class AugmentationPipeline:
    """
    Orchestrates the augmentation process.
    
    This class manages the execution of multiple augmentations in sequence,
    based on the configuration file.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the augmentation pipeline.
        
        Parameters
        ----------
        config : dict
            Augmentation configuration dictionary
        """
        self.config = config
        self.augmentors = self._initialize_augmentors()
    
    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Create pipeline from a configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to the augmentation configuration YAML file
            
        Returns
        -------
        AugmentationPipeline
            Initialized pipeline
        """
        config = load_config(config_path)
        return cls(config)
    
    def _initialize_augmentors(self) -> List[Tuple[str, object]]:
        """
        Initialize all enabled augmentors from config.
        
        Returns
        -------
        list
            List of tuples (name, augmentor_instance)
        """
        augmentors = []
        
        augmentation_configs = self.config.get('augmentations', {})
        
        for name, aug_config in augmentation_configs.items():
            # Skip if not enabled
            if isinstance(aug_config, bool):
                if not aug_config:
                    print(f"Augmentation '{name}' is disabled. Skipping.")                    
                    continue
                aug_config = {"enabled": True}                
            
            # Check if augmentation is registered
            if not AugmentationRegistry.is_registered(name):
                print(f"Warning: Augmentation '{name}' is not registered. Skipping.")
                continue
            
            # Create augmentor instance
            try:
                augmentor = AugmentationRegistry.get_augmentor(name, aug_config)
                augmentors.append((name, augmentor))
                print(f"Initialized augmentation: {name}")
            except Exception as e:
                print(f"Error initializing augmentation '{name}': {e}")
        
        return augmentors
    
    def run(
        self,
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame,
        output_base_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute all augmentations in sequence.
        
        Parameters
        ----------
        df_images : pd.DataFrame
            DataFrame containing image metadata
        df_annotations : pd.DataFrame
            DataFrame containing annotations
        output_base_dir : str
            Base directory for saving augmented images
            
        Returns
        -------
        tuple
            (updated df_images, updated df_annotations)
        """
        if not self.augmentors:
            print("No augmentations enabled. Returning original data.")
            return df_images, df_annotations
        
        print(f"\n{'='*80}")
        print(f"Starting Augmentation Pipeline")
        print(f"Number of augmentations: {len(self.augmentors)}")
        print(f"{'='*80}\n")
        
        # Execute each augmentation
        for name, augmentor in self.augmentors:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
            
            # Create output directory for this augmentation
            output_dir = os.path.join(output_base_dir, name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Run augmentation
            try:
                df_images, df_annotations = augmentor.augment_dataset(
                    df_images,
                    df_annotations,
                    output_dir
                )
            except Exception as e:
                print(f"Error running augmentation '{name}': {e}")
                print("Continuing with next augmentation...")
        
        print(f"\n{'='*80}")
        print(f"Augmentation Pipeline Completed")
        print(f"Total images: {len(df_images)}")
        print(f"Total annotations: {len(df_annotations)}")
        print(f"{'='*80}\n")
        
        return df_images, df_annotations
    
    def list_enabled_augmentations(self) -> List[str]:
        """
        Get list of enabled augmentation names.
        
        Returns
        -------
        list
            List of enabled augmentation names
        """
        return [name for name, _ in self.augmentors]
    
    def enable_augmentation(self, name: str) -> None:
        """
        Enable a specific augmentation.
        
        Parameters
        ----------
        name : str
            Name of the augmentation to enable
        """
        if name in self.config['augmentations']:
            self.config['augmentations'][name]['enabled'] = True
            # Re-initialize augmentors
            self.augmentors = self._initialize_augmentors()
        else:
            print(f"Augmentation '{name}' not found in config.")
    
    def disable_augmentation(self, name: str) -> None:
        """
        Disable a specific augmentation.
        
        Parameters
        ----------
        name : str
            Name of the augmentation to disable
        """
        if name in self.config['augmentations']:
            self.config['augmentations'][name]['enabled'] = False
            # Re-initialize augmentors
            self.augmentors = self._initialize_augmentors()
        else:
            print(f"Augmentation '{name}' not found in config.")
