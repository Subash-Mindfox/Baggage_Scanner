"""
Augmentation script - Apply augmentations to preprocessed data.

This script loads preprocessed images and applies configured augmentations.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.augmentation.pipeline import AugmentationPipeline
from src.data.statistics import get_image_dataframe_lazy, compute_csv_stats
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

# Import augmentation implementations to register them
from src.augmentation import color_jitter, flip_augmentor


def main():
    """Main function for running augmentations."""
    
    # Setup logger
    logger = setup_logger('augmentation', 'outputs/logs/augmentation.log')
    logger.info("Starting augmentation pipeline")
    
    # Load configurations
    paths_config = load_config('configs/paths.yaml')
    aug_config = load_config('configs/augmentation.yaml')
    
    # Load preprocessed data
    logger.info("Loading preprocessed training data...")
    df_train_images = get_image_dataframe_lazy(
        paths_config['paths']['preprocessed_train'],
        "Preprocessed Training Images"
    )
    
    df_train_annotations = compute_csv_stats(
        os.path.join(paths_config['paths']['preprocessed_train'], '_annotations.csv'),
        'Preprocessed Training Annotations'
    )
    
    # Create augmentation pipeline
    logger.info("Initializing augmentation pipeline...")
    pipeline = AugmentationPipeline(aug_config)
    
    # Show enabled augmentations
    enabled = pipeline.list_enabled_augmentations()
    logger.info(f"Enabled augmentations: {enabled}")
    
    # Run augmentations
    logger.info("Running augmentation pipeline...")
    df_images_aug, df_annotations_aug = pipeline.run(
        df_train_images,
        df_train_annotations,
        paths_config['paths']['augmented_root']
    )
    
    # Save final augmented data metadata
    output_csv = os.path.join(
        paths_config['paths']['augmented_root'],
        'augmented_annotations.csv'
    )
    df_annotations_aug.to_csv(output_csv, index=False)
    logger.info(f"Saved augmented annotations to {output_csv}")
    
    # Print summary
    logger.info(f"\nAugmentation Summary:")
    logger.info(f"Original images: {len(df_train_images)}")
    logger.info(f"Augmented images (total): {len(df_images_aug)}")
    logger.info(f"New images added: {len(df_images_aug) - len(df_train_images)}")
    logger.info(f"Total annotations: {len(df_annotations_aug)}")
    
    logger.info("Augmentation pipeline completed successfully!")


if __name__ == "__main__":
    main()
