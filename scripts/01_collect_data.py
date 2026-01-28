"""
Data collection and analysis script.

This script loads raw data, computes statistics, and prepares it for preprocessing.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import random

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data.statistics import (
    get_image_dataframe_lazy,
    compute_csv_stats,
    get_class_distribution,
    get_bbox_statistics
)


def subsample_dataset(df_images, df_annotations, percentage, random_seed=42):
    """
    Subsample the dataset to a percentage of the original size.
    
    Parameters
    ----------
    df_images : pd.DataFrame
        DataFrame containing image metadata
    df_annotations : pd.DataFrame
        DataFrame containing annotations
    percentage : float
        Percentage of data to keep (0.0 to 1.0)
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (subsampled_df_images, subsampled_df_annotations)
    """
    random.seed(random_seed)
    
    # Calculate number of images to select
    num_images_to_select = int(percentage * len(df_images))
    
    # Randomly select filenames
    selected_filenames = random.sample(
        list(df_images['filename']),
        num_images_to_select
    )
    
    # Filter dataframes
    df_images_subset = df_images[
        df_images['filename'].isin(selected_filenames)
    ].reset_index(drop=True)
    
    df_annotations_subset = df_annotations[
        df_annotations['filename'].isin(selected_filenames)
    ].reset_index(drop=True)
    
    return df_images_subset, df_annotations_subset


def main():
    """Main data collection function."""
    
    # Setup logger
    logger = setup_logger('data_collection', 'outputs/logs/data_collection.log')
    logger.info("Starting data collection and analysis")
    
    # Load configuration
    paths_config = load_config('configs/paths.yaml')
    
    logger.info("\n" + "="*80)
    logger.info("LOADING RAW DATA")
    logger.info("="*80)
    
    # Load training data
    logger.info("\n--- Training Data ---")
    df_train_images = get_image_dataframe_lazy(
        paths_config['paths']['trainingImages'],
        "Training Images"
    )
    
    df_train_annotations = compute_csv_stats(
        paths_config['paths']['trainningAnnot_file'],
        'Training Annotations'
    )
    
    # Load test data
    logger.info("\n--- Test Data ---")
    df_test_images = get_image_dataframe_lazy(
        paths_config['paths']['testImages'],
        "Testing Images"
    )
    
    df_test_annotations = compute_csv_stats(
        paths_config['paths']['testAnnot_file'],
        'Testing Annotations'
    )
    
    # Analyze class distribution
    logger.info("\n" + "="*80)
    logger.info("CLASS DISTRIBUTION ANALYSIS")
    logger.info("="*80)
    
    if df_train_annotations is not None:
        logger.info("\nTraining Set:")
        train_class_dist = get_class_distribution(df_train_annotations)
    
    if df_test_annotations is not None:
        logger.info("\nTest Set:")
        test_class_dist = get_class_distribution(df_test_annotations)
    
    # Analyze bounding box statistics
    logger.info("\n" + "="*80)
    logger.info("BOUNDING BOX STATISTICS")
    logger.info("="*80)
    
    if df_train_annotations is not None:
        logger.info("\nTraining Set:")
        train_bbox_stats = get_bbox_statistics(df_train_annotations)
    
    if df_test_annotations is not None:
        logger.info("\nTest Set:")
        test_bbox_stats = get_bbox_statistics(df_test_annotations)
    
    # Optional: Subsample for faster experimentation
    # Uncomment these lines if you want to work with a smaller dataset initially
    
    # logger.info("\n" + "="*80)
    # logger.info("SUBSAMPLING DATASET (Optional)")
    # logger.info("="*80)
    # 
    # TRAIN_SUBSAMPLE_PERCENTAGE = 0.10  # Use 10% of training data
    # TEST_SUBSAMPLE_PERCENTAGE = 0.10   # Use 10% of test data
    # 
    # logger.info(f"\nSubsampling training set to {TRAIN_SUBSAMPLE_PERCENTAGE*100}%...")
    # df_train_images, df_train_annotations = subsample_dataset(
    #     df_train_images,
    #     df_train_annotations,
    #     TRAIN_SUBSAMPLE_PERCENTAGE,
    #     random_seed=42
    # )
    # logger.info(f"Training images after subsampling: {len(df_train_images)}")
    # logger.info(f"Training annotations after subsampling: {len(df_train_annotations)}")
    # 
    # logger.info(f"\nSubsampling test set to {TEST_SUBSAMPLE_PERCENTAGE*100}%...")
    # df_test_images, df_test_annotations = subsample_dataset(
    #     df_test_images,
    #     df_test_annotations,
    #     TEST_SUBSAMPLE_PERCENTAGE,
    #     random_seed=42
    # )
    # logger.info(f"Test images after subsampling: {len(df_test_images)}")
    # logger.info(f"Test annotations after subsampling: {len(df_test_annotations)}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("="*80)
    logger.info(f"\nTraining Images: {len(df_train_images)}")
    logger.info(f"Training Annotations: {len(df_train_annotations) if df_train_annotations is not None else 0}")
    logger.info(f"\nTest Images: {len(df_test_images)}")
    logger.info(f"Test Annotations: {len(df_test_annotations) if df_test_annotations is not None else 0}")
    logger.info("\nData collection completed successfully!")
    logger.info("Next step: Run 02_preprocess.py to preprocess the data")


if __name__ == "__main__":
    main()
