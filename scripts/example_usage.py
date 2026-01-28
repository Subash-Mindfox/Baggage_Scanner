"""
Example usage script demonstrating the baggage scanner system.

This script shows how to use the various components of the system.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config_loader import load_config
from src.augmentation.pipeline import AugmentationPipeline
from src.preprocessing.whitespace_removal import WhitespaceRemover

# Import augmentation modules to register them
from src.augmentation import color_jitter, flip_augmentor


def example_load_config():
    """Example: Load configuration files."""
    print("\n" + "="*80)
    print("Example 1: Loading Configurations")
    print("="*80)
    
    paths_config = load_config('configs/paths.yaml')
    aug_config = load_config('configs/augmentation.yaml')
    
    print(f"Raw data root: {paths_config['paths']['raw_data_root']}")
    print(f"Augmentations enabled: {aug_config['augmentations']['enabled']}")
    print(f"Available augmentations: {list(aug_config['augmentations'].keys())}")


def example_whitespace_removal():
    """Example: Configure whitespace removal."""
    print("\n" + "="*80)
    print("Example 2: Whitespace Removal Configuration")
    print("="*80)
    
    preprocessing_config = load_config('configs/preprocessing.yaml')
    
    # Create whitespace remover
    ws_remover = WhitespaceRemover(preprocessing_config['preprocessing']['whitespace_removal'])
    
    print(f"Whitespace removal enabled: {ws_remover.enabled}")
    print(f"Vertical threshold: {ws_remover.v_threshold}")
    print(f"Horizontal threshold: {ws_remover.h_threshold}")
    
    # Disable whitespace removal
    preprocessing_config['preprocessing']['whitespace_removal']['enabled'] = False
    ws_remover_disabled = WhitespaceRemover(preprocessing_config['preprocessing']['whitespace_removal'])
    print(f"After disabling - enabled: {ws_remover_disabled.enabled}")


def example_augmentation_pipeline():
    """Example: Create and configure augmentation pipeline."""
    print("\n" + "="*80)
    print("Example 3: Augmentation Pipeline")
    print("="*80)
    
    # Load config
    aug_config = load_config('configs/augmentation.yaml')
    
    # Create pipeline
    pipeline = AugmentationPipeline(aug_config)
    
    # Show enabled augmentations
    enabled = pipeline.list_enabled_augmentations()
    print(f"Enabled augmentations: {enabled}")
    
    # Disable an augmentation programmatically
    pipeline.disable_augmentation('color_jitter')
    print(f"After disabling color_jitter: {pipeline.list_enabled_augmentations()}")
    
    # Re-enable it
    pipeline.enable_augmentation('color_jitter')
    print(f"After re-enabling: {pipeline.list_enabled_augmentations()}")


def example_selective_augmentation():
    """Example: Run only specific augmentations."""
    print("\n" + "="*80)
    print("Example 4: Selective Augmentation")
    print("="*80)
    
    # Load config
    aug_config = load_config('configs/augmentation.yaml')
    
    # Disable all except flip
    aug_config['augmentations']['color_jitter']['enabled'] = False
    aug_config['augmentations']['gaussian_blur']['enabled'] = False
    aug_config['augmentations']['flip']['enabled'] = True
    
    pipeline = AugmentationPipeline(aug_config)
    
    print(f"Only flip enabled: {pipeline.list_enabled_augmentations()}")
    
    # This would run only flip augmentation:
    # df_images, df_annotations = pipeline.run(df_images, df_annotations, output_dir)


def example_model_configuration():
    """Example: Model configuration."""
    print("\n" + "="*80)
    print("Example 5: Model Configuration")
    print("="*80)
    
    model_config = load_config('configs/model_config.yaml')
    
    print(f"Model architecture: {model_config['model']['architecture']}")
    print(f"Number of classes: {model_config['model']['num_classes']}")
    print(f"Batch size: {model_config['training']['batch_size']}")
    print(f"Learning rate: {model_config['training']['learning_rate']}")
    
    # To switch to YOLO, just change config:
    model_config['model']['architecture'] = 'yolo'
    print(f"Switched to: {model_config['model']['architecture']}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("BAGGAGE SCANNER SYSTEM - USAGE EXAMPLES")
    print("="*80)
    
    try:
        example_load_config()
        example_whitespace_removal()
        example_augmentation_pipeline()
        example_selective_augmentation()
        example_model_configuration()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have all config files in the configs/ directory")


if __name__ == "__main__":
    main()
