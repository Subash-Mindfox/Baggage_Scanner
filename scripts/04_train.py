"""
Training script - Train object detection model.

This script trains a detection model on the augmented dataset.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_experiment_logger
from src.models.faster_rcnn import FasterRCNNModel


def create_dataloader(df_images, df_annotations, batch_size=4):
    """
    Create a DataLoader for training.
    
    This is a placeholder - you'll need to implement a proper Dataset class.
    """
    # TODO: Implement proper Dataset class in src/data/dataset.py
    print("Note: Implement custom Dataset class for your data format")
    return None


def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Parameters
    ----------
    model : BaseDetectionModel
        Model to train
    dataloader : DataLoader
        Training data loader
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to train on
    epoch : int
        Current epoch number
        
    Returns
    -------
    dict
        Training metrics
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {losses.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return {'loss': avg_loss}


def main():
    """Main training function."""
    
    # Setup experiment logger
    logger = get_experiment_logger('training', 'outputs/logs')
    logger.info("Starting training")
    
    # Load configurations
    paths_config = load_config('configs/paths.yaml')
    model_config = load_config('configs/model_config.yaml')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load augmented data
    logger.info("Loading augmented data...")
    augmented_csv = os.path.join(
        paths_config['paths']['augmented_root'],
        'augmented_annotations.csv'
    )
    
    if not os.path.exists(augmented_csv):
        logger.error(f"Augmented data not found at {augmented_csv}")
        logger.error("Please run 03_augment.py first")
        return
    
    df_annotations = pd.read_csv(augmented_csv)
    logger.info(f"Loaded {len(df_annotations)} annotations")
    
    # Create model
    logger.info("Creating model...")
    model = FasterRCNNModel(model_config['model'])
    model.to_device(device)
    logger.info(f"Model has {model.count_parameters():,} trainable parameters")
    
    # TODO: Create DataLoader
    # train_loader = create_dataloader(df_images, df_annotations, 
    #                                  batch_size=model_config['training']['batch_size'])
    
    # Setup optimizer
    training_config = model_config['training']
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=training_config['learning_rate'],
        momentum=training_config['momentum'],
        weight_decay=training_config['weight_decay']
    )
    
    # Training loop placeholder
    logger.info("Training loop...")
    logger.info("Note: Implement proper Dataset and DataLoader before training")
    
    # num_epochs = training_config['num_epochs']
    # for epoch in range(num_epochs):
    #     train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
    #     logger.info(f"Epoch {epoch}: {train_metrics}")
    #     
    #     # Save checkpoint
    #     if (epoch + 1) % training_config['checkpoint']['save_frequency'] == 0:
    #         checkpoint_path = os.path.join(
    #             paths_config['paths']['checkpoints'],
    #             f'checkpoint_epoch_{epoch+1}.pth'
    #         )
    #         model.save_checkpoint(checkpoint_path, epoch, optimizer, train_metrics)
    
    logger.info("Training script template created. Implement Dataset class to begin training.")


if __name__ == "__main__":
    main()
