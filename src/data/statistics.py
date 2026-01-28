"""
Data statistics and analysis functions.
"""

import pandas as pd
import os
from typing import Optional


def compute_csv_stats(csv_path: str, name: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV file, compute basic statistics, and return the data as a DataFrame.
    
    Parameters
    ----------
    csv_path : str
        The path to the CSV file containing annotations
    name : str
        Name to use in the printed output for the dataset
        
    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the loaded CSV data with computed stats.
        Returns None if an error occurs while loading the CSV.
    """
    try:
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_path)
        
        # Display basic stats
        print(f"--- Stats for {name} ---")
        print(f"Total Rows: {len(df)}")
        print(f"Unique Images: {df['filename'].nunique()}")
        
        # Show a summary of the dataset (describe numeric fields)
        print("\nSummary Statistics:")
        print(df.describe())
        
        # Add a blank line after each section for readability
        print("\n" * 3)
        
        # Return the DataFrame for later use
        return df
    
    except Exception as e:
        print(f"An error occurred while processing {name}: {e}")
        return None


def get_image_dataframe_lazy(image_folder: str, name: str) -> pd.DataFrame:
    """
    Create a DataFrame with image metadata only (NO tensors).
    Images are loaded lazily when needed.
    
    Parameters
    ----------
    image_folder : str
        Path to the folder containing images
    name : str
        Name for logging purposes
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: filename, path, width, height, mode
    """
    from PIL import Image
    
    image_stats = []
    
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, image_file)
            
            try:
                # Use PIL to get metadata WITHOUT loading full tensor
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode
                
                image_stats.append([
                    image_file,
                    image_path,
                    width,
                    height,
                    mode
                ])
            
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    
    df = pd.DataFrame(
        image_stats,
        columns=["filename", "path", "width", "height", "mode"]
    )
    
    print(f"--- Stats for {name} ---")
    print(f"Unique Images: {df['filename'].nunique()}")
    print("\nSummary of Image Dimensions:")
    print(df[["width", "height"]].describe())
    print("\n" * 3)
    
    return df


def get_class_distribution(df_annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Get the distribution of classes in the annotations.
    
    Parameters
    ----------
    df_annotations : pd.DataFrame
        DataFrame containing annotations with a 'class' column
        
    Returns
    -------
    pd.DataFrame
        Class distribution with counts and percentages
    """
    if 'class' not in df_annotations.columns:
        print("Warning: 'class' column not found in annotations")
        return pd.DataFrame()
    
    class_counts = df_annotations['class'].value_counts()
    class_percentages = (class_counts / len(df_annotations)) * 100
    
    distribution = pd.DataFrame({
        'count': class_counts,
        'percentage': class_percentages
    })
    
    print("\n--- Class Distribution ---")
    print(distribution)
    print()
    
    return distribution


def get_bbox_statistics(df_annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics about bounding box sizes.
    
    Parameters
    ----------
    df_annotations : pd.DataFrame
        DataFrame containing annotations with bbox columns
        
    Returns
    -------
    pd.DataFrame
        Statistics about bbox dimensions
    """
    required_cols = ['xmin', 'ymin', 'xmax', 'ymax']
    if not all(col in df_annotations.columns for col in required_cols):
        print("Warning: Required bbox columns not found")
        return pd.DataFrame()
    
    # Calculate bbox dimensions
    df_annotations['bbox_width'] = df_annotations['xmax'] - df_annotations['xmin']
    df_annotations['bbox_height'] = df_annotations['ymax'] - df_annotations['ymin']
    df_annotations['bbox_area'] = df_annotations['bbox_width'] * df_annotations['bbox_height']
    
    # Get statistics
    stats = df_annotations[['bbox_width', 'bbox_height', 'bbox_area']].describe()
    
    print("\n--- Bounding Box Statistics ---")
    print(stats)
    print()
    
    return stats
