import yaml
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import xml.etree.ElementTree as ET
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import read_image
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv
import random


def get_image_dataframe_lazy(image_folder, name):
    """
    Create a DataFrame with image metadata only (NO tensors).
    Images are loaded lazily when needed.
    """

    image_stats = []

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(".jpg"):
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

def compute_csv_stats(csv_path, name):
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
        return None  # Return None if there's an error
    

df_train_images = get_image_dataframe_lazy(
    "Temp\Train", "Training Images"
)
df_test_images = get_image_dataframe_lazy(
    "Temp\Test", "Testing Images"
)

df_train_annotations = compute_csv_stats("Temp\Test\_annotationstest.csv", 'Training Annotations')
df_test_annotations = compute_csv_stats("Temp\Train\_annotationstest.csv", 'Testing Annotations')

random_seed = 42
random.seed(random_seed)

# Calculate the number of images to select (1% of total)
num_images_to_select = int(0.01 * len(df_train_images))

# Randomly select 30% of the images
selected_filenames = random.sample(list(df_train_images['filename']), num_images_to_select)

# Filter df_test_images to keep only the selected images
df_train_images = df_train_images[df_train_images['filename'].isin(selected_filenames)].reset_index(drop=True)

# Filter df_test_annotations to keep only annotations matching the selected images
df_train_annotations = df_train_annotations[df_train_annotations['filename'].isin(selected_filenames)].reset_index(drop=True)

# TEST SET (Our subest is 7% of the test dataset)

# Set seed for reproducibility
random_seed = 42
random.seed(random_seed)

# Calculate the number of images to select (10% of total)
num_images_to_select = int(0.07 * len(df_test_images))

# Randomly select 30% of the images
selected_filenames = random.sample(list(df_test_images['filename']), num_images_to_select)

# Filter df_test_images to keep only the selected images
df_test_images = df_test_images[df_test_images['filename'].isin(selected_filenames)].reset_index(drop=True)

# Filter df_test_annotations to keep only annotations matching the selected images
df_test_annotations = df_test_annotations[df_test_annotations['filename'].isin(selected_filenames)].reset_index(drop=True)