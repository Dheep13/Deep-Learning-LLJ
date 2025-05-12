"""
Script to download the dataset from Kaggle.
"""

import os
import sys
import zipfile
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
from tqdm import tqdm

def download_dataset():
    """
    Download the dataset from Kaggle.
    
    Note: You need to have a Kaggle API token set up in ~/.kaggle/kaggle.json
    To set up the Kaggle API token:
    1. Go to your Kaggle account settings (https://www.kaggle.com/account)
    2. Click on "Create New API Token" to download kaggle.json
    3. Place the downloaded file in ~/.kaggle/ (create the directory if it doesn't exist)
    4. Set the permissions: chmod 600 ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
    except ImportError:
        print("Kaggle API not found. Installing...")
        os.system("pip install kaggle")
        import kaggle
    
    print(f"Downloading dataset: {config.KAGGLE_DATASET}")
    
    # Check if the dataset already exists
    if os.path.exists(config.RAW_DATA_DIR) and len(os.listdir(config.RAW_DATA_DIR)) > 0:
        print(f"Dataset already exists in {config.RAW_DATA_DIR}")
        user_input = input("Do you want to re-download the dataset? (y/n): ")
        if user_input.lower() != 'y':
            print("Download skipped.")
            return
    
    # Download the dataset
    try:
        kaggle.api.dataset_download_files(
            config.KAGGLE_DATASET,
            path=config.DATA_DIR,
            unzip=True
        )
        print(f"Dataset downloaded successfully to {config.DATA_DIR}")
        
        # Move files to the raw directory if needed
        dataset_dir = os.path.join(config.DATA_DIR, "age-detection-human-faces-18-60-years")
        if os.path.exists(dataset_dir):
            for file in os.listdir(dataset_dir):
                src_path = os.path.join(dataset_dir, file)
                dst_path = os.path.join(config.RAW_DATA_DIR, file)
                os.rename(src_path, dst_path)
            os.rmdir(dataset_dir)
            print(f"Files moved to {config.RAW_DATA_DIR}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print(f"1. Go to {config.DATASET_URL}")
        print("2. Click on 'Download' button")
        print(f"3. Extract the downloaded zip file to {config.RAW_DATA_DIR}")

if __name__ == "__main__":
    download_dataset()
