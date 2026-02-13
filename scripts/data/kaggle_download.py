import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This must happen BEFORE importing kaggle to avoid auth errors
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset_slug, output_path):
    """
    Downloads and unzips a dataset from Kaggle to the specified path.
    """
    # Ensure Kaggle environment variables are set from .env
    # The Kaggle API reads os.environ['KAGGLE_USERNAME'] and os.environ['KAGGLE_KEY']
    # If they are in the .env file, load_dotenv() puts them in os.environ
    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables are not set.")
        print("Please ensure you have a .env file with your Kaggle credentials.")
        return

    
    # Manually authenticate using environment variables
    kaggle_user = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')

    # Debug prints (masking key for security)
    print(f"Debug: KAGGLE_USERNAME found: {kaggle_user}")
    print(f"Debug: KAGGLE_KEY found: {'*' * 5 if kaggle_key else 'None'}")

    if not kaggle_user or not kaggle_key:
         print("Error: Credentials not found in environment.")
         return

    # Ensure they are in os.environ for the API to see
    os.environ['KAGGLE_USERNAME'] = kaggle_user
    os.environ['KAGGLE_KEY'] = kaggle_key

    try:
        api = KaggleApi()
        print(f"Authenticating as {kaggle_user}...")
        api.authenticate()
        
        print(f"Downloading {dataset_slug} to {output_path}...")
        api.dataset_download_files(dataset_slug, path=output_path, unzip=True)
        
        print(f"Successfully downloaded and unzipped {dataset_slug} to {output_path}.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have configured your kaggle.json file correctly.")
        print("See docs/kaggle_setup.md for instructions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and unzip a Kaggle dataset.")
    parser.add_argument("--dataset_slug", required=True, help="The slug of the dataset to download (e.g., 'user/dataset-name').")
    
    args = parser.parse_args()
    
    # Define the target directory relative to the project root
    # Assuming the script is run from the project root or scripts/ directory
    # Ideally, we find the project root. For simplicity, we assume running from root or check relative paths.
    
    # Check if we are in 'scripts' directory, if so go up one level
    if os.path.basename(os.getcwd()) == "scripts":
        base_dir = os.path.dirname(os.getcwd())
    else:
        base_dir = os.getcwd()
        
    target_dir = os.path.join(base_dir, "data", "raw")
    
    if not os.path.exists(target_dir):
        print(f"Creating directory: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        
    download_dataset(args.dataset_slug, target_dir)
