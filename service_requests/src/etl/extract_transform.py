import os
import json
import requests
import pandas as pd
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    filename="data_fetcher.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

try:
    # If running in a .py script
    base_dir = Path(__file__).parent  # This gives the directory containing the script
except NameError:
    # If running in a notebook
    base_dir = Path.cwd()  # This gives the current working directory

# Construct the path to config.json
config_path = base_dir / '..' / 'config' / 'api_config.json'
config_path = config_path.resolve()  # Resolve to an absolute path for safety

def load_config(config_file):
    """
    Load configuration from a JSON file.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration data as a dictionary.
    """
    with open(config_file, "r") as file:
        return json.load(file)

def get_offset(output_file):
    """
    Determine the starting offset based on the existing output file.

    Args:
        output_file (str): Path to the CSV file where data is saved.

    Returns:
        int: Starting offset (number of existing rows in the file).
    """
    if os.path.exists(output_file):
        logging.info(f"Resuming data fetch from existing file: {output_file}")
        existing_data = pd.read_csv(output_file)
        return len(existing_data)
    logging.info(f"No existing file found. Starting fresh.")
    return 0

def fetch_data(api_url, offset, chunk_size):
    """
    Fetch data from the API using offset and limit.

    Args:
        api_url (str): Base API URL with query parameters (excluding offset and limit).
        offset (int): The starting record number for the API call.
        chunk_size (int): Number of records to fetch per API call.

    Returns:
        list[dict] or None: List of records (JSON objects) if successful, or None if an error occurs.
    """
    try:
        paginated_url = f"{api_url}&$offset={offset}&$limit={chunk_size}"
        response = requests.get(paginated_url, timeout=60)
        response.raise_for_status()  # Raise HTTP errors if any
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None

def save_data(data, output_file):
    """
    Save fetched data to a CSV file.

    Args:
        data (list[dict]): List of records to save.
        output_file (str): Path to the CSV file where data will be appended or created.

    Returns:
        None
    """
    df = pd.DataFrame(data)
    if not df.empty:
        write_mode = "a" if os.path.exists(output_file) else "w"
        header = not os.path.exists(output_file)
        df.to_csv(output_file, index=False, mode=write_mode, header=header)
        logging.info(f"Saved {len(df)} records to {output_file}.")

def etl_pipeline(config_path):
    """
    Main function to orchestrate the data fetching process.
    
    """
    # Load configurations
    config = load_config(config_path)
    base_url = config["api_base_url"]
    output_file = config["output_file"]
    chunk_size = config["chunk_size"]
    total_records = config["total_records"]
    columns = config["columns"]
    
    # Prepare API endpoint
    select_fields = ",".join(columns)
    api_url = f"{base_url}?$select={select_fields}&$order=created_date DESC"
    
    # Determine the starting offset
    offset = get_offset(output_file)
    
    # Fetch and save data in chunks
    while offset < total_records:
        logging.info(f"Fetching records starting at offset {offset}")
        data = fetch_data(api_url, offset, chunk_size)
        
        if not data:
            logging.info("No more data to fetch or an error occurred.")
            break
        
        save_data(data, output_file)
        offset += chunk_size
        time.sleep(1)  # Throttle requests
    
    logging.info(f"Data fetching completed. Data saved to '{output_file}'.")