import logging
import pandas as pd
from pathlib import Path
from src.data_cleaning.data_cleaning import main_cleaning_pipeline
from src.etl.extract_transform import etl_pipeline
from src.feature_engineering.engineer_feature import feature_engineering_pipeline
from src.modeling import modeling_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # If running in a .py script
    base_dir = Path(__file__).parent  # This gives the directory containing the script
except NameError:
    # If running in a notebook
    base_dir = Path.cwd()  # This gives the current working directory

# Construct the path to config.json
config_path = base_dir / 'src' / 'config' / 'api_config.json'
config_path = config_path.resolve()  # Resolve to an absolute path for safety

# Construct the path to config.json
config_path = base_dir / '..' / 'config' / 'api_config.json'
config_path = config_path.resolve()  # Resolve to an absolute path for safety

def main():
    logging.info("Starting the data science pipeline...")
    
    # Load the datasets
    data_path = "data/processed/data.csv"

    # Read the data
    df = pd.read_csv(data_path,low_memory = False)

    # Step 1: ETL
    # logging.info("Running ETL process...")
    # etl_pipeline(config_path)
    
    # Step 2: Data Cleaning
    logging.info("Running data cleaning...")
    cleaned_data = main_cleaning_pipeline(df)

    # Step 3: Feature Engineering
    logging.info("Running data cleaning...")
    cleaned_data = feature_engineering_pipeline(cleaned_data)

    # Step 4: Modeling
    logging.info("Running modeling pipeline...")
    param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0, 0.5, 1, 1.5],
        "alpha": [0, 0.5, 1]
    }
    modeling_results = modeling_pipeline(cleaned_data, param_dist)

    logging.info(f"Modeling completed! Results: {modeling_results}")

    logging.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()