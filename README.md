# service_requests_project

This project implements an end-to-end data science pipeline for processing, analyzing, and modeling data. It encompasses the following stages:

- ETL (Extract, Transform, Load): Fetches data from an external API and saves it in a structured format.
- Data Cleaning: Cleans the raw data by handling missing values, transforming columns, and dropping redundant fields.
- Feature Engineering: Creates new features, applies encoding, and scales numerical data.
- Modeling: Trains a machine learning model, tunes hyperparameters, evaluates performance, and saves the trained model and metrics.

### Folder Structure
The project is organized as follows:

project_name/
│
├── data/                     # Data storage
│   ├── processed/            # Cleaned and transformed data
│   └── output/               # Final outputs before predictions
│
├── notebooks/                # Jupyter notebooks for exploration
│   └── notebook_code.ipynb   # Data exploration and initial analysis notebook
│
├── src/                      # Source code
│   ├── config/               # Configuration files
│   │   └── api_config.json   # API and pipeline configuration
│   ├── etl/                  # ETL scripts
│   │   └── extract_transform.py
│   ├── data_cleaning/        # Data cleaning scripts
│   │   └── clean_data.py
│   ├── feature_engineering/  # Feature engineering scripts
│   │   └── engineer_features.py
│   ├── modeling/             # Modeling scripts
│   │   ├── train_model.py    # Model training script
│   └── utils/                # Utility functions
│       └── helper_functions.py
│
├── main.py                   # Entry point for running the entire pipeline
└── .gitignore                # Files to ignore in version control

### Step-by-Step Instructions
1. Install Dependencies
Install the required Python packages

2. Run the Pipeline
To execute the full pipeline, run the main script:

python main.py

This script orchestrates all pipeline steps, including ETL, data cleaning, feature engineering, and modeling.

3. Run Individual Components
If you wish you can run each step independently 
