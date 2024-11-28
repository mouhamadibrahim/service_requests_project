import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="feature_engineering.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

# Utility functions

def convert_to_datetime(df, columns):
    """
    Convert specified columns to datetime format.

    Args:
        df (pd.DataFrame): The input DataFrame containing date columns.
        columns (list): List of column names to convert to datetime.

    Returns:
        pd.DataFrame: The DataFrame with converted datetime columns.

    Description:
        Converts the specified columns to Pandas datetime format, handling errors gracefully.
    """
    logger.info(f"Converting columns {columns} to datetime format.")
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def calculate_response_time(df, start_col, end_col, target_col):
    """
    Calculate response time in hours between two datetime columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        start_col (str): Name of the start datetime column.
        end_col (str): Name of the end datetime column.
        target_col (str): Name of the new column to store response time.

    Returns:
        pd.DataFrame: The DataFrame with the response time column added.

    Description:
        Calculates the difference in hours between the start and end datetime columns and stores
        the result in a new column.
    """
    logger.info(f"Calculating response time from '{start_col}' to '{end_col}'.")
    df[target_col] = (df[end_col] - df[start_col]).dt.total_seconds() / 3600
    return df


def extract_date_features(df, column, prefix):
    """
    Extract date-based features (hour, day, month) from a datetime column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the datetime column to process.
        prefix (str): The prefix for new feature column names.

    Returns:
        pd.DataFrame: The DataFrame with new date-based feature columns added.

    Description:
        Extracts the hour, day of the week, and month from the specified datetime column.
    """
    logger.info(f"Extracting date features from '{column}'.")
    df[f'{prefix}_hour'] = df[column].dt.hour
    df[f'{prefix}_day'] = df[column].dt.dayofweek
    df[f'{prefix}_month'] = df[column].dt.month
    return df


def create_weekday_feature(df, column, target_col):
    """
    Create a binary feature indicating if the date is a weekday or weekend.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the datetime column to process.
        target_col (str): The name of the new column for the binary feature.

    Returns:
        pd.DataFrame: The DataFrame with the weekday binary feature added.

    Description:
        Creates a binary column where 1 indicates a weekday and 0 indicates a weekend.
    """
    logger.info(f"Creating weekday feature '{target_col}' from '{column}'.")
    df[target_col] = (df[column].dt.dayofweek < 5).astype(int)
    return df


def create_interaction_features(df, col1, col2, target_col):
    """
    Create an interaction feature by combining two columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The first column to combine.
        col2 (str): The second column to combine.
        target_col (str): The name of the new column for the interaction feature.

    Returns:
        pd.DataFrame: The DataFrame with the interaction feature added.

    Description:
        Combines two columns into a new feature by concatenating their values as strings.
    """
    logger.info(f"Creating interaction feature '{target_col}' from '{col1}' and '{col2}'.")
    df[target_col] = df[col1].astype(str) + "_" + df[col2].astype(str)
    return df


def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (list): List of column names to drop.

    Returns:
        pd.DataFrame: The DataFrame with the specified columns removed.

    Description:
        Removes columns that are no longer needed for analysis or modeling.
    """
    logger.info(f"Dropping columns: {columns_to_drop}.")
    df.drop(columns=columns_to_drop, inplace=True)
    return df


def label_encode_columns(df, columns):
    """
    Apply label encoding to low-cardinality categorical columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to label encode.

    Returns:
        pd.DataFrame: The DataFrame with label-encoded columns.

    Description:
        Encodes low-cardinality categorical columns using integer labels.
    """
    logger.info(f"Applying label encoding to columns: {columns}.")
    encoder = LabelEncoder()
    for col in columns:
        df[col] = encoder.fit_transform(df[col])
    return df


def frequency_encoding(df, column):
    """
    Apply frequency encoding to a high-cardinality categorical column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to frequency encode.

    Returns:
        pd.DataFrame: The DataFrame with the frequency-encoded column.

    Description:
        Replaces each category in the column with its frequency of occurrence in the dataset.
    """
    logger.info(f"Applying frequency encoding to column: {column}.")
    freq_map = df[column].value_counts() / len(df)
    df[column] = df[column].map(freq_map)
    return df


def scale_columns(df, columns):
    """
    Scale numerical columns to standardize their values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numerical columns to scale.

    Returns:
        pd.DataFrame: The DataFrame with scaled numerical columns.

    Description:
        Normalizes numerical columns to have a mean of 0 and a standard deviation of 1.
    """
    logger.info(f"Scaling numerical columns: {columns}.")
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


# Main feature engineering pipeline
def feature_engineering_pipeline(df):
    """
    Orchestrate all feature engineering steps for the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with all engineered features.

    Description:
        Applies feature engineering steps in sequence, including datetime conversion,
        feature extraction, encoding, and scaling.
    """
    logger.info("Starting the feature engineering pipeline.")

    # Step 1: Convert date columns to datetime format
    df = convert_to_datetime(df, ['created_date', 'closed_date', 'resolution_action_updated_date'])

    # Step 2: Calculate response time
    df = calculate_response_time(df, 'created_date', 'closed_date', 'response_time')

    # Step 3: Extract date features
    df = extract_date_features(df, 'created_date', 'created')
    df = extract_date_features(df, 'closed_date', 'closed')

    # Step 4: Create weekday feature
    df = create_weekday_feature(df, 'created_date', 'response_weekday')

    # Step 5: Create interaction features
    df = create_interaction_features(df, 'borough', 'complaint_type', 'borough_complaint_type')
    df = create_interaction_features(df, 'created_hour', 'response_weekday', 'hour_weekday_interaction')

    # Step 6: Calculate time to update
    df['time_to_update'] = (df['resolution_action_updated_date'] - df['created_date']).dt.total_seconds() / 3600
    df['time_to_update'].fillna('Not Applicable', inplace=True)

    # Step 7: Drop redundant columns
    df = drop_columns(df, ['resolution_action_updated_date', 'incident_address'])

    # Step 8: Label encoding
    label_columns = ['agency_name', 'address_type', 'status', 'location_type', 'park_borough', 'complaint_type', 'descriptor']
    df = label_encode_columns(df, label_columns)

    # Step 9: One-hot encoding for interaction
    df = pd.get_dummies(df, columns=['hour_weekday_interaction'], drop_first=True)

    # Step 10: Frequency encoding for high-cardinality columns
    freq_columns = ['incident_zip', 'landmark', 'community_board', 'borough', 'open_data_channel_type', 'park_facility_name', 'borough_complaint_type']
    for col in freq_columns:
        df = frequency_encoding(df, col)

    # Step 11: Scale numerical features
    df = scale_columns(df, ['latitude', 'longitude', 'response_time'])

    logger.info("Feature engineering pipeline completed successfully.")
    return df