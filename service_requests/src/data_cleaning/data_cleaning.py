import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="data_cleaning.log",  # Log to a file
    filemode="w"  # Overwrite log file each time
)

logger = logging.getLogger(__name__)

def handle_missing_values(df):
    """
    Drops data from columns with over 80% missing values

    Args:
        df (Dataframe): The data that we are working with.

    Returns:
        dataframe: The processed dataframe after dropping.
    """
    
    # Drop columns with over 80% missing values using an advanced approach with thresholding
    missing_threshold = 0.8
    missing_percentages = df.isnull().mean()  # Calculate missing percentages
    columns_to_drop_advanced = missing_percentages[missing_percentages > missing_threshold].index.tolist()

    # Drop the identified columns
    df = df.drop(columns=columns_to_drop_advanced)

    return df

def clean_closed_date_column(closed_date):
    """
    Clean the 'closed_date' column by converting to datetime.

    Args:
        closed_date (pd.Series): The 'closed_date' column.

    Returns:
        pd.Series: The cleaned 'closed_date' column.
    """
    logger.info("Cleaning 'closed_date' column.")
    before_missing = closed_date.isnull().sum()
    cleaned = pd.to_datetime(closed_date, errors='coerce')
    after_missing = cleaned.isnull().sum()
    logger.info(f"'closed_date': {before_missing} missing values before, {after_missing} after cleaning.")
    return cleaned


def clean_lat_lon_columns(latitude, longitude, incident_zip):
    """
    Clean 'latitude' and 'longitude' columns by coercing to numeric
    and filling missing values using a mapping.

    Args:
        latitude (pd.Series): The 'latitude' column.
        longitude (pd.Series): The 'longitude' column.
        incident_zip (pd.Series): The 'incident_zip' column.

    Returns:
        pd.Series, pd.Series: Cleaned 'latitude' and 'longitude' columns.
    """
    logger.info("Cleaning 'latitude' and 'longitude' columns.")

    # Coerce latitude and longitude to numeric
    latitude = pd.to_numeric(latitude, errors='coerce')
    longitude = pd.to_numeric(longitude, errors='coerce')

    # Create mapping for 'incident_zip' based on cleaned latitude and longitude
    zip_mapping = (
        pd.DataFrame({'latitude': latitude, 'longitude': longitude, 'incident_zip': incident_zip})
        .groupby('incident_zip')[['latitude', 'longitude']]
        .mean()
        .dropna()
        .to_dict('index')
    )
    logger.info(f"Created ZIP mapping for {len(zip_mapping)} ZIP codes.")

    # Fill missing values using the ZIP mapping
    mapped_latitude = incident_zip.map(lambda x: zip_mapping.get(x, {}).get('latitude'))
    mapped_longitude = incident_zip.map(lambda x: zip_mapping.get(x, {}).get('longitude'))

    latitude = latitude.fillna(mapped_latitude)
    longitude = longitude.fillna(mapped_longitude)

    lat_missing_after = latitude.isnull().sum()
    lon_missing_after = longitude.isnull().sum()

    logger.info(f"'latitude': Missing values after cleaning: {lat_missing_after}.")
    logger.info(f"'longitude': Missing values after cleaning: {lon_missing_after}.")

    return latitude, longitude


def clean_categorical_column(column, df, group_columns, fill_value="Unknown"):
    """
    Fill missing values in a categorical column using grouped mode.

    Args:
        column (pd.Series): The categorical column to clean.
        df (pd.DataFrame): The DataFrame containing the column and grouping columns.
        group_columns (list): Columns to group by for filling missing values.
        fill_value (str): Value to fill if no mode is available.

    Returns:
        pd.Series: Cleaned column with missing values filled.
    """
    logger.info(f"Cleaning categorical column '{column.name}' using grouping on {group_columns}.")
    
    # Group by the specified columns and calculate the mode for the target column
    grouped = (
        df[group_columns + [column.name]]
        .groupby(group_columns)[column.name]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else fill_value)
    )

    # Map the grouped mode values to the rows
    column_filled = column.copy()
    column_filled = column_filled.fillna(df[group_columns].apply(tuple, axis=1).map(grouped.to_dict()))

    missing_after = column_filled.isnull().sum()
    logger.info(f"'{column.name}': Missing values after cleaning: {missing_after}.")
    return column_filled

def clean_address_columns(incident_address, street_name, landmark):
    """Clean 'incident_address' and related address fields."""
    logger.info("Cleaning 'incident_address' and related address fields.")
    incident_address = incident_address.fillna(street_name)
    landmark = landmark.fillna("Not Applicable")
    return incident_address, landmark

def clean_bbl_column(bbl, incident_zip):
    """Clean and impute 'bbl' column."""
    logger.info("Cleaning 'bbl' column.")
    bbl = pd.to_numeric(bbl, errors='coerce')
    bbl_mapping = (
        pd.DataFrame({'bbl': bbl, 'incident_zip': incident_zip})
        .groupby('incident_zip')['bbl']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    )
    bbl = bbl.fillna(incident_zip.map(bbl_mapping))
    missing_after = bbl.isnull().sum()
    logger.info(f"'bbl': Missing values after cleaning: {missing_after}.")
    return bbl

def drop_columns_with_high_missing_values(df, columns_to_drop):
    """Drop columns with excessive missing values."""
    logger.info(f"Dropping columns with high missing values: {columns_to_drop}.")
    return df.drop(columns=columns_to_drop, inplace=False)

def drop_redundant_location_columns(df, columns_to_drop):
    """Drop redundant location-related columns."""
    logger.info(f"Dropping redundant location columns: {columns_to_drop}.")
    return df.drop(columns=columns_to_drop, inplace=False)

def clean_remaining_columns(df):
    """Clean miscellaneous remaining columns."""
    logger.info("Cleaning remaining columns.")
    df['address_type'] = df['address_type'].fillna('Unknown')
    df['status'] = df['status'].fillna('Unknown')
    df['park_borough'] = df['park_borough'].fillna('Unknown')
    df['park_facility_name'] = df['park_facility_name'].fillna('Unknown')
    df['open_data_channel_type'] = df['open_data_channel_type'].fillna('Unknown')
    return df


# Main cleaning pipeline
def main_cleaning_pipeline(df):
    """
    Orchestrate all cleaning steps for the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    logger.info("Starting the main data cleaning pipeline.")

    # Drop columns with over 80% missing values
    df = handle_missing_values(df)

    # Clean 'closed_date'
    df['closed_date'] = clean_closed_date_column(df['closed_date'])
    df = df.dropna(subset=['closed_date'])

    # Clean 'latitude' and 'longitude'
    df['latitude'], df['longitude'] = clean_lat_lon_columns(df['latitude'], df['longitude'], df['incident_zip'])
    df = df.dropna(subset=['latitude', 'longitude'])

    # Clean 'descriptor', 'location_type', and 'city'
    df['descriptor'] = clean_categorical_column(
        df['descriptor'], df, ['complaint_type', 'location_type']
    )
    df['location_type'] = clean_categorical_column(
        df['location_type'], df, ['complaint_type', 'descriptor']
    )

    # Drop the 'City' column
    df.drop(columns=['city'], inplace=True)

    # Clean address-related columns
    df['incident_address'], df['landmark'] = clean_address_columns(df['incident_address'], df['street_name'], df['landmark'])
    df = df.drop(columns=['street_name'])
    df = df.dropna(subset=['incident_address'])

    # Clean 'bbl'
    df['bbl'] = clean_bbl_column(df['bbl'], df['incident_zip'])
    df = df.dropna(subset=['bbl'])

    df = df.dropna(subset=["resolution_action_updated_date"])
    df = df.dropna(subset=["incident_zip"])

    # Drop columns with excessive missing values
    df = drop_columns_with_high_missing_values(df, ['taxi_pick_up_location', 'vehicle_type'])

    # Drop redundant location columns
    df = drop_redundant_location_columns(df, ['intersection_street_1', 'intersection_street_2', 'cross_street_1', 'cross_street_2'])

    # Drop 'x_coordinate_state_plane' and 'y_coordinate_state_plane'
    logger.info("Dropping redundant columns: 'x_coordinate_state_plane', 'y_coordinate_state_plane'.")
    df = df.drop(columns=['x_coordinate_state_plane', 'y_coordinate_state_plane'])

    # Drop 'agency' and 'location'
    logger.info("Dropping redundant columns: 'agency', 'location'.")
    df = df.drop(columns=['agency', 'location'])

    # Clean remaining columns
    df = clean_remaining_columns(df)
    df = df.dropna(subset=['borough'])
    df = df.dropna(subset=['community_board'])


    logger.info("Data cleaning pipeline completed successfully.")
    return df

