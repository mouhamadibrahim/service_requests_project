import pandas as pd
import numpy as np
import json
import yaml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from joblib import dump
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="train_model.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

# Data Preparation
def prepare_data(df):
    """
    Prepare the dataset for modeling by handling missing values and converting types.
    """
    logger.info("Preparing data: Handling missing values and type conversions.")
    df['time_to_update'] = df['time_to_update'].astype(float)
    return df

# Train-Test Split
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    logger.info(f"Splitting data into train and test sets with test_size={test_size}.")
    X = df.drop(columns=[target_column, 'unique_key', 'created_date', 'closed_date'])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train XGBoost Model
def train_model(X_train, y_train, model_params):
    """
    Train an XGBoost model with given parameters.
    """
    logger.info(f"Training XGBoost model with parameters: {model_params}.")
    model = XGBRegressor(**model_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# Hyperparameter Tuning with Randomized Search
def tune_hyperparameters(X_train, y_train, param_dist):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    """
    logger.info("Starting hyperparameter tuning with RandomizedSearchCV.")
    model = XGBRegressor(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    logger.info(f"Best hyperparameters found: {best_params}")
    return best_params

# Feature Importance
def get_feature_importance(model, feature_names):
    """
    Extract feature importances from the trained XGBoost model.
    """
    logger.info("Extracting feature importances from the model.")
    importance = model.feature_importances_
    return pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values(by="Importance", ascending=False)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    logger.info("Evaluating model on the test set.")
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }
    logger.info(f"Model performance: {metrics}")
    return metrics

# Main Training Pipeline
def training_pipeline(df, param_dist):
    """
    Full pipeline for data preparation, model training, and hyperparameter tuning.
    """
    # Data Preparation
    df = prepare_data(df)

    # Train-Test Split
    X_train, X_test, y_train, y_test = split_data(df, 'response_time')

    # Initial Training
    initial_params = {
        "n_estimators": 50,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    model = train_model(X_train, y_train, initial_params)

    # Hyperparameter Tuning
    best_params = tune_hyperparameters(X_train, y_train, param_dist)

    # Retrain with Best Parameters
    model = train_model(X_train, y_train, best_params)

    # Save Feature Importance
    feature_importance = get_feature_importance(model, X_train.columns)
    feature_importance.to_csv("feature_importances.csv", index=False)
    logger.info("Saved feature importances to 'feature_importances.csv'.")

    # Step 7: Model Evaluation
    metrics = evaluate_model(model, X_test, y_test)

    # Save Best Parameters
    with open("params/best_params.yaml", "w") as f:
        yaml.dump(best_params, f)
    logger.info("Saved best hyperparameters to 'best_params.yaml'.")

    with open("params/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved evaluation metrics to 'metrics.json'.")

    # Save the Final Model
    dump(model, "params/final_model.joblib")
    logger.info("Saved the final model to 'final_model.joblib'.")

    return {"Best Parameters": best_params, "Model": model,"Metrics": metrics}