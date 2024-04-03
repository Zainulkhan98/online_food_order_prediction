


from typing import Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from zenml.client import Client
import mlflow.sklearn
import mlflow

from zenml import (step, pipeline)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    data = pd.read_csv(data_path)
    return data


@step
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Performs preprocessing on the data."""
    try:
        # Label encoding for the target column
        le = LabelEncoder()
        data['Output'] = le.fit_transform(data['Output'])

        # One-hot encoding for categorical columns
        categorical_cols = ['Gender', 'Marital Status', 'Family size', 'unknown','Educational Qualifications', 'Feedback', 'Occupation', 'Monthly Income',
                            'Pin code', 'longitude']
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cols = pd.DataFrame(ohe.fit_transform(data[categorical_cols]))
        encoded_cols.columns = ohe.get_feature_names_out(categorical_cols)

        # Drop original categorical columns and concatenate encoded columns
        data = data.drop(columns=categorical_cols)
        data = pd.concat([data, encoded_cols], axis=1)

    except Exception as e:
        print(f"Error during preprocessing: {e}")
    return data


@step
def split(data: pd.DataFrame, test_size: float, target_column: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = data.drop([target_column], axis=1)
    y = data[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    return x_train, x_test, y_train, y_test

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker = experiment_tracker.name)
def modeling(x_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Trains a logistic regression model."""
    model = LogisticRegression(max_iter=10000)
    model.fit(x_train, y_train)
    mlflow.sklearn.autolog(model)
    return model


@step
def predict(model: LogisticRegression, x_test: pd.DataFrame) -> pd.Series:
    """Makes predictions using the trained model."""
    pred = model.predict(x_test)
    predictions = pd.Series(pred)
    return predictions


@step(experiment_tracker= experiment_tracker.name)
def evaluate(predictions: pd.Series, y_test: pd.Series) -> float:
    """Evaluates model performance using accuracy."""
    accu = accuracy_score(y_test, predictions)
    accuracy = float(accu)
    mlflow.log_metric("accuracy", accuracy)
    return accuracy


@pipeline
def ml_pipeline(data_path: str) -> float:
    """Full ML pipeline."""
    print("Ingesting data...")
    data = ingest_data(data_path=data_path)
    print("Preprocessing data...")
    processed_data = preprocess_data(data)
    print('Splitting data...')
    x_train, x_test, y_train, y_test = split(processed_data, test_size=0.2, target_column='Output')
    print('Modeling data...')
    model = modeling(x_train, y_train)
    print('Predicting data...')
    predictions = predict(model, x_test)
    print('Evaluating model...')
    accuracy = evaluate(predictions, y_test)
    print('Done!')
    return accuracy


if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    accuracy = ml_pipeline(data_path='onlinefoods.csv')
    print(accuracy)


