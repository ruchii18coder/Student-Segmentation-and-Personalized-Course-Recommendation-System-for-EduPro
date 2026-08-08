# src/preprocessing.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


# ============================================================
# LOAD DATA
# ============================================================

def load_student_data(file_path):
    """
    Load student data from CSV file.
    """

    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully: {df.shape}")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )


# ============================================================
# CLEAN DATA
# ============================================================

def clean_student_data(df):
    """
    Clean student dataset.
    """

    data = df.copy()

    # Remove duplicate records
    data = data.drop_duplicates()

    # Convert numerical columns
    numerical_columns = [
        "age",
        "study_hours",
        "average_score",
        "course_completion_rate",
        "quiz_score",
        "assignments_completed",
        "courses_enrolled",
        "login_frequency"
    ]

    for column in numerical_columns:

        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # Fill missing numerical values
    for column in numerical_columns:

        if column in data.columns:
            data[column] = data[column].fillna(
                data[column].median()
            )

    return data


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(df):
    """
    Select features used for Machine Learning.
    """

    feature_columns = [
        "age",
        "study_hours",
        "average_score",
        "course_completion_rate",
        "quiz_score",
        "assignments_completed",
        "courses_enrolled",
        "login_frequency"
    ]

    available_features = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    if len(available_features) < 2:

        raise ValueError(
            "Not enough ML features found in dataset."
        )

    return df[available_features].copy()


# ============================================================
# FEATURE SCALING
# ============================================================

def scale_features(X):
    """
    Standardize numerical features.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )

    return X_scaled, scaler


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_students(file_path):
    """
    Complete preprocessing pipeline.

    Returns:
        cleaned_data
        selected_features
        scaled_features
        scaler
    """

    # Load
    df = load_student_data(file_path)

    # Clean
    cleaned_data = clean_student_data(df)

    # Select features
    features = select_features(
        cleaned_data
    )

    # Scale
    scaled_features, scaler = scale_features(
        features
    )

    return (
        cleaned_data,
        features,
        scaled_features,
        scaler
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DATA_PATH = "data/students.csv"

    try:

        (
            cleaned_data,
            features,
            scaled_features,
            scaler
        ) = preprocess_students(DATA_PATH)

        print("\n--------------------------------")
        print("Preprocessing completed!")
        print("--------------------------------")

        print(
            f"Original/Cleaned shape: "
            f"{cleaned_data.shape}"
        )

        print(
            f"Selected features: "
            f"{list(features.columns)}"
        )

        print(
            f"Scaled data shape: "
            f"{scaled_features.shape}"
        )

        print("\nScaled Feature Sample:")
        print(
            scaled_features.head()
        )

    except Exception as e:

        print(
            f"Preprocessing error: {e}"
        )
