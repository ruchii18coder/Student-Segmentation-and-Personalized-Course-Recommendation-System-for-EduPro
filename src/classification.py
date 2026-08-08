# src/classification.py

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):
    """
    Load student dataset from CSV.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)


# ============================================================
# CREATE TARGET VARIABLE
# ============================================================

def create_learning_level(df):
    """
    Create learning-level target:
        Beginner
        Intermediate
        Advanced

    Target is based on average score and
    course completion rate.
    """

    data = df.copy()

    required_columns = [
        "average_score",
        "course_completion_rate"
    ]

    for column in required_columns:

        if column not in data.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    def determine_level(row):

        score = row["average_score"]
        completion = row["course_completion_rate"]

        if score >= 80 and completion >= 75:
            return "Advanced"

        elif score >= 60 and completion >= 50:
            return "Intermediate"

        else:
            return "Beginner"

    data["learning_level"] = data.apply(
        determine_level,
        axis=1
    )

    return data


# ============================================================
# SELECT FEATURES
# ============================================================

def select_features(df):
    """
    Select numerical features for classification.
    """

    feature_columns = [
        "age",
        "study_hours",
        "average_score",
        "course_completion_rate",
        "quiz_score",
        "assignments_completed",
        "courses_enrolled",
        "login_frequency",
        "overall_performance",
        "engagement_score",
        "study_intensity",
        "assignment_activity",
        "normalized_performance"
    ]

    available_features = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    if len(available_features) < 2:

        raise ValueError(
            "Not enough classification features "
            "available in dataset."
        )

    X = df[
        available_features
    ].copy()

    # Replace infinity values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    X = X.fillna(
        X.median(numeric_only=True)
    )

    return X


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """
    Prepare X and y for classification.
    """

    data = create_learning_level(
        df
    )

    X = select_features(
        data
    )

    y = data[
        "learning_level"
    ]

    return X, y


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train,
    random_state=42
):
    """
    Train Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate classification model.
    """

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return (
        accuracy,
        report,
        matrix,
        predictions
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    feature_names
):
    """
    Return feature importance from Random Forest.
    """

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    return importance


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_names,
    file_path
):
    """
    Save model and feature names.
    """

    directory = os.path.dirname(
        file_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    model_package = {
        "model": model,
        "features": feature_names
    }

    with open(
        file_path,
        "wb"
    ) as file:

        pickle.dump(
            model_package,
            file
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(file_path):
    """
    Load saved Random Forest model.
    """

    if not os.path.exists(
        file_path
    ):
        raise FileNotFoundError(
            f"Model not found: {file_path}"
        )

    with open(
        file_path,
        "rb"
    ) as file:

        return pickle.load(
            file
        )


# ============================================================
# PREDICT LEARNING LEVEL
# ============================================================

def predict_learning_level(
    model_package,
    student_data
):
    """
    Predict learning level for a student.
    """

    model = model_package["model"]
    features = model_package["features"]

    # Convert dictionary to DataFrame
    if isinstance(
        student_data,
        dict
    ):

        student_data = pd.DataFrame(
            [student_data]
        )

    # Add missing features
    for feature in features:

        if feature not in student_data.columns:
            student_data[feature] = 0

    X = student_data[
        features
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    prediction = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    return prediction, probabilities


# ============================================================
# COMPLETE TRAINING PIPELINE
# ============================================================

def train_classifier(
    data_path="data/students.csv",
    model_path="models/random_forest.pkl"
):
    """
    Complete Random Forest training pipeline.
    """

    # Load data
    students = load_data(
        data_path
    )

    # Try feature engineering
    try:

        from feature_engineering import (
            engineer_features
        )

        students = engineer_features(
            students
        )

    except ImportError:

        print(
            "Feature engineering module "
            "not found. Continuing with raw features."
        )

    # Prepare X and y
    X, y = prepare_data(
        students
    )

    # Check target classes
    if y.nunique() < 2:

        raise ValueError(
            "Classification requires at least "
            "two different learning-level classes."
        )

    # Train/test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # Train model
    model = train_random_forest(
        X_train,
        y_train
    )

    # Evaluate
    (
        accuracy,
        report,
        matrix,
        predictions
    ) = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n--------------------------------")
    print("Random Forest Evaluation")
    print("--------------------------------")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(matrix)

    # Feature importance
    importance = get_feature_importance(
        model,
        X.columns
    )

    print("\nFeature Importance:")
    print(importance)

    # Save model
    save_model(
        model,
        list(X.columns),
        model_path
    )

    print("\n--------------------------------")
    print("Model saved successfully!")
    print("--------------------------------")

    print(
        f"Model path: {model_path}"
    )

    return (
        model,
        accuracy,
        importance
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DATA_PATH = "data/students.csv"

    MODEL_PATH = (
        "models/random_forest.pkl"
    )

    try:

        train_classifier(
            data_path=DATA_PATH,
            model_path=MODEL_PATH
        )

    except Exception as e:

        print(
            f"\nClassification error: {e}"
        )