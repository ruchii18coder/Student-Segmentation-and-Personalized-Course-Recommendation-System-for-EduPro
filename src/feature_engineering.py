# src/feature_engineering.py

import pandas as pd
import numpy as np


# ============================================================
# PERFORMANCE FEATURES
# ============================================================

def create_performance_features(df):
    """
    Create student performance-related features.
    """

    data = df.copy()

    if "average_score" in data.columns and "quiz_score" in data.columns:

        data["overall_performance"] = (
            data["average_score"] * 0.6
            + data["quiz_score"] * 0.4
        )

    elif "average_score" in data.columns:

        data["overall_performance"] = (
            data["average_score"]
        )

    return data


# ============================================================
# ENGAGEMENT FEATURES
# ============================================================

def create_engagement_features(df):
    """
    Create student engagement-related features.
    """

    data = df.copy()

    if (
        "study_hours" in data.columns
        and "login_frequency" in data.columns
    ):

        data["engagement_score"] = (
            data["study_hours"] * 5
            + data["login_frequency"] * 2
        )

    elif "study_hours" in data.columns:

        data["engagement_score"] = (
            data["study_hours"] * 10
        )

    return data


# ============================================================
# COMPLETION FEATURES
# ============================================================

def create_completion_features(df):
    """
    Create course completion-related features.
    """

    data = df.copy()

    if "course_completion_rate" in data.columns:

        data["completion_level"] = pd.cut(
            data["course_completion_rate"],
            bins=[
                -np.inf,
                40,
                70,
                90,
                np.inf
            ],
            labels=[
                "Low",
                "Medium",
                "High",
                "Excellent"
            ]
        )

    return data


# ============================================================
# LEARNING LEVEL
# ============================================================

def create_learning_level(df):
    """
    Assign a learning level based on student performance.
    """

    data = df.copy()

    if (
        "average_score" not in data.columns
        or "course_completion_rate"
        not in data.columns
    ):
        return data

    def determine_level(row):

        score = row["average_score"]
        completion = row["course_completion_rate"]

        if score >= 80 and completion >= 75:
            return "Advanced"

        elif score >= 60 and completion >= 50:
            return "Intermediate"

        return "Beginner"

    data["learning_level"] = data.apply(
        determine_level,
        axis=1
    )

    return data


# ============================================================
# ACTIVITY LEVEL
# ============================================================

def create_activity_level(df):
    """
    Categorize students according to their activity.
    """

    data = df.copy()

    if "engagement_score" not in data.columns:
        return data

    data["activity_level"] = pd.cut(
        data["engagement_score"],
        bins=[
            -np.inf,
            30,
            60,
            100,
            np.inf
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]
    )

    return data


# ============================================================
# SCORE CATEGORIES
# ============================================================

def create_score_category(df):
    """
    Convert numerical scores into performance categories.
    """

    data = df.copy()

    if "average_score" not in data.columns:
        return data

    data["score_category"] = pd.cut(
        data["average_score"],
        bins=[
            -np.inf,
            40,
            60,
            75,
            90,
            np.inf
        ],
        labels=[
            "Poor",
            "Below Average",
            "Average",
            "Good",
            "Excellent"
        ]
    )

    return data


# ============================================================
# STUDY INTENSITY
# ============================================================

def create_study_intensity(df):
    """
    Create a study intensity feature.
    """

    data = df.copy()

    if (
        "study_hours" in data.columns
        and "courses_enrolled" in data.columns
    ):

        courses = data["courses_enrolled"].replace(
            0,
            1
        )

        data["study_intensity"] = (
            data["study_hours"] / courses
        )

    return data


# ============================================================
# ASSIGNMENT PERFORMANCE
# ============================================================

def create_assignment_feature(df):
    """
    Create assignment-related features.
    """

    data = df.copy()

    if "assignments_completed" not in data.columns:
        return data

    data["assignment_activity"] = np.log1p(
        data["assignments_completed"]
    )

    return data


# ============================================================
# NORMALIZED PERFORMANCE SCORE
# ============================================================

def create_normalized_performance(df):
    """
    Create a normalized performance score between 0 and 1.
    """

    data = df.copy()

    if "overall_performance" not in data.columns:
        return data

    min_score = data[
        "overall_performance"
    ].min()

    max_score = data[
        "overall_performance"
    ].max()

    if max_score == min_score:

        data["normalized_performance"] = 0.5

    else:

        data["normalized_performance"] = (
            data["overall_performance"] - min_score
        ) / (
            max_score - min_score
        )

    return data


# ============================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================

def engineer_features(df):
    """
    Apply all feature engineering steps.
    """

    data = df.copy()

    # Performance
    data = create_performance_features(
        data
    )

    # Engagement
    data = create_engagement_features(
        data
    )

    # Completion
    data = create_completion_features(
        data
    )

    # Learning level
    data = create_learning_level(
        data
    )

    # Activity
    data = create_activity_level(
        data
    )

    # Score category
    data = create_score_category(
        data
    )

    # Study intensity
    data = create_study_intensity(
        data
    )

    # Assignment activity
    data = create_assignment_feature(
        data
    )

    # Normalized performance
    data = create_normalized_performance(
        data
    )

    return data


# ============================================================
# GET ML FEATURES
# ============================================================

def get_ml_features(df):
    """
    Return numerical features suitable for ML models.
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

    return df[
        available_features
    ].copy()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DATA_PATH = "data/students.csv"

    try:

        students = pd.read_csv(
            DATA_PATH
        )

        print("--------------------------------")
        print("Original Dataset")
        print("--------------------------------")

        print(
            students.head()
        )

        engineered_data = engineer_features(
            students
        )

        print("\n--------------------------------")
        print("Feature Engineering Completed")
        print("--------------------------------")

        print(
            f"Dataset Shape: "
            f"{engineered_data.shape}"
        )

        print("\nNew Features:")

        original_columns = set(
            students.columns
        )

        new_columns = [
            column
            for column in engineered_data.columns
            if column not in original_columns
        ]

        print(new_columns)

        print("\nEngineered Dataset:")

        print(
            engineered_data.head()
        )

    except FileNotFoundError:

        print(
            f"Dataset not found: {DATA_PATH}"
        )

    except Exception as e:

        print(
            f"Feature engineering error: {e}"
        )
