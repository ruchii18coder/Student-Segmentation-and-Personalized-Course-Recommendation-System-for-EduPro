# src/clustering.py

import os
import pickle

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):
    """
    Load student data from CSV.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)


# ============================================================
# SELECT CLUSTERING FEATURES
# ============================================================

def select_clustering_features(df):
    """
    Select numerical features for student clustering.
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
            "At least two numerical features are required "
            "for clustering."
        )

    X = df[available_features].copy()

    # Handle missing values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    return X


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(X):
    """
    Standardize clustering features.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

def perform_kmeans(
    X_scaled,
    n_clusters=4,
    random_state=42
):
    """
    Apply K-Means clustering.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    return model, labels


# ============================================================
# SILHOUETTE SCORE
# ============================================================

def calculate_silhouette_score(
    X_scaled,
    labels
):
    """
    Calculate clustering quality.
    """

    unique_labels = np.unique(
        labels
    )

    if len(unique_labels) < 2:
        return 0

    return silhouette_score(
        X_scaled,
        labels
    )


# ============================================================
# PCA
# ============================================================

def apply_pca(X_scaled):
    """
    Reduce clustering features to two dimensions.
    """

    pca = PCA(
        n_components=2,
        random_state=42
    )

    components = pca.fit_transform(
        X_scaled
    )

    pca_df = pd.DataFrame(
        components,
        columns=[
            "PC1",
            "PC2"
        ]
    )

    return pca_df, pca


# ============================================================
# NAME STUDENT CLUSTERS
# ============================================================

def assign_cluster_names(
    df,
    cluster_column="cluster"
):
    """
    Assign meaningful names to clusters based
    on student performance and engagement.
    """

    result = df.copy()

    if cluster_column not in result.columns:
        return result

    cluster_names = {}

    for cluster in sorted(
        result[cluster_column].unique()
    ):

        cluster_data = result[
            result[cluster_column] == cluster
        ]

        # Default values
        avg_score = (
            cluster_data["average_score"].mean()
            if "average_score" in cluster_data.columns
            else 0
        )

        completion = (
            cluster_data[
                "course_completion_rate"
            ].mean()
            if "course_completion_rate"
            in cluster_data.columns
            else 0
        )

        study_hours = (
            cluster_data["study_hours"].mean()
            if "study_hours" in cluster_data.columns
            else 0
        )

        engagement = (
            cluster_data["engagement_score"].mean()
            if "engagement_score"
            in cluster_data.columns
            else 0
        )

        # Segment logic
        if (
            avg_score >= 80
            and completion >= 75
        ):

            name = "High Performing Students"

        elif (
            study_hours >= 6
            or engagement >= 70
        ):

            name = "Highly Engaged Students"

        elif (
            avg_score < 55
            or completion < 45
        ):

            name = "Students Needing Support"

        else:

            name = "Moderately Engaged Students"

        cluster_names[cluster] = name

    result["segment"] = result[
        cluster_column
    ].map(cluster_names)

    return result


# ============================================================
# COMPLETE CLUSTERING PIPELINE
# ============================================================

def create_student_segments(
    df,
    n_clusters=4
):
    """
    Complete student segmentation pipeline.

    Returns:
        segmented_data
        kmeans_model
        scaler
        pca_model
        silhouette
    """

    data = df.copy()

    # Select features
    X = select_clustering_features(
        data
    )

    # Scale
    X_scaled, scaler = scale_features(
        X
    )

    # K-Means
    kmeans, labels = perform_kmeans(
        X_scaled,
        n_clusters=n_clusters
    )

    # Add cluster labels
    data["cluster"] = labels

    # Add segment names
    data = assign_cluster_names(
        data,
        "cluster"
    )

    # PCA
    pca_df, pca = apply_pca(
        X_scaled
    )

    data["PC1"] = pca_df["PC1"]
    data["PC2"] = pca_df["PC2"]

    # Silhouette score
    silhouette = calculate_silhouette_score(
        X_scaled,
        labels
    )

    return (
        data,
        kmeans,
        scaler,
        pca,
        silhouette
    )


# ============================================================
# CLUSTER SUMMARY
# ============================================================

def generate_cluster_summary(
    segmented_data
):
    """
    Generate summary statistics for each segment.
    """

    aggregation = {}

    if "student_id" in segmented_data.columns:
        aggregation["student_id"] = "count"

    if "average_score" in segmented_data.columns:
        aggregation["average_score"] = "mean"

    if "course_completion_rate" in segmented_data.columns:
        aggregation[
            "course_completion_rate"
        ] = "mean"

    if "study_hours" in segmented_data.columns:
        aggregation["study_hours"] = "mean"

    if "engagement_score" in segmented_data.columns:
        aggregation["engagement_score"] = "mean"

    if not aggregation:
        return pd.DataFrame()

    summary = segmented_data.groupby(
        ["cluster", "segment"]
    ).agg(
        aggregation
    ).reset_index()

    # Rename student count
    if "student_id" in summary.columns:
        summary = summary.rename(
            columns={
                "student_id": "student_count"
            }
        )

    # Round numeric values
    numeric_columns = summary.select_dtypes(
        include=np.number
    ).columns

    summary[numeric_columns] = (
        summary[numeric_columns].round(2)
    )

    return summary


# ============================================================
# SAVE MODELS
# ============================================================

def save_model(model, file_path):
    """
    Save a trained model using pickle.
    """

    directory = os.path.dirname(
        file_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        file_path,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DATA_PATH = "data/students.csv"

    MODEL_DIR = "models"

    KMEANS_PATH = os.path.join(
        MODEL_DIR,
        "kmeans.pkl"
    )

    SCALER_PATH = os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )

    PCA_PATH = os.path.join(
        MODEL_DIR,
        "pca.pkl"
    )

    try:

        print("--------------------------------")
        print("EduPro Student Segmentation")
        print("--------------------------------")

        # Load dataset
        students = load_data(
            DATA_PATH
        )

        print(
            f"Dataset shape: "
            f"{students.shape}"
        )

        # Feature engineering
        try:

            from feature_engineering import (
                engineer_features
            )

            students = engineer_features(
                students
            )

            print(
                "Feature engineering completed."
            )

        except ImportError:

            print(
                "Feature engineering module "
                "not available. Using original data."
            )

        # Create segments
        (
            segmented,
            kmeans,
            scaler,
            pca,
            silhouette
        ) = create_student_segments(
            students,
            n_clusters=4
        )

        print("\n--------------------------------")
        print("Clustering Completed")
        print("--------------------------------")

        print(
            f"Silhouette Score: "
            f"{silhouette:.4f}"
        )

        # Summary
        summary = generate_cluster_summary(
            segmented
        )

        print("\nCluster Summary:")
        print(summary)

        # Save models
        save_model(
            kmeans,
            KMEANS_PATH
        )

        save_model(
            scaler,
            SCALER_PATH
        )

        save_model(
            pca,
            PCA_PATH
        )

        # Save segmented dataset
        os.makedirs(
            "data",
            exist_ok=True
        )

        segmented.to_csv(
            "data/segmented_students.csv",
            index=False
        )

        print("\n--------------------------------")
        print("Models saved successfully!")
        print("--------------------------------")

        print(
            f"K-Means: {KMEANS_PATH}"
        )

        print(
            f"Scaler: {SCALER_PATH}"
        )

        print(
            f"PCA: {PCA_PATH}"
        )

        print(
            "Segmented data: "
            "data/segmented_students.csv"
        )

    except Exception as e:

        print(
            f"Clustering error: {e}"
        )
