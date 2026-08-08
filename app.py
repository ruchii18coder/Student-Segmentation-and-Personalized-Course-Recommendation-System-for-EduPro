# app.py
# EduPro - Student Segmentation & Personalized Course Recommendation System
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduPro - Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .title {
            font-size: 42px;
            font-weight: 700;
            color: #17324d;
        }

        .subtitle {
            font-size: 18px;
            color: #607080;
        }

        .card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 15px;
        }

        .recommendation {
            background: #eef6ff;
            border-left: 5px solid #2196f3;
            padding: 15px;
            border-radius: 8px;
            margin: 8px 0;
        }

        .success-box {
            background: #ecfdf3;
            border-left: 5px solid #16a34a;
            padding: 15px;
            border-radius: 8px;
        }

        .warning-box {
            background: #fff8e6;
            border-left: 5px solid #f59e0b;
            padding: 15px;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = "data/students.csv"
COURSE_PATH = "data/courses.csv"

MODEL_DIR = "models"

KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_pickle(path):
    """Load a pickle model if it exists."""
    if os.path.exists(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    return None


@st.cache_data
def load_csv(path):
    """Load CSV data."""
    if os.path.exists(path):
        return pd.read_csv(path)

    return None


def create_demo_students():
    """
    Creates demo data so that the dashboard can run even
    when the actual dataset is not available.
    """

    np.random.seed(42)

    n = 500

    data = pd.DataFrame({
        "student_id": range(1, n + 1),
        "age": np.random.randint(17, 35, n),
        "study_hours": np.round(np.random.uniform(1, 10, n), 2),
        "course_completion_rate": np.round(
            np.random.uniform(20, 100, n), 2
        ),
        "average_score": np.round(
            np.random.uniform(35, 100, n), 2
        ),
        "quiz_score": np.round(
            np.random.uniform(30, 100, n), 2
        ),
        "assignments_completed": np.random.randint(
            0, 20, n
        ),
        "courses_enrolled": np.random.randint(
            1, 8, n
        ),
        "login_frequency": np.random.randint(
            1, 30, n
        )
    })

    return data


def create_demo_courses():
    """Creates demo course catalogue."""

    courses = pd.DataFrame({
        "course_id": range(1, 11),

        "course_name": [
            "Python for Beginners",
            "Advanced Python",
            "Machine Learning",
            "Deep Learning",
            "Data Science Fundamentals",
            "SQL & Database Management",
            "Web Development",
            "Cloud Computing",
            "Data Visualization",
            "Artificial Intelligence"
        ],

        "category": [
            "Programming",
            "Programming",
            "Machine Learning",
            "Deep Learning",
            "Data Science",
            "Database",
            "Web Development",
            "Cloud",
            "Data Science",
            "Artificial Intelligence"
        ],

        "difficulty": [
            "Beginner",
            "Advanced",
            "Intermediate",
            "Advanced",
            "Beginner",
            "Intermediate",
            "Beginner",
            "Intermediate",
            "Beginner",
            "Advanced"
        ],

        "duration_hours": [
            20, 35, 40, 50, 25,
            30, 35, 40, 20, 45
        ],

        "rating": [
            4.6, 4.7, 4.8, 4.7, 4.5,
            4.6, 4.4, 4.5, 4.7, 4.8
        ]
    })

    return courses


def get_student_features(df):
    """
    Select numerical features used for segmentation.
    """

    possible_features = [
        "age",
        "study_hours",
        "course_completion_rate",
        "average_score",
        "quiz_score",
        "assignments_completed",
        "courses_enrolled",
        "login_frequency"
    ]

    features = [
        column for column in possible_features
        if column in df.columns
    ]

    return features


def perform_clustering(df, features, n_clusters=4):
    """
    Perform K-Means clustering.
    """

    X = df[features].copy()

    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    result = df.copy()
    result["cluster"] = clusters

    return result, scaler, kmeans, X_scaled


def get_cluster_name(cluster_data):
    """
    Generate a human-readable name for a student segment.
    """

    score = cluster_data["average_score"].mean()
    completion = cluster_data["course_completion_rate"].mean()
    study = cluster_data["study_hours"].mean()

    if score >= 75 and completion >= 70:
        return "High Performing Students"

    elif study >= 6:
        return "Highly Engaged Students"

    elif score < 55 or completion < 45:
        return "Students Needing Support"

    else:
        return "Moderately Engaged Students"


def recommend_courses(student, courses, top_n=5):
    """
    Recommend courses based on student performance and engagement.
    """

    score = float(student.get("average_score", 60))
    completion = float(
        student.get("course_completion_rate", 60)
    )
    study_hours = float(
        student.get("study_hours", 5)
    )

    # Determine preferred difficulty
    if score < 55:
        preferred_difficulty = "Beginner"

    elif score < 75:
        preferred_difficulty = "Intermediate"

    else:
        preferred_difficulty = "Advanced"

    recommendations = courses.copy()

    recommendations["match_score"] = 0.0

    # Difficulty match
    recommendations.loc[
        recommendations["difficulty"] == preferred_difficulty,
        "match_score"
    ] += 50

    # Rating
    if "rating" in recommendations.columns:
        recommendations["match_score"] += (
            recommendations["rating"] * 8
        )

    # Duration preference
    if study_hours < 4:
        recommendations["match_score"] += np.where(
            recommendations["duration_hours"] <= 30,
            15,
            0
        )

    elif study_hours >= 7:
        recommendations["match_score"] += np.where(
            recommendations["duration_hours"] >= 30,
            15,
            0
        )

    # Beginner students benefit from shorter courses
    if score < 55:
        recommendations["match_score"] += np.where(
            recommendations["duration_hours"] <= 30,
            10,
            0
        )

    recommendations = recommendations.sort_values(
        "match_score",
        ascending=False
    )

    return recommendations.head(top_n)


def predict_learning_level(score, completion):
    """
    Predict an interpretable learning level.
    """

    if score >= 80 and completion >= 75:
        return "Advanced"

    elif score >= 60 and completion >= 50:
        return "Intermediate"

    return "Beginner"


# ============================================================
# LOAD DATA
# ============================================================

students = load_csv(DATA_PATH)
courses = load_csv(COURSE_PATH)

if students is None:
    students = create_demo_students()

if courses is None:
    courses = create_demo_courses()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 EduPro")

st.sidebar.markdown(
    "### Student Analytics & Recommendation"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "👤 Student Profile",
        "📊 Segmentation",
        "🎯 Course Recommendation",
        "📚 Course Catalogue"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "EduPro uses machine learning and clustering "
    "to analyze student behavior and personalize "
    "course recommendations."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="title">🎓 EduPro</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Student Segmentation & Personalized Course "
        "Recommendation System"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    total_students = len(students)
    total_courses = len(courses)

    avg_score = students["average_score"].mean()

    avg_completion = students[
        "course_completion_rate"
    ].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👨‍🎓 Students",
            f"{total_students:,}"
        )

    with col2:
        st.metric(
            "📚 Courses",
            f"{total_courses:,}"
        )

    with col3:
        st.metric(
            "📈 Average Score",
            f"{avg_score:.1f}%"
        )

    with col4:
        st.metric(
            "✅ Completion Rate",
            f"{avg_completion:.1f}%"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Score Distribution")

        if "average_score" in students.columns:

            histogram = pd.cut(
                students["average_score"],
                bins=[0, 40, 60, 75, 90, 100],
                labels=[
                    "0-40",
                    "41-60",
                    "61-75",
                    "76-90",
                    "91-100"
                ]
            ).value_counts().sort_index()

            st.bar_chart(histogram)

    with col2:

        st.subheader("⏱️ Study Hours")

        if "study_hours" in students.columns:

            study_data = students[
                "study_hours"
            ].round().value_counts().sort_index()

            st.bar_chart(study_data)


# ============================================================
# STUDENT PROFILE
# ============================================================

elif page == "👤 Student Profile":

    st.title("👤 Student Profile")

    student_ids = students["student_id"].tolist()

    selected_id = st.selectbox(
        "Select Student",
        student_ids
    )

    student = students[
        students["student_id"] == selected_id
    ].iloc[0]

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Score",
            f"{student.get('average_score', 0):.1f}%"
        )

    with col2:
        st.metric(
            "Completion Rate",
            f"{student.get('course_completion_rate', 0):.1f}%"
        )

    with col3:
        st.metric(
            "Study Hours",
            f"{student.get('study_hours', 0):.1f}"
        )

    st.markdown("---")

    learning_level = predict_learning_level(
        student.get("average_score", 60),
        student.get("course_completion_rate", 60)
    )

    st.subheader("🧠 Learning Level")

    if learning_level == "Advanced":

        st.markdown(
            """
            <div class="success-box">
            <b>Advanced Learner</b><br>
            This student is performing strongly and can
            be recommended advanced-level courses.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif learning_level == "Intermediate":

        st.info(
            "Intermediate learner — suitable for "
            "intermediate-level courses."
        )

    else:

        st.markdown(
            """
            <div class="warning-box">
            <b>Beginner / Support Required</b><br>
            Start with foundational courses and gradually
            increase difficulty.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📋 Student Details")

    profile = student.to_frame(
        name="Value"
    )

    st.dataframe(
        profile,
        use_container_width=True
    )


# ============================================================
# SEGMENTATION
# ============================================================

elif page == "📊 Segmentation":

    st.title("📊 Student Segmentation")

    features = get_student_features(students)

    if len(features) < 2:

        st.error(
            "Not enough numerical features available "
            "for clustering."
        )

    else:

        st.write(
            "K-Means clustering groups students according "
            "to their learning behavior and performance."
        )

        n_clusters = st.slider(
            "Number of Student Segments",
            min_value=2,
            max_value=8,
            value=4
        )

        segmented, scaler, kmeans, X_scaled = (
            perform_clustering(
                students,
                features,
                n_clusters
            )
        )

        st.markdown("---")

        # PCA visualization
        pca = PCA(n_components=2)

        components = pca.fit_transform(
            X_scaled
        )

        plot_df = pd.DataFrame({
            "PC1": components[:, 0],
            "PC2": components[:, 1],
            "Cluster": segmented["cluster"].astype(str)
        })

        st.subheader("🔬 PCA Cluster Visualization")

        st.scatter_chart(
            plot_df,
            x="PC1",
            y="PC2",
            color="Cluster"
        )

        st.markdown("---")

        st.subheader("📌 Segment Summary")

        summary = segmented.groupby(
            "cluster"
        ).agg({
            "average_score": "mean",
            "course_completion_rate": "mean",
            "study_hours": "mean"
        }).reset_index()

        summary["segment_name"] = [
            get_cluster_name(
                segmented[
                    segmented["cluster"] == cluster
                ]
            )
            for cluster in summary["cluster"]
        ]

        summary["average_score"] = (
            summary["average_score"].round(2)
        )

        summary["course_completion_rate"] = (
            summary["course_completion_rate"].round(2)
        )

        summary["study_hours"] = (
            summary["study_hours"].round(2)
        )

        st.dataframe(
            summary,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("👥 Student Cluster Distribution")

        distribution = (
            segmented["cluster"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(distribution)


# ============================================================
# COURSE RECOMMENDATION
# ============================================================

elif page == "🎯 Course Recommendation":

    st.title("🎯 Personalized Course Recommendation")

    student_ids = students["student_id"].tolist()

    selected_id = st.selectbox(
        "Select Student",
        student_ids,
        key="recommendation_student"
    )

    student = students[
        students["student_id"] == selected_id
    ].iloc[0]

    recommendations = recommend_courses(
        student,
        courses,
        top_n=5
    )

    st.markdown("---")

    st.subheader("👤 Student Learning Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Score",
            f"{student.get('average_score', 0):.1f}%"
        )

    with col2:
        st.metric(
            "Completion",
            f"{student.get('course_completion_rate', 0):.1f}%"
        )

    with col3:
        st.metric(
            "Study Hours",
            f"{student.get('study_hours', 0):.1f}"
        )

    level = predict_learning_level(
        student.get("average_score", 60),
        student.get("course_completion_rate", 60)
    )

    st.info(
        f"Recommended learning level: **{level}**"
    )

    st.markdown("---")

    st.subheader("⭐ Recommended Courses")

    for index, course in recommendations.iterrows():

        st.markdown(
            f"""
            <div class="recommendation">
                <h4>📘 {course['course_name']}</h4>
                <b>Category:</b> {course['category']}<br>
                <b>Difficulty:</b> {course['difficulty']}<br>
                <b>Duration:</b> {course['duration_hours']} hours<br>
                <b>Rating:</b> ⭐ {course['rating']}<br>
                <b>Match Score:</b>
                {course['match_score']:.1f}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# COURSE CATALOGUE
# ============================================================

elif page == "📚 Course Catalogue":

    st.title("📚 Course Catalogue")

    col1, col2, col3 = st.columns(3)

    with col1:

        categories = [
            "All"
        ] + sorted(
            courses["category"].dropna().unique().tolist()
        )

        selected_category = st.selectbox(
            "Category",
            categories
        )

    with col2:

        difficulties = [
            "All"
        ] + sorted(
            courses["difficulty"].dropna().unique().tolist()
        )

        selected_difficulty = st.selectbox(
            "Difficulty",
            difficulties
        )

    with col3:

        min_rating = st.slider(
            "Minimum Rating",
            0.0,
            5.0,
            0.0,
            0.1
        )

    filtered = courses.copy()

    if selected_category != "All":

        filtered = filtered[
            filtered["category"] == selected_category
        ]

    if selected_difficulty != "All":

        filtered = filtered[
            filtered["difficulty"] == selected_difficulty
        ]

    if "rating" in filtered.columns:

        filtered = filtered[
            filtered["rating"] >= min_rating
        ]

    st.markdown("---")

    st.write(
        f"Showing **{len(filtered)}** courses"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎓 EduPro | Student Segmentation & "
    "Personalized Course Recommen