# src/recommendation.py

import os
import pandas as pd
import numpy as np


# ============================================================
# LOAD DATA
# ============================================================

def load_csv(file_path):
    """
    Load CSV file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return pd.read_csv(file_path)


# ============================================================
# LEARNING LEVEL
# ============================================================

def get_learning_level(
    average_score,
    completion_rate
):
    """
    Determine student's learning level.
    """

    if (
        average_score >= 80
        and completion_rate >= 75
    ):
        return "Advanced"

    elif (
        average_score >= 60
        and completion_rate >= 50
    ):
        return "Intermediate"

    return "Beginner"


# ============================================================
# COURSE DIFFICULTY SCORE
# ============================================================

def difficulty_match(
    student_level,
    course_difficulty
):
    """
    Calculate difficulty compatibility.
    """

    level_scores = {
        "Beginner": 1,
        "Intermediate": 2,
        "Advanced": 3
    }

    student_score = level_scores.get(
        student_level,
        2
    )

    course_score = level_scores.get(
        str(course_difficulty),
        2
    )

    difference = abs(
        student_score - course_score
    )

    if difference == 0:
        return 40

    elif difference == 1:
        return 25

    return 10


# ============================================================
# DURATION SCORE
# ============================================================

def duration_match(
    study_hours,
    duration_hours
):
    """
    Match course duration with student's
    available study time.
    """

    if pd.isna(duration_hours):
        return 0

    # Students with lower study hours
    # are better matched with shorter courses.
    if study_hours < 4:

        if duration_hours <= 25:
            return 20

        elif duration_hours <= 40:
            return 10

        return 5

    # Moderate study time
    elif study_hours < 7:

        if 20 <= duration_hours <= 40:
            return 20

        elif duration_hours < 20:
            return 15

        return 10

    # Highly active students
    else:

        if duration_hours >= 35:
            return 20

        elif duration_hours >= 20:
            return 15

        return 10


# ============================================================
# RATING SCORE
# ============================================================

def rating_score(rating):
    """
    Convert course rating into recommendation score.
    """

    if pd.isna(rating):
        return 0

    return (
        float(rating) / 5
    ) * 20


# ============================================================
# PREVIOUS COURSE FILTER
# ============================================================

def remove_completed_courses(
    courses,
    student_id,
    transactions
):
    """
    Remove courses already completed by the student.
    """

    if transactions is None:
        return courses

    if "user_id" not in transactions.columns:
        return courses

    if "course_id" not in transactions.columns:
        return courses

    student_transactions = transactions[
        transactions["user_id"] == student_id
    ]

    if student_transactions.empty:
        return courses

    completed = student_transactions[
        student_transactions[
            "completion_status"
        ].astype(str).str.lower()
        == "completed"
    ]

    completed_course_ids = set(
        completed["course_id"].tolist()
    )

    if not completed_course_ids:
        return courses

    return courses[
        ~courses["course_id"].isin(
            completed_course_ids
        )
    ].copy()


# ============================================================
# CATEGORY PREFERENCE
# ============================================================

def calculate_category_preferences(
    student_id,
    transactions,
    courses
):
    """
    Identify categories the student has previously
    interacted with.
    """

    if transactions is None:
        return {}

    required = [
        "user_id",
        "course_id"
    ]

    if not all(
        column in transactions.columns
        for column in required
    ):
        return {}

    student_transactions = transactions[
        transactions["user_id"] == student_id
    ]

    if student_transactions.empty:
        return {}

    history = student_transactions.merge(
        courses[
            ["course_id", "category"]
        ],
        on="course_id",
        how="left"
    )

    preferences = (
        history["category"]
        .value_counts()
        .to_dict()
    )

    return preferences


# ============================================================
# RECOMMENDATION SCORE
# ============================================================

def calculate_recommendation_score(
    student,
    course,
    category_preferences=None
):
    """
    Calculate final recommendation score.
    """

    average_score = float(
        student.get(
            "average_score",
            60
        )
    )

    completion_rate = float(
        student.get(
            "course_completion_rate",
            60
        )
    )

    study_hours = float(
        student.get(
            "study_hours",
            5
        )
    )

    student_level = get_learning_level(
        average_score,
        completion_rate
    )

    score = 0

    # --------------------------------------------------------
    # Difficulty compatibility
    # --------------------------------------------------------

    score += difficulty_match(
        student_level,
        course.get(
            "difficulty",
            "Intermediate"
        )
    )

    # --------------------------------------------------------
    # Course rating
    # --------------------------------------------------------

    score += rating_score(
        course.get(
            "rating",
            0
        )
    )

    # --------------------------------------------------------
    # Course duration
    # --------------------------------------------------------

    score += duration_match(
        study_hours,
        course.get(
            "duration_hours",
            30
        )
    )

    # --------------------------------------------------------
    # Category preference
    # --------------------------------------------------------

    if (
        category_preferences
        and "category" in course
    ):

        category = course[
            "category"
        ]

        if category in category_preferences:

            # Maximum additional preference score
            # is intentionally capped.
            preference_count = (
                category_preferences[
                    category
                ]
            )

            score += min(
                preference_count * 5,
                20
            )

    # --------------------------------------------------------
    # Performance-based bonus
    # --------------------------------------------------------

    if average_score >= 80:

        if course.get(
            "difficulty"
        ) == "Advanced":

            score += 10

    elif average_score < 55:

        if course.get(
            "difficulty"
        ) == "Beginner":

            score += 10

    return round(
        score,
        2
    )


# ============================================================
# RECOMMEND COURSES
# ============================================================

def recommend_courses(
    student,
    courses,
    transactions=None,
    top_n=5
):
    """
    Generate personalized course recommendations.

    Parameters
    ----------
    student : pandas Series or dict
        Student information.

    courses : pandas DataFrame
        Course catalogue.

    transactions : pandas DataFrame, optional
        Student transaction/enrollment history.

    top_n : int
        Number of courses to recommend.
    """

    course_data = courses.copy()

    # --------------------------------------------------------
    # Remove completed courses
    # --------------------------------------------------------

    student_id = student.get(
        "student_id",
        student.get("user_id", None)
    )

    if student_id is not None:

        course_data = remove_completed_courses(
            course_data,
            student_id,
            transactions
        )

    # --------------------------------------------------------
    # Category preferences
    # --------------------------------------------------------

    category_preferences = (
        calculate_category_preferences(
            student_id,
            transactions,
            course_data
        )
        if student_id is not None
        else {}
    )

    # --------------------------------------------------------
    # Calculate scores
    # --------------------------------------------------------

    course_data[
        "recommendation_score"
    ] = course_data.apply(
        lambda course: calculate_recommendation_score(
            student,
            course,
            category_preferences
        ),
        axis=1
    )

    # --------------------------------------------------------
    # Sort recommendations
    # --------------------------------------------------------

    course_data = course_data.sort_values(
        by="recommendation_score",
        ascending=False
    )

    # --------------------------------------------------------
    # Remove duplicate course names
    # --------------------------------------------------------

    if "course_name" in course_data.columns:

        course_data = course_data.drop_duplicates(
            subset=["course_name"]
        )

    return course_data.head(
        top_n
    ).reset_index(
        drop=True
    )


# ============================================================
# RECOMMENDATION EXPLANATION
# ============================================================

def generate_reason(
    student,
    course
):
    """
    Generate a simple explanation for
    why a course was recommended.
    """

    average_score = float(
        student.get(
            "average_score",
            60
        )
    )

    completion_rate = float(
        student.get(
            "course_completion_rate",
            60
        )
    )

    study_hours = float(
        student.get(
            "study_hours",
            5
        )
    )

    level = get_learning_level(
        average_score,
        completion_rate
    )

    reasons = []

    # Difficulty reason
    course_difficulty = course.get(
        "difficulty",
        "Intermediate"
    )

    if course_difficulty == level:

        reasons.append(
            f"matches your {level.lower()} "
            "learning level"
        )

    # Performance reason
    if (
        average_score >= 80
        and course_difficulty == "Advanced"
    ):

        reasons.append(
            "your strong academic performance "
            "supports an advanced course"
        )

    elif (
        average_score < 55
        and course_difficulty == "Beginner"
    ):

        reasons.append(
            "the beginner level is suitable "
            "for strengthening your fundamentals"
        )

    # Study-time reason
    duration = course.get(
        "duration_hours",
        30
    )

    if study_hours < 4 and duration <= 25:

        reasons.append(
            "the shorter duration suits "
            "your available study time"
        )

    elif study_hours >= 7 and duration >= 35:

        reasons.append(
            "the course duration suits "
            "your high study activity"
        )

    # Rating
    rating = course.get(
        "rating",
        0
    )

    if not pd.isna(rating) and rating >= 4.7:

        reasons.append(
            "it has a high learner rating"
        )

    if not reasons:

        reasons.append(
            "it matches your overall "
            "learning profile"
        )

    return "; ".join(
        reasons
    )


# ============================================================
# ADD RECOMMENDATION REASONS
# ============================================================

def add_recommendation_reasons(
    student,
    recommendations
):
    """
    Add human-readable explanation to recommendations.
    """

    result = recommendations.copy()

    result["reason"] = result.apply(
        lambda course: generate_reason(
            student,
            course
        ),
        axis=1
    )

    return result


# ============================================================
# STUDENT RECOMMENDATION REPORT
# ============================================================

def generate_student_report(
    student,
    courses,
    transactions=None,
    top_n=5
):
    """
    Generate a complete recommendation report.
    """

    recommendations = recommend_courses(
        student,
        courses,
        transactions,
        top_n
    )

    recommendations = add_recommendation_reasons(
        student,
        recommendations
    )

    student_id = student.get(
        "student_id",
        student.get("user_id", "N/A")
    )

    average_score = student.get(
        "average_score",
        0
    )

    completion_rate = student.get(
        "course_completion_rate",
        0
    )

    learning_level = get_learning_level(
        average_score,
        completion_rate
    )

    report = {
        "student_id": student_id,
        "learning_level": learning_level,
        "average_score": average_score,
        "completion_rate": completion_rate,
        "recommendations": recommendations
    }

    return report


# ============================================================
# SAVE RECOMMENDATIONS
# ============================================================

def save_recommendations(
    recommendations,
    file_path="data/recommendations.csv"
):
    """
    Save recommendation results to CSV.
    """

    directory = os.path.dirname(
        file_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    recommendations.to_csv(
        file_path,
        index=False
    )

    print(
        f"Recommendations saved to: "
        f"{file_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    STUDENTS_PATH = (
        "data/students.csv"
    )

    COURSES_PATH = (
        "data/courses.csv"
    )

    TRANSACTIONS_PATH = (
        "data/transactions.csv"
    )

    try:

        print("--------------------------------")
        print("EduPro Course Recommendation")
        print("--------------------------------")

        # ----------------------------------------------------
        # Load students
        # ----------------------------------------------------

        students = load_csv(
            STUDENTS_PATH
        )

        # ----------------------------------------------------
        # Load courses
        # ----------------------------------------------------

        courses = load_csv(
            COURSES_PATH
        )

        # ----------------------------------------------------
        # Load transactions if available
        # ----------------------------------------------------

        transactions = None

        if os.path.exists(
            TRANSACTIONS_PATH
        ):

            transactions = pd.read_csv(
                TRANSACTIONS_PATH
            )

            print(
                "Transaction history loaded."
            )

        else:

            print(
                "Transaction file not found. "
                "Continuing without history."
            )

        # ----------------------------------------------------
        # Select a student
        # ----------------------------------------------------

        student = students.iloc[0]

        # ----------------------------------------------------
        # Generate recommendations
        # ----------------------------------------------------

        report = generate_student_report(
            student,
            courses,
            transactions,
            top_n=5
        )

        print("\n--------------------------------")
        print("Student Profile")
        print("--------------------------------")

        print(
            f"Student ID: "
            f"{report['student_id']}"
        )

        print(
            f"Learning Level: "
            f"{report['learning_level']}"
        )

        print(
            f"Average Score: "
            f"{report['average_score']}"
        )

        print(
            f"Completion Rate: "
            f"{report['completion_rate']}"
        )

        print("\n--------------------------------")
        print("Recommended Courses")
        print("--------------------------------")

        recommendations = (
            report["recommendations"]
        )

        display_columns = [
            "course_id",
            "course_name",
            "category",
            "difficulty",
            "duration_hours",
            "rating",
            "recommendation_score",
            "reason"
        ]

        available_columns = [
            column
            for column in display_columns
            if column in recommendations.columns
        ]

        print(
            recommendations[
                available_columns
            ].to_string(index=False)
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_recommendations(
            recommendations
        )

        print("\nRecommendation completed!")

    except Exception as e:

        print(
            f"Recommendation error: {e}"
        )
