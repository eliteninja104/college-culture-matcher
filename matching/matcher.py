"""
Cosine similarity matching between user preferences and school culture vectors.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_vector(
    survey_responses: dict[str, float],
    feature_columns: list[str],
    dimension_mapping: dict[str, list[str]],
) -> np.ndarray:
    """
    Convert survey slider values (1-10) into a culture vector aligned with
    the school profile feature columns.

    Args:
        survey_responses: mapping of survey dimension name to slider value (1-10)
        feature_columns: ordered list of feature column names from school profiles
        dimension_mapping: maps each survey dimension to the feature columns it affects

    Returns:
        numpy array of the same length as feature_columns, normalized to 0-1
    """
    vector = np.full(len(feature_columns), 0.5)  # default neutral

    for survey_dim, slider_val in survey_responses.items():
        normalized_val = slider_val / 10.0  # convert 1-10 to 0.1-1.0
        if survey_dim in dimension_mapping:
            for feat_col in dimension_mapping[survey_dim]:
                if feat_col in feature_columns:
                    idx = feature_columns.index(feat_col)
                    vector[idx] = normalized_val

    return vector


# Maps each survey question dimension to the school profile feature columns it affects
SURVEY_TO_FEATURES: dict[str, list[str]] = {
    "social": ["social_sentiment", "party_scene_grade"],
    "academics": ["academics_sentiment", "academics_grade"],
    "campus": ["campus_sentiment", "campus_grade", "food_grade", "dorms_grade"],
    "diversity": ["diversity_sentiment", "diversity_grade"],
    "career": ["career_sentiment", "value_grade"],
    "location": ["location_sentiment"],
    "religion": ["religion_sentiment"],
    "spirit": ["spirit_sentiment", "athletics_grade"],
}


def compute_matches(
    user_vector: np.ndarray,
    school_profiles: pd.DataFrame,
    feature_columns: list[str],
    weights: np.ndarray | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute cosine similarity between user vector and all school culture vectors.

    Args:
        user_vector: 1D array of user preferences (same length as feature_columns)
        school_profiles: DataFrame with school culture profiles
        feature_columns: list of column names to use from school_profiles
        weights: optional array of per-dimension weights
        top_n: number of top matches to return

    Returns:
        DataFrame with columns: school, match_score, plus all original profile columns
    """
    X = school_profiles[feature_columns].fillna(0.5).values

    user_vec = user_vector.copy()

    if weights is not None:
        X = X * weights
        user_vec = user_vec * weights

    # Compute cosine similarity
    sims = cosine_similarity(user_vec.reshape(1, -1), X)[0]

    # Convert to 0-100 percentage (cosine sim ranges -1 to 1, but usually 0 to 1 for positive vectors)
    match_scores = ((sims + 1) / 2) * 100  # shift to 0-100

    results = school_profiles.copy()
    results["match_score"] = match_scores
    results = results.sort_values("match_score", ascending=False)

    if top_n:
        return results.head(top_n)
    return results


def get_school_strengths(
    school_row: pd.Series,
    feature_columns: list[str],
    feature_labels: dict[str, str],
    top_n: int = 3,
) -> list[str]:
    """Return the top N culture strengths for a school."""
    scores = {
        feature_labels.get(col, col): school_row.get(col, 0)
        for col in feature_columns
    }
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_scores[:top_n]]
