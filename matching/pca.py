"""
PCA dimensionality reduction for school culture vectors.

Can be run standalone to fit and save the PCA model:
    python matching/pca.py

Or imported and used by the Streamlit app.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.dirname(__file__)

# These are the culture dimension columns used for PCA.
# Sentiment columns + Niche grade columns (not volume columns).
SENTIMENT_COLS = [
    "social_sentiment", "academics_sentiment", "campus_sentiment",
    "diversity_sentiment", "career_sentiment", "location_sentiment",
    "religion_sentiment", "spirit_sentiment",
]

NICHE_COLS = [
    "academics_grade", "value_grade", "diversity_grade", "campus_grade",
    "athletics_grade", "party_scene_grade", "safety_grade", "food_grade",
    "dorms_grade",
]

# All feature columns used in the culture vector
FEATURE_COLS = SENTIMENT_COLS + NICHE_COLS

# Human-readable labels for the feature columns (for PCA loadings display)
FEATURE_LABELS = {
    "social_sentiment": "Social/Party Life",
    "academics_sentiment": "Academic Culture",
    "campus_sentiment": "Campus Feel",
    "diversity_sentiment": "Diversity Vibe",
    "career_sentiment": "Career Focus",
    "location_sentiment": "Location Appeal",
    "religion_sentiment": "Religious Culture",
    "spirit_sentiment": "School Spirit",
    "academics_grade": "Academic Quality",
    "value_grade": "Value",
    "diversity_grade": "Diversity Score",
    "campus_grade": "Campus Quality",
    "athletics_grade": "Athletics",
    "party_scene_grade": "Party Scene",
    "safety_grade": "Safety",
    "food_grade": "Food Quality",
    "dorms_grade": "Dorm Quality",
}


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of FEATURE_COLS that exist in the dataframe."""
    return [c for c in FEATURE_COLS if c in df.columns]


def fit_pca(profiles_df: pd.DataFrame, n_components: int = 2):
    """
    Fit PCA on school culture vectors.

    Returns:
        pca: fitted PCA object
        scaler: fitted StandardScaler
        coords: DataFrame with school, pc1, pc2
        loadings: DataFrame with feature, pc1_loading, pc2_loading
    """
    features = get_available_features(profiles_df)
    X = profiles_df[features].fillna(0.5).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # School coordinates in PCA space
    coords = pd.DataFrame({
        "school": profiles_df["school"].values,
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
    })

    # Loadings for axis labelling
    loadings = pd.DataFrame({
        "feature": features,
        "pc1_loading": pca.components_[0],
        "pc2_loading": pca.components_[1],
    })
    loadings["feature_label"] = loadings["feature"].map(
        lambda f: FEATURE_LABELS.get(f, f)
    )

    return pca, scaler, coords, loadings


def get_axis_labels(loadings: pd.DataFrame) -> tuple[str, str]:
    """Generate human-readable axis labels from PCA loadings."""
    # PC1
    pc1_sorted = loadings.sort_values("pc1_loading")
    pc1_neg = pc1_sorted.head(2)["feature_label"].tolist()
    pc1_pos = pc1_sorted.tail(2)["feature_label"].tolist()
    pc1_label = f"← {', '.join(pc1_neg)} | {', '.join(pc1_pos)} →"

    # PC2
    pc2_sorted = loadings.sort_values("pc2_loading")
    pc2_neg = pc2_sorted.head(2)["feature_label"].tolist()
    pc2_pos = pc2_sorted.tail(2)["feature_label"].tolist()
    pc2_label = f"← {', '.join(pc2_neg)} | {', '.join(pc2_pos)} →"

    return pc1_label, pc2_label


def project_user_vector(user_vector: np.ndarray, scaler, pca) -> np.ndarray:
    """Project a user's preference vector into PCA space."""
    user_scaled = scaler.transform(user_vector.reshape(1, -1))
    return pca.transform(user_scaled)[0]


def save_model(pca, scaler, coords, loadings, features):
    """Save PCA model artifacts to disk."""
    model_data = {
        "pca": pca,
        "scaler": scaler,
        "coords": coords,
        "loadings": loadings,
        "features": features,
    }
    path = os.path.join(MODEL_DIR, "pca_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Saved PCA model to {path}")


def load_model():
    """Load saved PCA model artifacts."""
    path = os.path.join(MODEL_DIR, "pca_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    """Fit PCA on school profiles and save the model."""
    profiles_path = os.path.join(DATA_DIR, "school_profiles.csv")
    profiles = pd.read_csv(profiles_path)
    print(f"Loaded {len(profiles)} school profiles.")

    features = get_available_features(profiles)
    print(f"Using {len(features)} features: {features}")

    pca, scaler, coords, loadings = fit_pca(profiles)

    pc1_label, pc2_label = get_axis_labels(loadings)
    print(f"PC1: {pc1_label}")
    print(f"PC2: {pc2_label}")
    print(f"Explained variance: {pca.explained_variance_ratio_}")

    save_model(pca, scaler, coords, loadings, features)

    # Also save coordinates as CSV for easy inspection
    coords_path = os.path.join(MODEL_DIR, "school_coords.csv")
    coords.to_csv(coords_path, index=False)
    print(f"Saved school coordinates to {coords_path}")


if __name__ == "__main__":
    main()
