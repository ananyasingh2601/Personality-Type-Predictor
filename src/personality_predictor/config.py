from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DOC_DIR = BASE_DIR / "output" / "doc"
DATA_DIR = BASE_DIR / "data"

DATASET_CANDIDATES = [
    DATA_DIR / "MBTI 500.csv",
    Path(r"C:\Users\HP\Downloads\ML LAB\MBTI 500.csv"),
]

APP_TITLE = "Prism Persona"
APP_SUBTITLE = "A colorful MBTI-style personality predictor powered by Decision Tree and KNN models."

MODEL_CONFIGS = {
    "Decision Tree": {
        "filename": "decision_tree_pipeline.joblib",
        "classifier": "decision_tree",
        "label": "Decision Tree",
        "sample_size": 25000,
        "max_features": 4000,
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_depth": 30,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "class_weight": "balanced",
    },
    "KNN": {
        "filename": "knn_pipeline.joblib",
        "classifier": "knn",
        "label": "KNN",
        "sample_size": 12000,
        "max_features": 3000,
        "ngram_range": (1, 2),
        "min_df": 4,
        "n_neighbors": 11,
        "weights": "distance",
        "metric": "cosine",
        "algorithm": "brute",
    },
}

TYPE_GROUPS = {
    "INTJ": "Analysts",
    "INTP": "Analysts",
    "ENTJ": "Analysts",
    "ENTP": "Analysts",
    "INFJ": "Diplomats",
    "INFP": "Diplomats",
    "ENFJ": "Diplomats",
    "ENFP": "Diplomats",
    "ISTJ": "Sentinels",
    "ISFJ": "Sentinels",
    "ESTJ": "Sentinels",
    "ESFJ": "Sentinels",
    "ISTP": "Explorers",
    "ISFP": "Explorers",
    "ESTP": "Explorers",
    "ESFP": "Explorers",
}

GROUP_ACCENTS = {
    "Analysts": "#326BFF",
    "Diplomats": "#E35D8F",
    "Sentinels": "#0B9C8F",
    "Explorers": "#FF9F1C",
}

COLOR_SYSTEM = {
    "paper": "#F7F1E8",
    "paper_alt": "#FFF9F1",
    "ink": "#152238",
    "muted": "#5F6B7A",
    "border": "#E1D7C8",
    "shadow": "rgba(21, 34, 56, 0.12)",
    "good": "#1F8A70",
    "warning": "#C9722A",
    "gradient_start": "#FFF5E7",
    "gradient_mid": "#F9FBFF",
    "gradient_end": "#E9F6F2",
}

TYPOGRAPHY_SCALE = {
    "display": "3.4rem",
    "h1": "2.2rem",
    "h2": "1.5rem",
    "h3": "1.1rem",
    "body": "1rem",
    "caption": "0.88rem",
}

TYPE_PROFILES = {
    "INTJ": {"title": "Strategic Architect", "summary": "Independent, future-focused, and energized by building systems that work."},
    "INTP": {"title": "Curious Analyst", "summary": "Concept-driven thinker who likes exploring how ideas fit together."},
    "ENTJ": {"title": "Bold Commander", "summary": "Decisive organizer who naturally steers teams toward ambitious goals."},
    "ENTP": {"title": "Inventive Debater", "summary": "Playful strategist who enjoys patterns, possibility, and challenge."},
    "INFJ": {"title": "Insightful Guide", "summary": "Meaning-oriented planner who blends empathy with long-range vision."},
    "INFP": {"title": "Idealistic Mediator", "summary": "Values-led creative who seeks authenticity, purpose, and growth."},
    "ENFJ": {"title": "Inspiring Mentor", "summary": "People-centered organizer who motivates others with warmth and clarity."},
    "ENFP": {"title": "Imaginative Campaigner", "summary": "Energetic explorer who follows curiosity, connection, and fresh ideas."},
    "ISTJ": {"title": "Reliable Inspector", "summary": "Practical, steady, and committed to doing things carefully and well."},
    "ISFJ": {"title": "Supportive Defender", "summary": "Quietly dependable helper who notices needs and follows through."},
    "ESTJ": {"title": "Organized Executive", "summary": "Structured leader who values efficiency, clarity, and accountability."},
    "ESFJ": {"title": "Community Host", "summary": "Warm coordinator who keeps groups connected, supported, and on track."},
    "ISTP": {"title": "Hands-on Explorer", "summary": "Action-oriented problem solver who likes practical freedom and craft."},
    "ISFP": {"title": "Creative Adventurer", "summary": "Flexible, aesthetic, and grounded in personal values and experience."},
    "ESTP": {"title": "Energetic Dynamo", "summary": "Fast-moving realist who learns by doing and adapts in the moment."},
    "ESFP": {"title": "Vibrant Performer", "summary": "Expressive connector who brings momentum, warmth, and spontaneity."},
}

DIMENSION_META = {
    "IE": {"title": "Energy", "left": "I", "right": "E", "left_label": "Introversion", "right_label": "Extraversion"},
    "SN": {"title": "Focus", "left": "S", "right": "N", "left_label": "Sensing", "right_label": "Intuition"},
    "TF": {"title": "Decisions", "left": "T", "right": "F", "left_label": "Thinking", "right_label": "Feeling"},
    "JP": {"title": "Lifestyle", "left": "J", "right": "P", "left_label": "Judging", "right_label": "Perceiving"},
}

QUIZ_DISCLAIMER = (
    "This project is for educational use. MBTI-style outputs are descriptive, not clinical or diagnostic."
)

