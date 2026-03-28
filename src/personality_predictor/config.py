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
APP_TAGLINE = "Answer 10 quick questions, compare two classic ML models, and explore your MBTI-style profile through a vivid visual dashboard."

MODEL_CONFIGS = {
    "Decision Tree": {
        "filename": "decision_tree_pipeline.joblib",
        "classifier": "decision_tree",
        "label": "Decision Tree",
        "sample_size": 42000,
        "max_features": 6500,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.92,
        "criterion": "entropy",
        "max_depth": 42,
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
    },
    "KNN": {
        "filename": "knn_pipeline.joblib",
        "classifier": "knn",
        "label": "KNN",
        "sample_size": 18000,
        "max_features": 7000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "svd_components": 220,
        "n_neighbors": 17,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2,
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

GROUP_ORDER = ["Analysts", "Diplomats", "Sentinels", "Explorers"]
TYPE_ORDER = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

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
    "INTJ": {"title": "Strategic Architect", "summary": "Independent, future-focused, and energized by building systems that work.", "strengths": "strategy, structure, independent thinking", "collab": "Give them room to think, then invite them to shape the plan."},
    "INTP": {"title": "Curious Analyst", "summary": "Concept-driven thinker who likes exploring how ideas fit together.", "strengths": "logic, curiosity, model-building", "collab": "They thrive when discussion is open, exploratory, and idea-rich."},
    "ENTJ": {"title": "Bold Commander", "summary": "Decisive organizer who naturally steers teams toward ambitious goals.", "strengths": "leadership, execution, momentum", "collab": "Keep goals explicit and they will push the work forward fast."},
    "ENTP": {"title": "Inventive Debater", "summary": "Playful strategist who enjoys patterns, possibility, and challenge.", "strengths": "innovation, reframing, energetic ideation", "collab": "Use them early when a problem needs unconventional angles."},
    "INFJ": {"title": "Insightful Guide", "summary": "Meaning-oriented planner who blends empathy with long-range vision.", "strengths": "insight, empathy, long-range thinking", "collab": "They do their best work when purpose is clear and human impact matters."},
    "INFP": {"title": "Idealistic Mediator", "summary": "Values-led creative who seeks authenticity, purpose, and growth.", "strengths": "creativity, sincerity, reflective depth", "collab": "Connect the work to values and they will bring originality."},
    "ENFJ": {"title": "Inspiring Mentor", "summary": "People-centered organizer who motivates others with warmth and clarity.", "strengths": "encouragement, orchestration, communication", "collab": "They shine when helping people move together around a shared goal."},
    "ENFP": {"title": "Imaginative Campaigner", "summary": "Energetic explorer who follows curiosity, connection, and fresh ideas.", "strengths": "enthusiasm, possibility thinking, spontaneity", "collab": "Give them freedom to explore and they will generate momentum."},
    "ISTJ": {"title": "Reliable Inspector", "summary": "Practical, steady, and committed to doing things carefully and well.", "strengths": "reliability, precision, accountability", "collab": "They appreciate clear expectations and dependable systems."},
    "ISFJ": {"title": "Supportive Defender", "summary": "Quietly dependable helper who notices needs and follows through.", "strengths": "care, consistency, detail sensitivity", "collab": "They tend to anchor a team by protecting quality and people."},
    "ESTJ": {"title": "Organized Executive", "summary": "Structured leader who values efficiency, clarity, and accountability.", "strengths": "coordination, decisiveness, structure", "collab": "They work best when plans, roles, and deadlines are visible."},
    "ESFJ": {"title": "Community Host", "summary": "Warm coordinator who keeps groups connected, supported, and on track.", "strengths": "support, responsiveness, group awareness", "collab": "They often become the glue that keeps collaboration healthy."},
    "ISTP": {"title": "Hands-on Explorer", "summary": "Action-oriented problem solver who likes practical freedom and craft.", "strengths": "adaptability, troubleshooting, calm under pressure", "collab": "Let them test things directly and they will find what works."},
    "ISFP": {"title": "Creative Adventurer", "summary": "Flexible, aesthetic, and grounded in personal values and experience.", "strengths": "taste, adaptability, authenticity", "collab": "They respond well to autonomy, trust, and meaningful work."},
    "ESTP": {"title": "Energetic Dynamo", "summary": "Fast-moving realist who learns by doing and adapts in the moment.", "strengths": "action, improvisation, risk sense", "collab": "They bring momentum when a team needs rapid practical decisions."},
    "ESFP": {"title": "Vibrant Performer", "summary": "Expressive connector who brings momentum, warmth, and spontaneity.", "strengths": "presence, warmth, quick engagement", "collab": "They tend to lift energy and make experiences feel vivid and human."},
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
