from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import DATASET_CANDIDATES, MODEL_CONFIGS, MODELS_DIR

MBTI_PATTERN = re.compile(r"\b(?:infj|infp|intj|intp|isfj|isfp|istj|istp|enfj|enfp|entj|entp|esfj|esfp|estj|estp)\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def resolve_dataset_path(explicit_path: str | None = None) -> Path:
    candidates = [Path(explicit_path)] if explicit_path else []
    candidates.extend(DATASET_CANDIDATES)
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError("MBTI dataset not found. Add MBTI 500.csv to data/ or update the dataset path.")


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("|||", " ")
    text = URL_PATTERN.sub(" ", text)
    text = MBTI_PATTERN.sub(" personality ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()


def load_dataset(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["posts", "type"])
    if max_rows:
        frame = frame.head(max_rows)
    frame = frame.dropna(subset=["posts", "type"]).copy()
    frame["posts"] = frame["posts"].astype(str).map(normalize_text)
    frame["type"] = frame["type"].astype(str).str.upper()
    frame = frame[frame["posts"].str.len() > 10]
    frame = frame.drop_duplicates(subset=["posts", "type"])
    return frame


def build_pipeline(model_name: str) -> Pipeline:
    spec = MODEL_CONFIGS[model_name]
    vectorizer = TfidfVectorizer(
        max_features=spec["max_features"],
        ngram_range=spec["ngram_range"],
        min_df=spec["min_df"],
        stop_words="english",
        sublinear_tf=True,
    )

    if spec["classifier"] == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=spec["max_depth"],
            min_samples_split=spec["min_samples_split"],
            min_samples_leaf=spec["min_samples_leaf"],
            class_weight=spec["class_weight"],
            random_state=42,
        )
    else:
        classifier = KNeighborsClassifier(
            n_neighbors=spec["n_neighbors"],
            weights=spec["weights"],
            metric=spec["metric"],
            algorithm=spec["algorithm"],
        )
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def sample_training_set(X_train, y_train, sample_size: int | None):
    if not sample_size or len(y_train) <= sample_size:
        return X_train, y_train
    sampled = (
        pd.DataFrame({"posts": X_train, "type": y_train})
        .groupby("type", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=max(1, round(sample_size * len(group) / len(y_train))),
                random_state=42,
                replace=False,
            )
        )
        .reset_index(drop=True)
    )
    return sampled["posts"], sampled["type"]


def train_and_save_models(dataset_path: str | None = None, max_rows: int | None = None) -> dict[str, object]:
    data_path = resolve_dataset_path(dataset_path)
    frame = load_dataset(data_path, max_rows=max_rows)

    X_train, X_test, y_train, y_test = train_test_split(
        frame["posts"],
        frame["type"],
        test_size=0.2,
        random_state=42,
        stratify=frame["type"],
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "dataset_path": str(data_path),
        "rows_used": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "models": {},
    }

    for model_name in MODEL_CONFIGS:
        spec = MODEL_CONFIGS[model_name]
        sampled_X_train, sampled_y_train = sample_training_set(X_train, y_train, spec["sample_size"])
        pipeline = build_pipeline(model_name)
        pipeline.fit(sampled_X_train, sampled_y_train)

        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

        model_path = MODELS_DIR / spec["filename"]
        joblib.dump(pipeline, model_path)

        metrics["models"][model_name] = {
            "model_path": str(model_path),
            "accuracy": round(float(accuracy), 4),
            "sample_size": int(len(sampled_X_train)),
            "vectorizer": {
                "max_features": spec["max_features"],
                "ngram_range": list(spec["ngram_range"]),
                "min_df": spec["min_df"],
            },
            "hyperparameters": {
                key: value
                for key, value in spec.items()
                if key not in {"filename", "label", "classifier", "sample_size", "max_features", "ngram_range", "min_df"}
            },
            "macro_avg_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "weighted_avg_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        }

    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_name: str):
    spec = MODEL_CONFIGS[model_name]
    model_path = MODELS_DIR / spec["filename"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_profile(model_name: str, profile_text: str) -> dict[str, object]:
    model = load_model(model_name)
    probabilities = model.predict_proba([profile_text])[0]
    labels = model.classes_
    ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
    prediction, confidence = ranked[0]
    top_three = [{"type": label, "probability": round(float(prob) * 100, 1)} for label, prob in ranked[:3]]
    return {
        "prediction": str(prediction),
        "confidence": round(float(confidence) * 100, 1),
        "top_three": top_three,
    }

