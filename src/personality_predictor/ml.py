from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier

from .config import DATASET_CANDIDATES, GROUP_ORDER, MODEL_CONFIGS, MODELS_DIR, TYPE_GROUPS, TYPE_ORDER

MBTI_PATTERN = re.compile(r"\b(?:infj|infp|intj|intp|isfj|isfp|istj|istp|enfj|enfp|entj|entp|esfj|esfp|estj|estp)\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")
METRICS_FILENAME = "metrics.json"


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


def load_metrics() -> dict[str, object]:
    metrics_path = MODELS_DIR / METRICS_FILENAME
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def summarize_dataset(dataset_path: str | None = None) -> dict[str, object]:
    path = resolve_dataset_path(dataset_path)
    counts = Counter()
    total_rows = 0
    chunk_iter = pd.read_csv(path, usecols=["type"], chunksize=50000)
    for chunk in chunk_iter:
        labels = chunk["type"].astype(str).str.upper()
        counts.update(labels)
        total_rows += int(len(labels))

    ordered_type_counts = {label: int(counts.get(label, 0)) for label in TYPE_ORDER}
    group_counts = {
        group: int(sum(ordered_type_counts[label] for label in TYPE_ORDER if TYPE_GROUPS[label] == group))
        for group in GROUP_ORDER
    }
    dominant_type = max(ordered_type_counts.items(), key=lambda item: item[1])[0]
    return {
        "rows": total_rows,
        "type_counts": ordered_type_counts,
        "group_counts": group_counts,
        "dominant_type": dominant_type,
        "type_count": len([label for label, value in ordered_type_counts.items() if value > 0]),
    }


def build_pipeline(model_name: str) -> Pipeline:
    spec = MODEL_CONFIGS[model_name]
    steps: list[tuple[str, object]] = [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=spec["max_features"],
                ngram_range=spec["ngram_range"],
                min_df=spec["min_df"],
                max_df=spec["max_df"],
                stop_words="english",
                sublinear_tf=True,
            ),
        )
    ]

    if spec["classifier"] == "decision_tree":
        steps.append(
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion=spec["criterion"],
                    max_depth=spec["max_depth"],
                    min_samples_split=spec["min_samples_split"],
                    min_samples_leaf=spec["min_samples_leaf"],
                    class_weight=spec["class_weight"],
                    random_state=42,
                ),
            )
        )
    else:
        steps.extend(
            [
                ("svd", TruncatedSVD(n_components=spec["svd_components"], random_state=42)),
                ("normalizer", Normalizer(copy=False)),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=spec["n_neighbors"],
                        weights=spec["weights"],
                        metric=spec["metric"],
                        p=spec["p"],
                        algorithm=spec["algorithm"],
                    ),
                ),
            ]
        )
    return Pipeline(steps)


def sample_training_set(X_train, y_train, sample_size: int | None):
    if not sample_size or len(y_train) <= sample_size:
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True)

    frame = pd.DataFrame({"posts": X_train, "type": y_train}).reset_index(drop=True)
    class_counts = frame["type"].value_counts().sort_index()
    exact_targets = class_counts / class_counts.sum() * sample_size
    base_targets = exact_targets.astype(int)
    remainder = int(sample_size - base_targets.sum())

    if remainder > 0:
        fractional = (exact_targets - base_targets).sort_values(ascending=False)
        for label in fractional.index[:remainder]:
            base_targets[label] += 1

    sampled_groups = []
    for label, count in base_targets.items():
        label_frame = frame[frame["type"] == label]
        count = min(int(count), len(label_frame))
        sampled_groups.append(label_frame.sample(n=count, random_state=42, replace=False))

    sampled = pd.concat(sampled_groups, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return sampled["posts"], sampled["type"]


def top_confusions_from_matrix(matrix: list[list[int]], labels: list[str], limit: int = 6) -> list[dict[str, object]]:
    rows = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            value = int(matrix[i][j])
            if i == j or value <= 0:
                continue
            rows.append({"actual": actual, "predicted": predicted, "count": value})
    rows.sort(key=lambda item: item["count"], reverse=True)
    return rows[:limit]


def rank_class_scores(report: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    class_rows = []
    for label in TYPE_ORDER:
        if label not in report:
            continue
        class_rows.append(
            {
                "type": label,
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }
        )
    best = sorted(class_rows, key=lambda item: item["f1"], reverse=True)[:5]
    weakest = sorted(class_rows, key=lambda item: item["f1"])[:5]
    return {"best_f1": best, "weakest_f1": weakest, "all": class_rows}


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

    dataset_summary = summarize_dataset(str(data_path))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "dataset_path": str(data_path),
        "rows_used": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_summary": dataset_summary,
        "models": {},
    }

    for model_name in MODEL_CONFIGS:
        spec = MODEL_CONFIGS[model_name]
        sampled_X_train, sampled_y_train = sample_training_set(X_train, y_train, spec["sample_size"])
        pipeline = build_pipeline(model_name)

        started = time.perf_counter()
        pipeline.fit(sampled_X_train, sampled_y_train)
        fit_seconds = round(time.perf_counter() - started, 2)

        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
        matrix = confusion_matrix(y_test, predictions, labels=TYPE_ORDER)

        model_path = MODELS_DIR / spec["filename"]
        joblib.dump(pipeline, model_path)

        metrics["models"][model_name] = {
            "model_path": str(model_path),
            "accuracy": round(float(accuracy), 4),
            "sample_size": int(len(sampled_X_train)),
            "fit_seconds": fit_seconds,
            "vectorizer": {
                "max_features": spec["max_features"],
                "ngram_range": list(spec["ngram_range"]),
                "min_df": spec["min_df"],
                "max_df": spec["max_df"],
            },
            "hyperparameters": {
                key: value
                for key, value in spec.items()
                if key not in {"filename", "label", "classifier", "sample_size", "max_features", "ngram_range", "min_df", "max_df"}
            },
            "macro_avg_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "weighted_avg_f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "class_scores": rank_class_scores(report),
            "confusion_matrix": matrix.tolist(),
            "labels": TYPE_ORDER,
            "top_confusions": top_confusions_from_matrix(matrix.tolist(), TYPE_ORDER),
        }

    metrics_path = MODELS_DIR / METRICS_FILENAME
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

    top_candidates = []
    group_probabilities = Counter()
    for label, prob in ranked[:6]:
        top_candidates.append({"type": label, "probability": round(float(prob) * 100, 1)})
    for label, prob in ranked:
        group_probabilities[TYPE_GROUPS.get(label, "Other")] += float(prob) * 100

    return {
        "prediction": str(prediction),
        "confidence": round(float(confidence) * 100, 1),
        "top_three": top_candidates[:3],
        "top_six": top_candidates,
        "group_probabilities": {group: round(value, 1) for group, value in group_probabilities.items()},
    }
