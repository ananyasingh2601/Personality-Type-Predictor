from __future__ import annotations

import argparse
import json

from src.personality_predictor.ml import train_and_save_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MBTI Decision Tree and KNN models.")
    parser.add_argument("--data", type=str, default=None, help="Optional path to MBTI 500.csv")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for faster experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_save_models(dataset_path=args.data, max_rows=args.max_rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

