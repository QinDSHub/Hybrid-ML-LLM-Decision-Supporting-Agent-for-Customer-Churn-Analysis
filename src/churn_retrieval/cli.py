from __future__ import annotations

import argparse

from churn_retrieval.config import load_config
from churn_retrieval.pipeline import prepare_runtime, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Customer churn retrieval pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default="configs/default.toml", help="Path to TOML config")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    add_config_argument(run_parser)

    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing only")
    add_config_argument(preprocess_parser)

    predict_parser = subparsers.add_parser("predict", help="Run retrieval prediction only")
    add_config_argument(predict_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run evaluation only")
    add_config_argument(evaluate_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_pipeline(args.config)
        return 0

    app_config = load_config(args.config)
    prepare_runtime(app_config)

    if args.command == "preprocess":
        from churn_retrieval.preprocessing.service import run_preprocessing
        run_preprocessing(app_config.preprocess)
        return 0
    if args.command == "predict":
        from churn_retrieval.modeling.service import run_prediction
        run_prediction(app_config.model)
        return 0
    if args.command == "evaluate":
        from churn_retrieval.evaluation.service import evaluate_predictions
        evaluate_predictions(app_config.evaluation)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
