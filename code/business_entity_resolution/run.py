"""CLI entry point to train and run the Business Entity Resolution pipeline."""

import argparse
import os
import sys

# Ensure src is in python path
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

from src.config import (
    DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, MODEL_PATH, MATCH_PROB_THRESHOLD
)
from src.pipeline import train_pipeline, run_inference

def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all",
                        help="Execution mode: train, predict, or all (default: all)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Path to dataset root folder containing train/ and test/")
    parser.add_argument("--test-dir", default=None,
                        help="Path to test folder containing test_source1/2/3.tsv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Path to output directory (default: output/)")
    parser.add_argument("--model-path", default=MODEL_PATH,
                        help="Path to XGBoost model file")
    parser.add_argument("--threshold", type=float, default=MATCH_PROB_THRESHOLD,
                        help="Match probability threshold (default: 0.78)")
    parser.add_argument("--sample-size", type=int, default=35000,
                        help="Number of Source 1 records to use for model training (default: 35000)")
    args = parser.parse_args()

    test_dir = args.test_dir or os.path.join(args.data_dir, "test")

    if args.mode in ["train", "all"]:
        print(">>> Step 1: Training Entity Resolution Model...")
        train_pipeline(data_dir=args.data_dir, sample_s1=args.sample_size)

    if args.mode in ["predict", "all"]:
        print(">>> Step 2: Running Inference on Test Set...")
        run_inference(test_dir=test_dir, output_dir=args.output_dir,
                      model_path=args.model_path, threshold=args.threshold)

if __name__ == "__main__":
    main()
