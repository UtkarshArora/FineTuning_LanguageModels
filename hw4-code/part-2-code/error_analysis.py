import os
import argparse
from tqdm import tqdm
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from load_data import T5Dataset
from torch.utils.data import DataLoader
import re
from collections import defaultdict


def normalize_sql(s: str) -> str:
    return " ".join(s.lower().split())


def categorize_error(nl_query, ground_truth, prediction):

    gt_lower = ground_truth.lower()
    pred_lower = prediction.lower()

    if "join" in gt_lower and "join" not in pred_lower:
        return "missing_join"
    elif "join" in gt_lower and "join" in pred_lower:
        gt_join_type = re.search(r"(inner|left|right|outer)\s+join", gt_lower)
        pred_join_type = re.search(r"(inner|left|right|outer)\s+join", pred_lower)
        if (
            gt_join_type
            and pred_join_type
            and gt_join_type.group(1) != pred_join_type.group(1)
        ):
            return "wrong_join_type"

    gt_agg = re.findall(r"(count|sum|avg|max|min)\s*\(", gt_lower)
    pred_agg = re.findall(r"(count|sum|avg|max|min)\s*\(", pred_lower)
    if gt_agg != pred_agg:
        return "wrong_aggregate"

    gt_tables = re.findall(r"from\s+(\w+)", gt_lower)
    pred_tables = re.findall(r"from\s+(\w+)", pred_lower)
    if gt_tables != pred_tables:
        return "wrong_table"

    if "group by" in gt_lower and "group by" not in pred_lower:
        return "missing_group_by"

    if "order by" in gt_lower and "order by" not in pred_lower:
        return "missing_order_by"

    if "where" in gt_lower and "where" not in pred_lower:
        return "missing_where"
    elif "where" in gt_lower and "where" in pred_lower:
        gt_conditions = (
            gt_lower.split("where")[1].split("group by")[0]
            if "group by" in gt_lower
            else gt_lower.split("where")[1]
        )
        pred_conditions = (
            pred_lower.split("where")[1].split("group by")[0]
            if "group by" in pred_lower
            else pred_lower.split("where")[1]
        )
        if gt_conditions.strip() != pred_conditions.strip():
            return "wrong_where_condition"

    return "other_error"


def analyze_errors(nl_file, ground_truth_file, predicted_file):
    """Analyze errors and categorize them."""

    with open(nl_file, "r") as f:
        nl_queries = [line.strip() for line in f.readlines()]

    with open(ground_truth_file, "r") as f:
        ground_truths = [line.strip() for line in f.readlines()]

    with open(predicted_file, "r") as f:
        predictions = [line.strip() for line in f.readlines()]

    error_categories = defaultdict(list)
    total_errors = 0

    for i, (nl, gt, pred) in enumerate(zip(nl_queries, ground_truths, predictions)):
        if normalize_sql(gt) != normalize_sql(pred):
            total_errors += 1
            error_type = categorize_error(nl, gt, pred)
            error_categories[error_type].append(
                {"index": i, "nl_query": nl, "ground_truth": gt, "prediction": pred}
            )

    print("=" * 80)
    print(f"ERROR ANALYSIS REPORT (Table 5)")
    print("=" * 80)
    print(f"Total examples: {len(ground_truths)}")
    print(f"Total errors: {total_errors}")
    print(
        f"Accuracy: {(len(ground_truths) - total_errors) / len(ground_truths) * 100:.2f}%"
    )
    print("=" * 80)

    sorted_categories = sorted(
        error_categories.items(), key=lambda x: len(x[1]), reverse=True
    )

    for category, errors in sorted_categories:
        count = len(errors)
        print(f"\n{category.upper().replace('_', ' ')}: {count}/{total_errors}")
        print("-" * 80)

        for idx, error in enumerate(errors[:3]):
            print(f"\nExample {idx + 1}:")
            print(f"  Natural Language: {error['nl_query']}")
            print(f"  Ground Truth:     {error['ground_truth']}")
            print(f"  Prediction:       {error['prediction']}")

        if len(errors) > 3:
            print(f"\n  ... and {len(errors) - 3} more examples")
        print()

    print("=" * 80)

    return error_categories, total_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/ft_experiments/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_folder", type=str, default="data", help="Path to data folder"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/predicted_dev.sql",
        help="Path to save predictions",
    )
    args = parser.parse_args()

    print("\nAnalyzing errors...")
    nl_file = os.path.join(args.data_folder, "dev.nl")
    gt_file = os.path.join(args.data_folder, "dev.sql")

    error_categories, total_errors = analyze_errors(nl_file, gt_file, args.output_file)

    print("\nError analysis complete!")


if __name__ == "__main__":
    main()
