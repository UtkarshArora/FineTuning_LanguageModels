import os
import argparse
from tqdm import tqdm
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from load_data import T5Dataset
from torch.utils.data import DataLoader
import re
from collections import defaultdict


# def generate_predictions(model, tokenizer, data_folder, output_file):
#     """Generate predictions on dev set and save to file."""
#     dev_dataset = T5Dataset(data_folder, "dev")
#     dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

#     model.eval()
#     predictions = []

#     with torch.no_grad():
#         for batch in dev_loader:
#             encoder_ids = batch["encoder_ids"].to(model.device)
#             attention_mask = batch["attention_mask"].to(model.device)

#             outputs = model.generate(
#                 input_ids=encoder_ids,
#                 attention_mask=attention_mask,
#                 max_length=512,
#                 num_beams=2,
#                 early_stopping=True,
#             )

#             for output in outputs:
#                 pred_sql = tokenizer.decode(output, skip_special_tokens=True)
#                 predictions.append(pred_sql)

#     # Save predictions
#     with open(output_file, "w") as f:
#         for pred in predictions:
#             f.write(pred + "\n")

#     return predictions


def normalize_sql(s: str) -> str:
    return " ".join(s.lower().split())


def categorize_error(nl_query, ground_truth, prediction):
    """Categorize the type of error based on ground truth vs prediction."""

    # Normalize both queries for comparison
    gt_lower = ground_truth.lower()
    pred_lower = prediction.lower()

    # Category 1: Missing or incorrect JOIN
    if "join" in gt_lower and "join" not in pred_lower:
        return "missing_join"
    elif "join" in gt_lower and "join" in pred_lower:
        # Check if JOIN type is wrong
        gt_join_type = re.search(r"(inner|left|right|outer)\s+join", gt_lower)
        pred_join_type = re.search(r"(inner|left|right|outer)\s+join", pred_lower)
        if (
            gt_join_type
            and pred_join_type
            and gt_join_type.group(1) != pred_join_type.group(1)
        ):
            return "wrong_join_type"

    # Category 2: Missing or incorrect WHERE clause
    if "where" in gt_lower and "where" not in pred_lower:
        return "missing_where"
    elif "where" in gt_lower and "where" in pred_lower:
        # Check for incorrect conditions
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

    # Category 3: Wrong aggregate function
    gt_agg = re.findall(r"(count|sum|avg|max|min)\s*\(", gt_lower)
    pred_agg = re.findall(r"(count|sum|avg|max|min)\s*\(", pred_lower)
    if gt_agg != pred_agg:
        return "wrong_aggregate"

    # Category 4: Incorrect table or column names
    gt_tables = re.findall(r"from\s+(\w+)", gt_lower)
    pred_tables = re.findall(r"from\s+(\w+)", pred_lower)
    if gt_tables != pred_tables:
        return "wrong_table"

    # Category 5: Missing or incorrect GROUP BY
    if "group by" in gt_lower and "group by" not in pred_lower:
        return "missing_group_by"

    # Category 6: Missing or incorrect ORDER BY
    if "order by" in gt_lower and "order by" not in pred_lower:
        return "missing_order_by"

    # Default: other error
    return "other_error"


def analyze_errors(nl_file, ground_truth_file, predicted_file):
    """Analyze errors and categorize them."""

    # Read files
    with open(nl_file, "r") as f:
        nl_queries = [line.strip() for line in f.readlines()]

    with open(ground_truth_file, "r") as f:
        ground_truths = [line.strip() for line in f.readlines()]

    with open(predicted_file, "r") as f:
        predictions = [line.strip() for line in f.readlines()]

    # Categorize errors
    error_categories = defaultdict(list)
    total_errors = 0

    for i, (nl, gt, pred) in enumerate(zip(nl_queries, ground_truths, predictions)):
        if normalize_sql(gt) != normalize_sql(pred):
            total_errors += 1
            error_type = categorize_error(nl, gt, pred)
            error_categories[error_type].append(
                {"index": i, "nl_query": nl, "ground_truth": gt, "prediction": pred}
            )

    # Print error analysis report
    print("=" * 80)
    print(f"ERROR ANALYSIS REPORT (Table 5)")
    print("=" * 80)
    print(f"Total examples: {len(ground_truths)}")
    print(f"Total errors: {total_errors}")
    print(
        f"Accuracy: {(len(ground_truths) - total_errors) / len(ground_truths) * 100:.2f}%"
    )
    print("=" * 80)

    # Sort by frequency
    sorted_categories = sorted(
        error_categories.items(), key=lambda x: len(x[1]), reverse=True
    )

    for category, errors in sorted_categories:
        count = len(errors)
        print(f"\n{category.upper().replace('_', ' ')}: {count}/{total_errors}")
        print("-" * 80)

        # Show first 3 examples
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

    # Load model and tokenizer
    # print("Loading model and tokenizer...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    # Load checkpoint if exists
    # if os.path.exists(args.model_path):
    #     checkpoint = torch.load(args.model_path, map_location=device)
    #     model.load_state_dict(checkpoint)
    #     print(f"Loaded checkpoint from {args.model_path}")

    # model = model.to(device)
    # tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    # # # Generate predictions
    # # print("Generating predictions...")
    # # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # # predictions = generate_predictions(
    # #     model, tokenizer, args.data_folder, args.output_file
    # # )
    # print(f"Predictions saved to {args.output_file}")

    # Analyze errors
    print("\nAnalyzing errors...")
    nl_file = os.path.join(args.data_folder, "dev.nl")
    gt_file = os.path.join(args.data_folder, "dev.sql")

    error_categories, total_errors = analyze_errors(nl_file, gt_file, args.output_file)

    print("\nError analysis complete!")


if __name__ == "__main__":
    main()
