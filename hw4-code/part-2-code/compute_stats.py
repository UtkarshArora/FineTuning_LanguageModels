from transformers import T5TokenizerFast
from load_data import load_lines, T5Dataset
import numpy as np


def compute_statistics(nl_path, sql_path, tokenizer, prefix: str = ""):
    nl_lines = load_lines(nl_path)
    sql_lines = load_lines(sql_path)

    assert len(nl_lines) == len(
        sql_lines
    ), f"NL and SQL files must have same length: {len(nl_lines)} vs {len(sql_lines)}"

    num_examples = len(nl_lines)

    nl_lengths = []
    sql_lengths = []
    nl_vocab = set()
    sql_vocab = set()

    for nl, sql in zip(nl_lines, sql_lines):
        nl_text = f"{prefix} {nl}" if prefix else nl

        nl_tokens = tokenizer.encode(nl_text, add_special_tokens=False)
        sql_tokens = tokenizer.encode(sql, add_special_tokens=False)

        nl_lengths.append(len(nl_tokens))
        sql_lengths.append(len(sql_tokens))

        nl_vocab.update(nl_tokens)
        sql_vocab.update(sql_tokens)

    stats = {
        "num_examples": num_examples,
        "mean_nl_length": float(np.mean(nl_lengths)) if nl_lengths else 0.0,
        "mean_sql_length": float(np.mean(sql_lengths)) if sql_lengths else 0.0,
        "vocab_size_nl": len(nl_vocab),
        "vocab_size_sql": len(sql_vocab),
    }
    return stats


def compute_statistics_from_dataset(dataset, tokenizer=None):
    nl_lengths = []
    sql_lengths = []
    nl_vocab = set()
    sql_vocab = set()

    for item in dataset:
        encoder_ids = item["encoder_ids"]

        if hasattr(encoder_ids, "tolist"):
            encoder_tokens = encoder_ids.tolist()
        else:
            encoder_tokens = list(encoder_ids)

        nl_lengths.append(len(encoder_tokens))
        nl_vocab.update(encoder_tokens)

        if "decoder_ids" in item:
            decoder_ids = item["decoder_ids"]
            if hasattr(decoder_ids, "tolist"):
                decoder_tokens = decoder_ids.tolist()
            else:
                decoder_tokens = list(decoder_ids)

            if decoder_tokens:
                sql_tokens = decoder_tokens[1:]
            else:
                sql_tokens = []

            sql_lengths.append(len(sql_tokens))
            sql_vocab.update(sql_tokens)

    stats = {
        "num_examples": len(dataset),
        "mean_nl_length": float(np.mean(nl_lengths)) if nl_lengths else 0.0,
        "mean_sql_length": float(np.mean(sql_lengths)) if sql_lengths else 0.0,
        "vocab_size_nl": len(nl_vocab),
        "vocab_size_sql": len(sql_vocab),
    }
    return stats


def main():
    print("Loading T5 tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    data_folder = "data"

    print("\n" + "=" * 70)
    print("Q4: Data Statistics")
    print("=" * 70)

    print("\nComputing statistics BEFORE preprocessing (Table 1)...")
    train_stats_before = compute_statistics(
        f"{data_folder}/train.nl",
        f"{data_folder}/train.sql",
        tokenizer,
    )
    dev_stats_before = compute_statistics(
        f"{data_folder}/dev.nl",
        f"{data_folder}/dev.sql",
        tokenizer,
    )

    print("\nComputing statistics AFTER preprocessing (Table 2)...")
    try:
        train_dataset = T5Dataset(data_folder, "train")
        dev_dataset = T5Dataset(data_folder, "dev")

        train_stats_after = compute_statistics_from_dataset(train_dataset)
        dev_stats_after = compute_statistics_from_dataset(dev_dataset)

        print("✓ Successfully computed statistics using T5Dataset")
    except Exception as e:
        print(f"⚠ Warning: Could not use T5Dataset: {e}")
        print("Falling back to manual preprocessing...")

        prefix = "translate to SQL:"

        train_stats_after = compute_statistics(
            f"{data_folder}/train.nl",
            f"{data_folder}/train.sql",
            tokenizer,
            prefix=prefix,
        )
        dev_stats_after = compute_statistics(
            f"{data_folder}/dev.nl",
            f"{data_folder}/dev.sql",
            tokenizer,
            prefix=prefix,
        )

    print("\n--- TABLE 1 (Before preprocessing) ---")
    print(f"Train examples: {train_stats_before['num_examples']}")
    print(f"Dev examples: {dev_stats_before['num_examples']}")
    print(f"Train mean NL length: {train_stats_before['mean_nl_length']:.2f}")
    print(f"Dev mean NL length: {dev_stats_before['mean_nl_length']:.2f}")
    print(f"Train mean SQL length: {train_stats_before['mean_sql_length']:.2f}")
    print(f"Dev mean SQL length: {dev_stats_before['mean_sql_length']:.2f}")
    print(f"Train NL vocab size: {train_stats_before['vocab_size_nl']}")
    print(f"Dev NL vocab size: {dev_stats_before['vocab_size_nl']}")
    print(f"Train SQL vocab size: {train_stats_before['vocab_size_sql']}")
    print(f"Dev SQL vocab size: {dev_stats_before['vocab_size_sql']}")

    print("\n--- TABLE 2 (After preprocessing) ---")
    print("Model name: T5-small")
    print(f"Train mean NL length: {train_stats_after['mean_nl_length']:.2f}")
    print(f"Dev mean NL length: {dev_stats_after['mean_nl_length']:.2f}")
    print(f"Train mean SQL length: {train_stats_after['mean_sql_length']:.2f}")
    print(f"Dev mean SQL length: {dev_stats_after['mean_sql_length']:.2f}")
    print(f"Train NL vocab size: {train_stats_after['vocab_size_nl']}")
    print(f"Dev NL vocab size: {dev_stats_after['vocab_size_nl']}")
    print(f"Train SQL vocab size: {train_stats_after['vocab_size_sql']}")
    print(f"Dev SQL vocab size: {dev_stats_after['vocab_size_sql']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
