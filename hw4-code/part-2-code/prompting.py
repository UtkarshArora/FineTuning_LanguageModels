import os, argparse, random
from tqdm import tqdm

import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import (
    set_random_seeds,
    compute_metrics,
    save_queries_and_records,
    compute_records,
)
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device('mps') if torch.backends.mps.is_available() else DEVICE # for MacOS


def get_args():
    """
    Arguments for prompting. You may choose to change or extend these as you see fit.
    """
    parser = argparse.ArgumentParser(
        description="Text-to-SQL experiments with prompting."
    )

    parser.add_argument(
        "-s",
        "--shot",
        type=int,
        default=0,
        help="Number of examples for k-shot learning (0 for zero-shot)",
    )
    parser.add_argument(
        "-p",
        "--ptype",
        type=int,
        default=0,
        help="Prompt type (e.g., 0 for base, 1 for more complex)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gemma",
        help="Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        action="store_true",
        help="Use a quantized version of the model (e.g. 4bits)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    return args


def load_model_and_tokenizer(model_name, quantization):
    """
    Load the model and tokenizer from Huggingface.
    """
    print(f"Loading model: {model_name} (Quantization: {quantization})")

    model_id_map = {
        "gemma": "google/gemma-1.1-2b-it",
        "codegemma": "google/codegemma-7b-it",  # Note: 7b model is much larger
    }

    model_id = model_id_map.get(model_name, model_name)

    bnb_config = None
    if quantization:
        print("Using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically map to GPU if available
    )
    print(f"Model {model_id} loaded successfully.")
    return tokenizer, model


def create_prompt(schema, prompt_type, k_shot_examples, nl_query):
    """
    Construct the final prompt string to be fed to the model.
    """

    # System Instruction
    # We define the task and provide the database schema
    prompt = f"""<start_of_turn>user
    You are an expert Text-to-SQL model. Your task is to generate a SQL query for a given natural language instruction.
    You must only output the SQL query and nothing else.

    Here is the database schema:
    {schema}

    """

    # Add K-shot examples (In-Context Learning)
    if k_shot_examples:
        prompt += "Here are some examples of instructions and their corresponding SQL queries:\n\n"
        for nl, sql in k_shot_examples:
            prompt += f"Instruction: {nl}\n"
            prompt += f"SQL: {sql}\n\n"

    # Add the target query
    # Prompt type can be used to experiment with different prompt structures
    if prompt_type == 0:
        prompt += f"Generate a SQL query for the following instruction:\nInstruction: {nl_query}\nSQL:"
    else:
        # Example of a different prompt type
        prompt += f"--- New Task ---\nInstruction: {nl_query}\n"
        prompt += "SQL Query:"

    prompt += "<end_of_turn>\n<start_of_turn>model\n"

    return prompt


def exp_kshot(tokenizer, model, schema, train_x, train_y, eval_x, k, ptype):
    """
    Run the K-shot experiment.
    Generates a prompt for each item in eval_x, gets the model's
    response, and extracts the SQL query.
    """
    raw_outputs = []
    extracted_queries = []

    # 1. Select K random examples if k > 0
    k_shot_examples = []
    if k > 0 and train_x:
        # Ensure we don't pick more examples than we have
        num_examples = min(k, len(train_x))
        # Randomly sample k (nl, sql) pairs
        k_shot_examples = random.sample(list(zip(train_x, train_y)), num_examples)

    # 2. Iterate through the evaluation set and generate queries
    print(f"Generating queries for {len(eval_x)} examples...")
    for nl_query in tqdm(eval_x):
        # 3. Create the prompt
        prompt = create_prompt(schema, ptype, k_shot_examples, nl_query)

        # 4. Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Note: max_new_tokens can be adjusted.
        # 512 is long, but SQL queries can be complex.
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, temperature=0.1, do_sample=False
            )

        # 5. Decode the output
        # We skip special tokens and only decode the *newly generated* part
        raw_output = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        raw_outputs.append(raw_output)

        # 6. Extract the SQL query from the raw output
        sql_query = extract_sql_query(raw_output)
        extracted_queries.append(sql_query)

    return raw_outputs, extracted_queries


def eval_outputs(
    eval_x,
    eval_y,
    extracted_queries,
    gt_path,
    model_path,
    gt_query_records,
    model_query_records,
):
    """
    Save the generated queries and compute metrics.
    """
    # 1. Save the model's generated queries and compute their records
    print(f"Saving {len(extracted_queries)} queries to {model_path}...")
    save_queries_and_records(extracted_queries, model_path, model_query_records)

    # 2. If we have ground truth, compute metrics
    if eval_y:
        print("Computing metrics against ground truth...")
        sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
            gt_path=gt_path,
            model_path=model_path,
            gt_query_records=gt_query_records,
            model_query_records=model_query_records,
        )
        num_errors = sum(1 for msg in model_error_msgs if msg)
        error_rate = num_errors / len(model_error_msgs) if model_error_msgs else 0.0

        return sql_em, record_em, record_f1, model_error_msgs, error_rate
    else:
        # Test set - no ground truth available
        print("Test set predictions saved. No metrics computed.")
        return 0.0, 0.0, 0.0, [], 0.0


def main():

    os.makedirs("results", exist_ok=True)
    os.makedirs("records", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    args = get_args()
    set_random_seeds(args.seed)
    random.seed(args.seed)

    # 1. Load data
    print("Loading data...")
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data("data")

    # 2. Read schema
    schema_path = "data/flight_database.schema"
    schema = read_schema(schema_path)
    if not schema:
        print("Warning: Schema is empty. Queries may not be accurate.")
    # 3. Load model
    tokenizer, model = load_model_and_tokenizer(args.model, args.quantization)

    k = args.shot
    ptype = args.ptype
    experiment_name = f"{args.model}_k{k}_p{ptype}"
    if args.quantization:
        experiment_name += "_quant"

    # --- Run Evaluation ---
    # for eval_split in ["dev", "test"]:
    #     print(f"\n--- Running evaluation on {eval_split} set ---")

    #     if eval_split == "dev":
    #         eval_x, eval_y = dev_x, dev_y
    #         gt_sql_path = "data/dev.sql"
    #         gt_record_path = "records/ground_truth_dev.pkl"  # Pre-computed GT records
    #     else:
    #         eval_x, eval_y = test_x, None  # No labels for test set
    #         gt_sql_path = "data/test.sql"  # Does not exist, but path is needed
    #         gt_record_path = None

    #     # 4. Run K-shot experiment
    #     raw_outputs, extracted_queries = exp_kshot(
    #         tokenizer, model, schema, train_x, train_y, eval_x, k, ptype
    #     )

    #     # 5. Evaluate outputs
    #     # Define output paths
    #     if eval_split == "dev":
    #         model_sql_path = f"results/{experiment_name}_dev.sql"
    #         model_record_path = f"records/{experiment_name}_dev.pkl"
    #     else:
    #         model_sql_path = "results/gemma_test.sql"
    #         model_record_path = "records/gemma_test.pkl"
    #     # model_sql_path = os.path.join(f"results/{experiment_name}_{eval_split}.sql")
    #     # model_record_path = os.path.join(f"records/{experiment_name}_{eval_split}.pkl")
    #     log_path = os.path.join(f"logs/{experiment_name}_{eval_split}.log")

    #     sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
    #         eval_x,
    #         eval_y,
    #         extracted_queries,
    #         gt_path=gt_sql_path,
    #         model_path=model_sql_path,
    #         gt_query_records=gt_record_path,
    #         model_query_records=model_record_path,
    #     )

    #     if eval_split == "dev":
    #         print(f"\n--- {eval_split} set results ---")
    #         print(
    #             f"Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}"
    #         )
    #         print(f"{error_rate*100:.2f}% of generated queries led to SQL errors.")
    #         # Save logs for the dev set
    #         save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)
    for eval_split in ["dev", "test"]:
        print(f"\n--- Running evaluation on {eval_split} set ---")

        if eval_split == "dev":
            eval_x, eval_y = dev_x, dev_y
            gt_sql_path = "data/dev.sql"
            gt_record_path = "records/ground_truth_dev.pkl"
            model_sql_path = f"results/{experiment_name}_dev.sql"
            model_record_path = f"records/{experiment_name}_dev.pkl"
        else:
            eval_x, eval_y = test_x, None
            model_sql_path = "results/gemma_test.sql"
            model_record_path = "records/gemma_test.pkl"

        # 4. Run K-shot experiment
        raw_outputs, extracted_queries = exp_kshot(
            tokenizer, model, schema, train_x, train_y, eval_x, k, ptype
        )

        # 5. Evaluate
        if eval_split == "dev":
            log_path = f"logs/{experiment_name}_dev.log"

            # FIX: Unpack 5 values, not 4
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                eval_x,
                eval_y,
                extracted_queries,
                gt_path=gt_sql_path,
                model_path=model_sql_path,
                gt_query_records=gt_record_path,
                model_query_records=model_record_path,
            )

            print(f"\n--- {eval_split} set results ---")
            print(
                f"Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}"
            )
            print(f"{error_rate*100:.2f}% of generated queries led to SQL errors.")
            save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)
        else:
            # Test set - just save predictions
            print(f"Saving test set predictions...")
            save_queries_and_records(
                extracted_queries, model_sql_path, model_record_path
            )
            print(f"Test predictions saved to {model_sql_path}")

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
