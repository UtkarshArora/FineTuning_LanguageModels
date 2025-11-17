import os
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import T5TokenizerFast

PAD_IDX = 0


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        if self.bos_token_id == self.tokenizer.unk_token_id:
            # Fallback: just use pad as BOS if extra_id_0 not available
            self.bos_token_id = self.tokenizer.pad_token_id

        self.data = self.process_data(data_folder, split)

    # def load_and_simplify_schema(self, data_folder):
    #     """Load and simplify the schema"""
    #     schema_path = os.path.join(data_folder, "flight_database.schema")

    #     try:
    #         with open(schema_path, "r") as f:
    #             schema_data = json.load(f)

    #         # Get main table names
    #         main_tables = ["flight", "airline", "airport", "fare", "city"]
    #         schema_text = "Tables: " + ", ".join(main_tables)
    #         return schema_text

    #     except Exception as e:
    #         print(f"Warning: Could not load schema: {e}")
    #         return ""

    def process_data(self, data_folder, split):
        # Load natural language queries
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_queries = load_lines(nl_path)

        # Load SQL queries (if not test set)
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_queries = load_lines(sql_path)
        else:
            sql_queries = None

        # Tokenize data
        data = []
        prompt = "translate to SQL:"  # match hp style (lowercase)
        for i, nl_query in enumerate(nl_queries):
            input_text = f"{prompt} {nl_query}"

            # Tokenize encoder input
            encoder_ids = self.tokenizer.encode(
                input_text, add_special_tokens=True, max_length=512, truncation=True
            )

            item = {"encoder_ids": encoder_ids}

            # For train/dev, also tokenize decoder targets
            if sql_queries is not None:
                # Tokenize decoder output (SQL query)
                decoder_target_ids = self.tokenizer.encode(
                    sql_queries[i],
                    add_special_tokens=True,
                    max_length=256,
                    truncation=True,
                )
                # NEW: prepend BOS token so collate can shift correctly
                decoder_ids = [self.bos_token_id] + decoder_target_ids

                item["decoder_ids"] = decoder_ids

            data.append(item)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    # Extract encoder inputs
    encoder_ids = [torch.LongTensor(item["encoder_ids"]) for item in batch]
    decoder_ids = [torch.LongTensor(item["decoder_ids"]) for item in batch]

    # Pad sequences
    encoder_ids_padded = pad_sequence(
        encoder_ids, batch_first=True, padding_value=PAD_IDX
    )
    decoder_ids_padded = pad_sequence(
        decoder_ids, batch_first=True, padding_value=PAD_IDX
    )

    # Create attention mask for encoder (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids_padded != PAD_IDX).long()

    # Decoder inputs: all tokens except the last one
    decoder_inputs = decoder_ids_padded[:, :-1]

    # Decoder targets: all tokens except the first one (shifted by 1)
    decoder_targets = decoder_ids_padded[:, 1:]

    # Initial decoder input: just the first token of each sequence
    initial_decoder_inputs = decoder_ids_padded[:, 0:1]

    return (
        encoder_ids_padded,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )


# def test_collate_fn(batch):
#     # Extract encoder inputs
#     encoder_ids = [torch.LongTensor(item["encoder_ids"]) for item in batch]

#     # Pad sequences
#     encoder_ids_padded = pad_sequence(
#         encoder_ids, batch_first=True, padding_value=PAD_IDX
#     )

#     # Create attention mask for encoder
#     encoder_mask = (encoder_ids_padded != PAD_IDX).long()

#     # For test, we need initial decoder input (start token)
#     # Use extra_id_0 as the beginning token as suggested in the comments
#     tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
#     initial_token_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
#     initial_decoder_inputs = torch.LongTensor([[initial_token_id]] * len(batch))

#     return encoder_ids_padded, encoder_mask, initial_decoder_inputs


def test_collate_fn(batch):
    # Extract encoder inputs
    encoder_ids = [torch.LongTensor(item["encoder_ids"]) for item in batch]

    # Pad sequences
    encoder_ids_padded = pad_sequence(
        encoder_ids, batch_first=True, padding_value=PAD_IDX
    )

    # Create attention mask for encoder
    encoder_mask = (encoder_ids_padded != PAD_IDX).long()

    # NEW: use same BOS token as during training (<extra_id_0>, or pad as fallback)
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    bos_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    if bos_token_id == tokenizer.unk_token_id:
        bos_token_id = tokenizer.pad_token_id

    initial_decoder_inputs = torch.full((len(batch), 1), bos_token_id, dtype=torch.long)

    return encoder_ids_padded, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = "data"
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    dataloader = DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
