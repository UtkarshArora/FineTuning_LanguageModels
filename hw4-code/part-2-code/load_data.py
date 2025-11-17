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
            self.bos_token_id = self.tokenizer.pad_token_id

        self.data = self.process_data(data_folder, split)

    def process_data(self, data_folder, split):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_queries = load_lines(nl_path)

        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_queries = load_lines(sql_path)
        else:
            sql_queries = None

        data = []
        prompt = "translate to SQL:"
        for i, nl_query in enumerate(nl_queries):
            input_text = f"{prompt} {nl_query}"

            encoder_ids = self.tokenizer.encode(
                input_text, add_special_tokens=True, max_length=512, truncation=True
            )

            item = {"encoder_ids": encoder_ids}

            if sql_queries is not None:
                decoder_target_ids = self.tokenizer.encode(
                    sql_queries[i],
                    add_special_tokens=True,
                    max_length=256,
                    truncation=True,
                )
                decoder_ids = [self.bos_token_id] + decoder_target_ids

                item["decoder_ids"] = decoder_ids

            data.append(item)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    encoder_ids = [torch.LongTensor(item["encoder_ids"]) for item in batch]
    decoder_ids = [torch.LongTensor(item["decoder_ids"]) for item in batch]

    encoder_ids_padded = pad_sequence(
        encoder_ids, batch_first=True, padding_value=PAD_IDX
    )
    decoder_ids_padded = pad_sequence(
        decoder_ids, batch_first=True, padding_value=PAD_IDX
    )

    encoder_mask = (encoder_ids_padded != PAD_IDX).long()

    decoder_inputs = decoder_ids_padded[:, :-1]

    decoder_targets = decoder_ids_padded[:, 1:]

    initial_decoder_inputs = decoder_ids_padded[:, 0:1]

    return (
        encoder_ids_padded,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )


def test_collate_fn(batch):
    encoder_ids = [torch.LongTensor(item["encoder_ids"]) for item in batch]

    encoder_ids_padded = pad_sequence(
        encoder_ids, batch_first=True, padding_value=PAD_IDX
    )

    encoder_mask = (encoder_ids_padded != PAD_IDX).long()

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
