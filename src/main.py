from pathlib import Path
from typing import List, Tuple

import tiktoken
from tiktoken import Encoding
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: Encoding, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def main():
    raw_text = read_the_verdict()

    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)


def read_the_verdict() -> str:
    with open(f"{Path(__file__).parents[1]}/data/the-verdict.txt") as f:
        text = f.read()
    return text


if __name__ == "__main__":
    main()
