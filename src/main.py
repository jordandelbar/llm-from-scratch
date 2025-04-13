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
    vocab_size = 50257
    output_dim = 256

    raw_text = read_the_verdict()
    torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    token_embeddings = token_embedding_layer(inputs)
    print("Tokens embeddings shape:\n", token_embeddings.shape)
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print("Positional embeddings shape:\n", pos_embeddings.shape)
    input_embeddings = token_embeddings + pos_embeddings
    print("Input embeddings shape:\n", input_embeddings.shape)
    simple_attention_mechanism()


def simple_attention_mechanism():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your
            [0.55, 0.87, 0.66],  # journey
            [0.57, 0.85, 0.64],  # starts
            [0.22, 0.58, 0.33],  # with
            [0.77, 0.25, 0.10],  # one
            [0.05, 0.80, 0.55],
        ]  # step
    )
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    print(attn_scores_2)
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention wieghts:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
    print(context_vec_2)
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    print(attn_scores)
    attn_scores = inputs @ inputs.T
    print(attn_scores)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)
    all_context_weights = attn_weights @ inputs
    print(all_context_weights)


def read_the_verdict() -> str:
    with open(f"{Path(__file__).parents[1]}/data/the-verdict.txt") as f:
        text = f.read()
    return text


if __name__ == "__main__":
    main()
