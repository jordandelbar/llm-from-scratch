import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 2. Working with text data""")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path

    return (Path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2.2 Tokenizing text""")
    return


@app.cell
def _(Path):
    with open(
        f"{Path(__file__).parents[1]}/data/the-verdict.txt", "r", encoding="utf-8"
    ) as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    return (raw_text,)


@app.cell
def _():
    import re

    text_1 = "Hello, world. This, is a test."
    result_1 = re.split(r"(\s)", text_1)
    print(result_1)
    return re, text_1


@app.cell
def _(re, text_1):
    result_2 = re.split(r"([,.]|\s)", text_1)
    print(result_2)
    return (result_2,)


@app.cell
def _(result_2):
    result_3 = [item for item in result_2 if item.strip()]
    print(result_3)
    return


@app.cell
def _(re):
    text_2 = "Hello, world. Is this-- a test?"
    result_4 = re.split(r"([,.:;?_!\"()']|--|\s)", text_2)
    result_4 = [item.strip() for item in result_4 if item.strip()]
    print(result_4)
    return


@app.cell
def _(raw_text, re):
    preprocessed = re.split(r"([,.:;?_!\"()']|--|\s)", raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(len(preprocessed))
    print(preprocessed[:30])
    return (preprocessed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2.3 Converting tokens into token IDs""")
    return


@app.cell
def _(preprocessed):
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)
    return (all_words,)


@app.cell
def _(all_words):
    vocab_1 = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab_1.items()):
        print(item)
        if i >= 50:
            break
    return (vocab_1,)


@app.cell
def _(re):
    class SimpleTokenizerV1:
        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r"([,.?_!\"()']|--|\s)", text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            text = re.sub(r"\s+([,.?!()'])", r"\1", text)
            return text

    return (SimpleTokenizerV1,)


@app.cell
def _(SimpleTokenizerV1, vocab_1):
    tokenizer_1 = SimpleTokenizerV1(vocab_1)
    text_3 = """"It's the last he painted, you know,"
              Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer_1.encode(text_3)
    print(ids)
    return ids, tokenizer_1


@app.cell
def _(ids, tokenizer_1):
    print(tokenizer_1.decode(ids))
    return


@app.cell
def _(tokenizer_1):
    try:
        text_4 = "Hello, do you like tea?"
        print(tokenizer_1.encode(text_4))
    except KeyError as e:
        print(f"key error, {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2.4 Adding special context tokens""")
    return


@app.cell
def _(preprocessed):
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_2 = {token: integer for integer, token in enumerate(all_tokens)}
    print(len(vocab_2.items()))
    return (vocab_2,)


@app.cell
def _(vocab_2):
    for _, item_1 in enumerate(list(vocab_2.items())[-5:]):
        print(item_1)
    return


@app.cell
def _(re):
    class SimpleTokenizerV2:
        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r"([,.:;?_!\"()']|--|\s)", text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            preprocessed = [
                item if item in self.str_to_int else "<|unk|>" for item in preprocessed
            ]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            text = re.sub(r"\s+([,.:;?!\"()'])", r"\1", text)
            return text

    return (SimpleTokenizerV2,)


@app.cell
def _():
    text_5 = "Hello, do you like tea?"
    text_6 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text_5, text_6))
    print(text)
    return (text,)


@app.cell
def _(SimpleTokenizerV2, text, vocab_2):
    tokenizer_2 = SimpleTokenizerV2(vocab_2)
    print(tokenizer_2.encode(text))
    print(tokenizer_2.decode(tokenizer_2.encode(text)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2.5 Byte pair encoding""")
    return


@app.cell
def _():
    import tiktoken

    return (tiktoken,)


@app.cell
def _(tiktoken):
    tokenizer_3 = tiktoken.get_encoding("gpt2")
    return (tokenizer_3,)


@app.cell
def _(tokenizer_3):
    text_7 = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
        "of someunknownPlace."
    )
    integers = tokenizer_3.encode(text_7, allowed_special={"<|endoftext|>"})
    print(integers)
    return (integers,)


@app.cell
def _(integers, tokenizer_3):
    strings = tokenizer_3.decode(integers)
    print(strings)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2.6 Data sampling with a sliding window""")
    return


@app.cell
def _(raw_text, tokenizer_3):
    enc_text = tokenizer_3.encode(raw_text)
    print(len(enc_text))
    return (enc_text,)


@app.cell
def _(enc_text):
    enc_sample = enc_text[50:]
    return (enc_sample,)


@app.cell
def _(enc_sample):
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1 : context_size + 1]
    print(f"x: {x}")
    print(f"y:      {y}")
    return (context_size,)


@app.cell
def _(context_size, enc_sample):
    for j in range(1, context_size + 1):
        context_1 = enc_sample[:j]
        desired_1 = enc_sample[j]
        print(context_1, "---->", desired_1)
    return


@app.cell
def _(context_size, enc_sample, tokenizer_3):
    for k in range(1, context_size + 1):
        context_2 = enc_sample[:k]
        desired_2 = enc_sample[k]
        print(tokenizer_3.decode(context_2), "---->", tokenizer_3.decode([desired_2]))
    return


@app.cell
def _():
    from typing import List, Tuple

    import torch
    from torch.utils.data import Dataset, DataLoader
    from tiktoken import Encoding

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

    return DataLoader, GPTDatasetV1, torch


@app.cell
def _(DataLoader, GPTDatasetV1, tiktoken):
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

    return (create_dataloader_v1,)


@app.cell
def _(create_dataloader_v1, raw_text):
    dataloader_1 = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter_1 = iter(dataloader_1)
    first_batch = next(data_iter_1)
    print(first_batch)
    return (data_iter_1,)


@app.cell
def _(data_iter_1):
    second_batch = next(data_iter_1)
    print(second_batch)
    return


@app.cell
def _(create_dataloader_v1, raw_text):
    dataloader_2 = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    data_iter_2 = iter(dataloader_2)
    inputs, targets = next(data_iter_2)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    return (inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2.7 Creating token embeddings""")
    return


@app.cell
def _(torch):
    input_ids = torch.tensor([2, 3, 5, 1])

    second_vocab_size = 6
    output_dim = 3
    return input_ids, output_dim, second_vocab_size


@app.cell
def _(output_dim, second_vocab_size, torch):
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(second_vocab_size, output_dim)
    print(embedding_layer.weight)
    return (embedding_layer,)


@app.cell
def _(embedding_layer, torch):
    print(embedding_layer(torch.tensor([3])))
    return


@app.cell
def _(embedding_layer, input_ids):
    print(embedding_layer(input_ids))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2.8 Encoding word position""")
    return


@app.cell
def _(create_dataloader_v1, raw_text, torch):
    third_vocab_size = 50257
    second_output_dim = 256
    token_embedding_layer = torch.nn.Embedding(third_vocab_size, second_output_dim)

    max_length = 4
    dataloader_3 = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    data_iter_3 = iter(dataloader_3)
    second_inputs, second_targets = next(data_iter_3)

    print("Token IDs:\n", second_inputs)
    print("\nInput shape:\n", second_inputs.shape)
    return max_length, second_output_dim, token_embedding_layer


@app.cell
def _(inputs, token_embedding_layer):
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    return (token_embeddings,)


@app.cell
def _(max_length, second_output_dim, torch):
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, second_output_dim)
    pos_embedding = pos_embedding_layer(torch.arange(context_length))
    print(pos_embedding.shape)
    return (pos_embedding,)


@app.cell
def _(pos_embedding, token_embeddings):
    input_embeddings = token_embeddings + pos_embedding
    print(input_embeddings.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
