import re

from pathlib import Path
from typing import Dict, List

from simple_tokenizer import SimpleTokenizerV2


def main():
    text = read_the_verdict()
    preprocessed = simple_tokenizer(text=text)
    print(len(preprocessed))
    print(preprocessed[:30])
    vocab = get_vocab(preprocessed)
    tokenizer = SimpleTokenizerV2(vocab)
    text = """"It's the last he painted, you know,"
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))
    text1 = "Hello, do you like tea"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))


def read_the_verdict() -> str:
    with open(f"{Path(__file__).parents[1]}/data/the-verdict.txt") as f:
        text = f.read()
    return text


def simple_tokenizer(text: str) -> List[str]:
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item for item in result if item.strip()]
    return result


def get_vocab(preprocessed: List[str]) -> Dict[str, int]:
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_tokens)
    print(f"{vocab_size=}")
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab


if __name__ == "__main__":
    main()
