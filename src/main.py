import re

from pathlib import Path
from typing import Dict, List

from simple_tokenizer import SimpleTokenizerV1


def main():
    text = read_the_verdict()
    preprocessed = simple_tokenizer(text=text)
    print(len(preprocessed))
    print(preprocessed[:30])
    vocab = get_vocab(preprocessed)
    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)


def read_the_verdict() -> str:
    with open(f"{Path(__file__).parents[1]}/data/the-verdict.txt") as f:
        text = f.read()
    return text


def simple_tokenizer(text: str) -> List[str]:
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item for item in result if item.strip()]
    return result


def get_vocab(tokens: List[str]) -> Dict[str, int]:
    all_words = sorted(set(tokens))
    vocab_size = len(all_words)
    print(f"{vocab_size=}")
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab


if __name__ == "__main__":
    main()
