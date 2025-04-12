import re

from pathlib import Path


def main():
    text = read_the_verdict()
    preprocessed = simple_tokenizer(text=text)
    print(len(preprocessed))
    print(preprocessed[:30])


def read_the_verdict() -> str:
    with open(f"{Path(__file__).parents[1]}/data/the-verdict.txt") as f:
        text = f.read()
    return text


def simple_tokenizer(text: str):
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item for item in result if item.strip()]
    return result


if __name__ == "__main__":
    main()
