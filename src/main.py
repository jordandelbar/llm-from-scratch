from pathlib import Path

import tiktoken


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of someunknownPlace."
    text = " <|endoftext|> ".join((text1, text2))
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    string = tokenizer.decode(integers)
    print(string)
    text = "Akwirw ier."
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    string = tokenizer.decode(integers)
    print(string)


def read_the_verdict() -> str:
    with open(f"{Path(__file__).parents[1]}/data/the-verdict.txt") as f:
        text = f.read()
    return text


if __name__ == "__main__":
    main()
