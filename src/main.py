import re


def main():
    simple_tokenizer()


def simple_tokenizer():
    text = "Hello world. This, is a test"

    result = re.split(r"(\s)", text)
    print(result)


if __name__ == "__main__":
    main()
