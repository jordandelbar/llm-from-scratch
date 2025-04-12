import os
import urllib.request

from pathlib import Path

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)


def download_the_verdict():
    file_path = f"{Path(__file__).parents[1]}/data/the-verdict.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    else:
        print("file already downloaded")
