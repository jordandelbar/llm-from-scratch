import os
import urllib.request

from pathlib import Path

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)

file_path = f"{Path(__file__).parents[1]}/data/the-verdict.txt"
if not os.path.exists(file_path):
    urllib.request.urlretrieve(url, file_path)
else:
    print("file already downloaded")

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character: ", len(raw_text))
print(raw_text[:99])
