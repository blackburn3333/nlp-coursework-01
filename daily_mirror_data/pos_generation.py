"""
Author: Jayendra Matarage
Created on: 9/25/2026 1:23 PM
Description: 
"""
import nltk
from nltk.tokenize import word_tokenize

for resource in ["punkt_tab", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"taggers/{resource}" if "tagger" in resource else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)


def generate_pos_file(input_raw_path="daily_mirror_data.raw", output_pos_path="daily_mirror_data.pos"):
    with open(input_raw_path, "r", encoding="utf-8") as raw_file:
        raw_sentences = [line.strip() for line in raw_file if line.strip()]

    tagged_lines = []

    for sentence in raw_sentences:
        tokens = word_tokenize(sentence)

        pos_tuples = nltk.pos_tag(tokens)

        tagged_sentence = " ".join([f"{word}/{tag}" for word, tag in pos_tuples])
        tagged_lines.append(tagged_sentence)

    with open(output_pos_path, "w", encoding="utf-8") as pos_file:
        for line in tagged_lines:
            pos_file.write(f"{line}\n")

    print(f"Successfully tagged {len(tagged_lines)} sentences and saved to '{output_pos_path}'.")


if __name__ == "__main__":
    generate_pos_file()