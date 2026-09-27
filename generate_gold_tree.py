"""
Author: Jayendra Matarage
Created on: 9/25/2026 1:23 PM
Description:
"""
import re
from pathlib import Path

from nltk.tree import Tree


INPUT_PATH = Path("daily_mirror_data.reviewed.pos")
OUTPUT_PATH = Path("daily_mirror_data.gold_tree.txt")
TREE_HEADER = re.compile(r"(?m)^--- Gold Tree \d+ ---\s*$")


def parse_tagged_sentence(line: str, line_number: int) -> list[tuple[str, str]]:
    """Return ``(word, tag)`` pairs from one reviewed POS-tagged sentence."""
    tagged_words = []

    for token in line.split():
        if "/" not in token:
            raise ValueError(
                f"Line {line_number}: token {token!r} has no POS tag"
            )

        word, tag = token.rsplit("/", 1)
        if not word or not tag:
            raise ValueError(
                f"Line {line_number}: invalid word/POS token {token!r}"
            )

        tagged_words.append((word, tag))

    if not tagged_words:
        raise ValueError(f"Line {line_number}: sentence is empty")

    return tagged_words


def generate_gold_trees(input_path: Path = INPUT_PATH) -> list[Tree]:
    """Build one flat, clean POS-reference tree per non-empty input line."""
    trees = []

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            tagged_words = parse_tagged_sentence(line, line_number)
            pos_nodes = [Tree(tag, [word]) for word, tag in tagged_words]
            trees.append(Tree("S", pos_nodes))

    return trees


def save_gold_trees(trees: list[Tree], output_path: Path = OUTPUT_PATH) -> None:
    """Write numbered blocks that can be loaded with ``Tree.fromstring``."""
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for index, tree in enumerate(trees, start=1):
            output.write(f"--- Gold Tree {index} ---\n")
            output.write(tree.pformat(margin=120))
            output.write("\n\n")


def load_gold_trees(input_path: Path = OUTPUT_PATH) -> list[Tree]:
    """Load numbered tree blocks and validate their bracket syntax."""
    content = input_path.read_text(encoding="utf-8")
    blocks = TREE_HEADER.split(content)[1:]
    return [Tree.fromstring(block.strip()) for block in blocks if block.strip()]


def main() -> None:
    trees = generate_gold_trees()
    save_gold_trees(trees)

    # Reloading verifies that every emitted block is valid Tree.fromstring input.
    loaded_trees = load_gold_trees()
    if len(loaded_trees) != len(trees):
        raise RuntimeError(
            f"Wrote {len(trees)} trees but reloaded {len(loaded_trees)} trees"
        )

    print(f"Generated and validated {len(trees)} clean trees in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
