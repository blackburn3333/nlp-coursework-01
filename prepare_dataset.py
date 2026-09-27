"""
Author: Jayendra Matarage
Created on: 9/25/2026 1:23 PM
Description:
"""
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REVIEWED_POS_PATH = PROJECT_DIR / "daily_mirror_data.reviewed.pos"
CLEANED_RAW_PATH = PROJECT_DIR / "daily_mirror_data.cleaned.raw"
EXPECTED_SENTENCES = 150

VALID_PTB_TAGS = {
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
    "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$",
    "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
    "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", ".", ",", ":",
    "``", "''", "$", "#", "-LRB-", "-RRB-",
}

HIDDEN_CHARACTERS = {"\u200b", "\u200c", "\u200d", "\ufeff"}


def read_nonempty_lines(path: Path) -> list[str]:
    """Read non-empty UTF-8 lines without changing their token contents."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_reviewed_sentence(
    line: str, sentence_number: int
) -> list[tuple[str, str]]:
    """Parse and validate one whitespace-delimited ``word/POS`` sentence."""
    tagged_words = []

    for token_number, token in enumerate(line.split(), start=1):
        if "/" not in token:
            raise ValueError(
                f"Sentence {sentence_number}, token {token_number}: "
                f"missing '/' in {token!r}"
            )

        word, tag = token.rsplit("/", 1)

        if not word:
            raise ValueError(
                f"Sentence {sentence_number}, token {token_number}: empty word"
            )

        if tag not in VALID_PTB_TAGS:
            raise ValueError(
                f"Sentence {sentence_number}, token {token_number}: "
                f"invalid PTB tag {tag!r} for {word!r}"
            )

        hidden = HIDDEN_CHARACTERS.intersection(word)
        if hidden:
            codepoints = ", ".join(
                f"U+{ord(character):04X}" for character in sorted(hidden)
            )
            raise ValueError(
                f"Sentence {sentence_number}, token {token_number}: "
                f"hidden character {codepoints} in {word!r}"
            )

        tagged_words.append((word, tag))

    if not tagged_words:
        raise ValueError(f"Sentence {sentence_number}: no tokens found")

    return tagged_words


def build_cleaned_raw(
    reviewed_path: Path = REVIEWED_POS_PATH,
    output_path: Path = CLEANED_RAW_PATH,
) -> tuple[list[list[tuple[str, str]]], list[str]]:
    """Generate word-only sentences from the canonical reviewed POS file."""
    reviewed_lines = read_nonempty_lines(reviewed_path)

    if len(reviewed_lines) != EXPECTED_SENTENCES:
        raise ValueError(
            f"Expected {EXPECTED_SENTENCES} reviewed sentences, "
            f"found {len(reviewed_lines)}"
        )

    tagged_sentences = [
        parse_reviewed_sentence(line, sentence_number)
        for sentence_number, line in enumerate(reviewed_lines, start=1)
    ]
    raw_sentences = [
        " ".join(word for word, _ in sentence)
        for sentence in tagged_sentences
    ]

    output_path.write_text(
        "\n".join(raw_sentences) + "\n",
        encoding="utf-8",
    )

    return tagged_sentences, raw_sentences


def validate_alignment(
    tagged_sentences: list[list[tuple[str, str]]],
    raw_path: Path = CLEANED_RAW_PATH,
) -> int:
    """Verify exact sentence and token alignment after writing the raw file."""
    raw_lines = read_nonempty_lines(raw_path)

    if len(raw_lines) != len(tagged_sentences):
        raise ValueError(
            f"Sentence count mismatch: {len(raw_lines)} raw and "
            f"{len(tagged_sentences)} reviewed"
        )

    total_tokens = 0

    for sentence_number, (raw_line, tagged_sentence) in enumerate(
        zip(raw_lines, tagged_sentences), start=1
    ):
        raw_words = raw_line.split()
        reviewed_words = [word for word, _ in tagged_sentence]

        if raw_words != reviewed_words:
            raise ValueError(
                f"Sentence {sentence_number} is not aligned:\n"
                f"raw      = {raw_words}\n"
                f"reviewed = {reviewed_words}"
            )

        total_tokens += len(raw_words)

    return total_tokens


def main() -> None:
    tagged_sentences, _ = build_cleaned_raw()
    total_tokens = validate_alignment(tagged_sentences)

    print(f"Validated sentences : {len(tagged_sentences)}")
    print(f"Validated tokens    : {total_tokens}")
    print(f"Cleaned raw file    : {CLEANED_RAW_PATH.name}")
    print("Alignment status    : PASS")


if __name__ == "__main__":
    main()
