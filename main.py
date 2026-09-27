"""
Author: Jayendra Matarage
Created on: 9/25/2026 1:23 PM
Description:
"""
from __future__ import annotations

import collections
import concurrent.futures
import sys
import time
from pathlib import Path

from pcfg_train import parse_predicted_tags, predict_pos, train_models
from prepare_dataset import (
    CLEANED_RAW_PATH,
    EXPECTED_SENTENCES,
    REVIEWED_POS_PATH,
    parse_reviewed_sentence,
    read_nonempty_lines,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_DIR / "evaluation_report.txt"

_pos_tagger = None
_pcfg_grammar = None


class Tee:
    """Write report output to both the console and a file."""

    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


def init_worker(pos_tagger, grammar):
    """Install trained models once in each parsing worker."""
    global _pos_tagger, _pcfg_grammar
    _pos_tagger = pos_tagger
    _pcfg_grammar = grammar


def load_evaluation_data() -> list[tuple[int, list[str], list[str]]]:
    """Load aligned words and manually reviewed gold POS tags."""
    raw_lines = read_nonempty_lines(CLEANED_RAW_PATH)
    reviewed_lines = read_nonempty_lines(REVIEWED_POS_PATH)

    if len(raw_lines) != EXPECTED_SENTENCES:
        raise ValueError(
            f"Expected {EXPECTED_SENTENCES} raw sentences, found {len(raw_lines)}"
        )
    if len(reviewed_lines) != EXPECTED_SENTENCES:
        raise ValueError(
            f"Expected {EXPECTED_SENTENCES} reviewed sentences, "
            f"found {len(reviewed_lines)}"
        )

    evaluation_data = []

    for sentence_number, (raw_line, reviewed_line) in enumerate(
        zip(raw_lines, reviewed_lines), start=1
    ):
        words = raw_line.split()
        reviewed = parse_reviewed_sentence(reviewed_line, sentence_number)
        reviewed_words = [word for word, _ in reviewed]
        gold_tags = [tag for _, tag in reviewed]

        if words != reviewed_words:
            raise ValueError(
                f"Sentence {sentence_number}: cleaned raw and reviewed POS "
                "tokens do not match"
            )

        evaluation_data.append((sentence_number, words, gold_tags))

    return evaluation_data


def process_sentence(task):
    """Predict tags, attempt a parse, and return sentence-level results."""
    sentence_number, words, gold_tags = task

    started = time.perf_counter()
    predicted_tags = predict_pos(_pos_tagger, words)
    predicted_tree, failure_reason = parse_predicted_tags(
        _pcfg_grammar,
        predicted_tags,
        words,
    )
    elapsed = time.perf_counter() - started

    correct_tags = sum(
        predicted == gold
        for predicted, gold in zip(predicted_tags, gold_tags)
    )

    return {
        "sentence_number": sentence_number,
        "parsed": predicted_tree is not None,
        "failure_reason": failure_reason,
        "correct_tags": correct_tags,
        "token_count": len(gold_tags),
        "predicted_tags": predicted_tags,
        "gold_tags": gold_tags,
        "elapsed": elapsed,
    }


def calculate_pos_metrics(results):
    """Calculate micro accuracy and per-tag precision, recall, and F1."""
    confusion = collections.Counter()
    total_tokens = 0
    correct_tokens = 0

    for result in results:
        total_tokens += result["token_count"]
        correct_tokens += result["correct_tags"]
        confusion.update(zip(result["gold_tags"], result["predicted_tags"]))

    tags = sorted(
        {gold for gold, _ in confusion} | {predicted for _, predicted in confusion}
    )
    per_tag = []

    for tag in tags:
        true_positive = confusion[tag, tag]
        false_positive = sum(
            count
            for (gold, predicted), count in confusion.items()
            if predicted == tag and gold != tag
        )
        false_negative = sum(
            count
            for (gold, predicted), count in confusion.items()
            if gold == tag and predicted != tag
        )
        support = sum(
            count for (gold, _), count in confusion.items() if gold == tag
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_tag.append((tag, precision, recall, f1, support))

    accuracy = correct_tokens / total_tokens if total_tokens else 0.0
    macro_precision = (
        sum(row[1] for row in per_tag) / len(per_tag) if per_tag else 0.0
    )
    macro_recall = (
        sum(row[2] for row in per_tag) / len(per_tag) if per_tag else 0.0
    )
    macro_f1 = sum(row[3] for row in per_tag) / len(per_tag) if per_tag else 0.0

    return {
        "total_tokens": total_tokens,
        "correct_tokens": correct_tokens,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_tag": per_tag,
    }


def main():
    with REPORT_PATH.open("w", encoding="utf-8") as report_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, report_file)

        try:
            print("=== Penn Treebank Model Training ===")
            print("Training statistical POS tagger and POS-conditioned PCFG...")
            pos_tagger, grammar = train_models()
            print("Saved pos_tagger.pkl, pcfg_grammar.pkl, and pcfg_grammar.json.")

            tasks = load_evaluation_data()
            print(f"\n=== Evaluating {len(tasks)} Daily Mirror Sentences ===")

            results = []
            evaluation_started = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(
                initializer=init_worker,
                initargs=(pos_tagger, grammar),
            ) as executor:
                for result in executor.map(process_sentence, tasks):
                    results.append(result)
                    pos_accuracy = (
                        result["correct_tags"] / result["token_count"] * 100
                        if result["token_count"]
                        else 0.0
                    )

                    if result["parsed"]:
                        status = "Parsed"
                    else:
                        status = f"Parse Failed ({result['failure_reason']})"

                    print(
                        f"Sentence {result['sentence_number']:3d}: {status} | "
                        f"POS Acc: {pos_accuracy:5.1f}% | "
                        f"Time: {result['elapsed']:.3f}s"
                    )
            evaluation_wall_time = time.perf_counter() - evaluation_started

            pos_metrics = calculate_pos_metrics(results)
            parsed_count = sum(result["parsed"] for result in results)
            total_time = sum(result["elapsed"] for result in results)
            failure_counts = collections.Counter(
                result["failure_reason"]
                for result in results
                if not result["parsed"]
            )

            print("\n================ FINAL EVALUATION REPORT ================")
            print(f"Total Sentences Tested       : {len(results)}")
            print(
                f"Successfully Parsed Sentences: {parsed_count} "
                f"({parsed_count / len(results) * 100:.2f}%)"
            )
            print(f"Total Tokens Tested          : {pos_metrics['total_tokens']}")
            print(f"Correctly Tagged Tokens      : {pos_metrics['correct_tokens']}")
            print(f"POS Tagging Accuracy         : {pos_metrics['accuracy'] * 100:.2f}%")
            print(f"Macro POS Precision          : {pos_metrics['macro_precision']:.4f}")
            print(f"Macro POS Recall             : {pos_metrics['macro_recall']:.4f}")
            print(f"Macro POS F1                 : {pos_metrics['macro_f1']:.4f}")
            print(f"Cumulative Sentence Time     : {total_time:.3f}s")
            print(
                f"Average Sentence Time        : "
                f"{total_time / len(results):.3f}s"
            )
            print(f"Evaluation Wall Time         : {evaluation_wall_time:.3f}s")
            print(
                "Constituency LP/LR/F1         : N/A "
                "(no phrase-structure gold trees)"
            )

            if failure_counts:
                print("\nParse Failure Reasons:")
                for reason, count in failure_counts.most_common():
                    print(f"  {count:3d}  {reason}")

            print("\nPer-Tag POS Metrics:")
            print("Tag       Precision  Recall      F1   Support")
            for tag, precision, recall, f1, support in pos_metrics["per_tag"]:
                print(
                    f"{tag:<9} {precision:9.4f}  {recall:6.4f}  "
                    f"{f1:6.4f}  {support:7d}"
                )

            print("=========================================================")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
