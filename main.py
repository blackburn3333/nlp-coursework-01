"""
Filename: main.py
Author: Jayendra Matarage
Created on: 9/25/2026 4:51 PM
Description: Multi-processed fast evaluation pipeline for PCFG evaluation.
"""
import sys
import concurrent.futures
from pcfg_train import train_pcfg, parse_with_oov_fallback, parse_tag_sequence, evaluate_parse

_pcfg_model = None


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def init_worker(grammar):
    global _pcfg_model
    _pcfg_model = grammar


def process_sentence(args):
    idx, sentence_str = args
    raw_tokens = sentence_str.strip().split()
    if not raw_tokens:
        return None

    words = []
    gold_tags = []

    for token in raw_tokens:
        if '/' in token:
            parts = token.rsplit('/', 1)
            words.append(parts[0] if parts[0] else '/')
            gold_tags.append(parts[1])
        else:
            words.append(token)
            gold_tags.append('NN')

    tree = parse_with_oov_fallback(_pcfg_model, words)

    if tree is None:
        tree = parse_tag_sequence(_pcfg_model, gold_tags, words)

    if tree:
        p, r, f1, acc = evaluate_parse(tree, gold_tags, words)
        return (idx, True, p, r, f1, acc)
    else:
        return (idx, False, 0.0, 0.0, 0.0, 0.0)


def main():
    pos_file = "daily_mirror_data.reviewed.pos"

    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)

        try:
            print("\n=== Training PCFG Model on Penn Treebank ===")
            pcfg_model = train_pcfg()
            print("PCFG model compiled and saved to 'pcfg_grammar.json'.")

            print("\n=== Step 3: Parsing Test Sentences ===")
            try:
                with open(pos_file, "r", encoding="utf-8") as pos_f:
                    sentences = [line for line in pos_f if line.strip()]
            except FileNotFoundError:
                print(f"Error: File '{pos_file}' not found.")
                return

            tasks = [(idx, line) for idx, line in enumerate(sentences, 1)]

            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            total_pos_acc = 0.0
            parsed_count = 0

            with concurrent.futures.ProcessPoolExecutor(
                initializer=init_worker, initargs=(pcfg_model,)
            ) as executor:
                for result in executor.map(process_sentence, tasks):
                    if result is None:
                        continue
                    idx, success, p, r, f1, acc = result
                    if success:
                        parsed_count += 1
                        total_precision += p
                        total_recall += r
                        total_f1 += f1
                        total_pos_acc += acc
                        print(
                            f"Sentence {idx:3d}: Parsed | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f} | Tag Acc: {acc * 100:.1f}%"
                        )
                    else:
                        print(f"Sentence {idx:3d}: Parse Failed")

            total = len(sentences)
            avg_precision = total_precision / parsed_count if parsed_count else 0.0
            avg_recall = total_recall / parsed_count if parsed_count else 0.0
            avg_f1 = total_f1 / parsed_count if parsed_count else 0.0
            avg_pos_acc = (total_pos_acc / parsed_count) * 100 if parsed_count else 0.0

            print("\n================ FINAL EVALUATION REPORT ================")
            print(f"Total Sentences Tested        : {total}")
            print(f"Successfully Parsed Sentences : {parsed_count} ({(parsed_count / total) * 100 if total else 0:.2f}%)")
            print(f"Average Labeled Precision (LP): {avg_precision:.4f}")
            print(f"Average Labeled Recall (LR)   : {avg_recall:.4f}")
            print(f"Average F1-Score              : {avg_f1:.4f}")
            print(f"Average POS Tagging Accuracy  : {avg_pos_acc:.2f}%")
            print("=========================================================")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()