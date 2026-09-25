"""
Filename: validate_pos.py
Author: Jayendra Matarage
Created on: 9/25/2026 4:51 PM
Description:
"""

from pcfg_train import train_pcfg, parse_with_oov_fallback, parse_tag_sequence, evaluate_parse


def main():
    pos_file = "daily_mirror_data.reviewed.pos"
    print("\n=== Training PCFG Model on Penn Treebank ===")
    print("\n=== Step 2: Extracting Grammar & Building PCFG Model ===")
    pcfg_model = train_pcfg()
    print("PCFG model compiled and saved to 'pcfg_grammar.json'.")

    print("\n=== Step 3: Parsing & Evaluating Test Sentences ===")
    try:
        with open(pos_file, "r", encoding="utf-8") as f:
            sentences = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{pos_file}' not found.")
        return

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_pos_acc = 0.0
    parsed_count = 0

    for idx, sentence_str in enumerate(sentences, 1):
        print(f"Processing {sentence_str}")
        raw_tokens = sentence_str.strip().split()
        if not raw_tokens:
            continue

        words = []
        gold_tags = []

        # Robust token/tag extraction handling trailing slash edge cases
        for token in raw_tokens:
            if '/' in token:
                parts = token.rsplit('/', 1)
                words.append(parts[0] if parts[0] else '/')
                gold_tags.append(parts[1])
            else:
                words.append(token)
                gold_tags.append('NN')  # Fallback default tag if unannotated

        # Primary Parse Attempt using Words + OOV Heuristics
        tree = parse_with_oov_fallback(pcfg_model, words)

        # Secondary Fallback Attempt using Gold POS Tag Sequence
        if tree is None:
            tree = parse_tag_sequence(pcfg_model, gold_tags)

        if tree:
            parsed_count += 1
            p, r, f1, acc = evaluate_parse(tree, gold_tags, words)
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


if __name__ == '__main__':
    main()