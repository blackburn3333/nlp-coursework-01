"""
Filename: reporting.py
Author: Jayendra Matarage
Created on: 9/25/2026 7:03 PM
Description: 
"""
import time
from nltk.parse import ViterbiParser

# 1. Initialize Viterbi PCFG Parser
print("Initializing Viterbi PCFG Parser...")
parser = ViterbiParser(pcfg_grammar)

parsed_trees = []
successful_parses = 0
total_sentences = len(prepared_sentences_for_parser)
parse_times = []

print(f"\nParsing {total_sentences} Daily Mirror sentences...")

for idx, tokens in enumerate(prepared_sentences_for_parser, 1):
    start_time = time.time()
    try:
        # Generate the top parse tree using Viterbi
        parses = list(parser.parse(tokens))
        elapsed = time.time() - start_time
        parse_times.append(elapsed)

        if parses:
            top_tree = parses[0]
            parsed_trees.append(top_tree)
            successful_parses += 1
            print(f"[{idx}/{total_sentences}] Parsed successfully in {elapsed:.2f}s (Log Prob: {top_tree.prob():.4f})")
        else:
            parsed_trees.append(None)
            print(f"[{idx}/{total_sentences}] Failed to parse (No valid parse tree found)")

    except Exception as e:
        elapsed = time.time() - start_time
        parse_times.append(elapsed)
        parsed_trees.append(None)
        print(f"[{idx}/{total_sentences}] Parser Error: {e}")

# 2. Calculate Standard Performance Metrics
coverage = (successful_parses / total_sentences) * 100
avg_time = sum(parse_times) / len(parse_times) if parse_times else 0

print("\n" + "=" * 40)
print("       EVALUATION SUMMARY RESULTS       ")
print("=" * 40)
print(f"Total Test Sentences : {total_sentences}")
print(f"Successful Parses    : {successful_parses}")
print(f"Failed Parses        : {total_sentences - successful_parses}")
print(f"Parse Coverage Rate  : {coverage:.2f}%")
print(f"Avg Time per Sentence: {avg_time:.3f} seconds")
print("=" * 40)

# 3. Save Generated Parse Trees (Submission Deliverable)
with open("150_parsed_trees.txt", "w", encoding="utf-8") as f:
    for i, tree in enumerate(parsed_trees, 1):
        f.write(f"--- Sentence {i} ---\n")
        if tree:
            f.write(str(tree) + "\n\n")
        else:
            f.write("PARSE FAILED\n\n")

print("Saved predicted parse trees to '150_parsed_trees.txt'")

# 1. Initialize Viterbi PCFG Parser
print("Initializing Viterbi PCFG Parser...")
parser = ViterbiParser(pcfg_grammar)

parsed_trees = []
successful_parses = 0
total_sentences = len(prepared_sentences_for_parser)
parse_times = []

print(f"\nParsing {total_sentences} Daily Mirror sentences...")

for idx, tokens in enumerate(prepared_sentences_for_parser, 1):
    start_time = time.time()
    try:
        # Generate the top parse tree using Viterbi
        parses = list(parser.parse(tokens))
        elapsed = time.time() - start_time
        parse_times.append(elapsed)

        if parses:
            top_tree = parses[0]
            parsed_trees.append(top_tree)
            successful_parses += 1
            print(f"[{idx}/{total_sentences}] Parsed successfully in {elapsed:.2f}s (Log Prob: {top_tree.prob():.4f})")
        else:
            parsed_trees.append(None)
            print(f"[{idx}/{total_sentences}] Failed to parse (No valid parse tree found)")

    except Exception as e:
        elapsed = time.time() - start_time
        parse_times.append(elapsed)
        parsed_trees.append(None)
        print(f"[{idx}/{total_sentences}] Parser Error: {e}")

# 2. Calculate Standard Performance Metrics
coverage = (successful_parses / total_sentences) * 100
avg_time = sum(parse_times) / len(parse_times) if parse_times else 0

print("\n" + "=" * 40)
print("       EVALUATION SUMMARY RESULTS       ")
print("=" * 40)
print(f"Total Test Sentences : {total_sentences}")
print(f"Successful Parses    : {successful_parses}")
print(f"Failed Parses        : {total_sentences - successful_parses}")
print(f"Parse Coverage Rate  : {coverage:.2f}%")
print(f"Avg Time per Sentence: {avg_time:.3f} seconds")
print("=" * 40)

# 3. Save Generated Parse Trees (Submission Deliverable)
with open("150_parsed_trees.txt", "w", encoding="utf-8") as f:
    for i, tree in enumerate(parsed_trees, 1):
        f.write(f"--- Sentence {i} ---\n")
        if tree:
            f.write(str(tree) + "\n\n")
        else:
            f.write("PARSE FAILED\n\n")

print("Saved predicted parse trees to '150_parsed_trees.txt'")