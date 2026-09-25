"""
Filename: fix_manual_review.py
Author: Jayendra Matarage
Created on: 9/24/2026 7:21 AM
Description: 
"""
import pandas as pd

# Load review fixes
fixes = pd.read_csv("daily_mirror_pos_manual_review.tsv", sep="\t")

# Read current daily_mirror_data.pos file
with open("daily_mirror_data.pos", "r", encoding="utf-8") as f:
    lines = [line.strip().split() for line in f]

# Apply modifications line by line
for idx, row in fixes.iterrows():
    line_no = int(row["line"]) - 1  # 0-based index
    tok_idx = int(row["token_index"]) - 1

    if row["status"] == "CONFIRMED":
        lines[line_no][tok_idx] = row["proposed"]
    elif row["status"] == "STRUCTURAL" and row["proposed"] == "[delete]":
        lines[line_no].pop(tok_idx)
    elif row["status"] == "TEXT_ERROR":
        lines[line_no][tok_idx] = row["proposed"]

# Reassemble lines and save output
with open("daily_mirror_data.pos", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(" ".join(line) + "\n")

print("Successfully updated 'daily_mirror_data.pos' with manual review changes.")