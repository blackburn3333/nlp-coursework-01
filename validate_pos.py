"""
Author: Jayendra Matarage
Created on: 9/25/2026 4:51 PM
Description: 
"""


def validate_gold_standard(pos_file_path):
    """Validates that the input .pos file adheres to Penn Treebank tags and format."""
    valid_tags = {
        'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
        'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
        'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
        'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.', ',', ':', '``', "''", '$', '#'
    }
    with open(pos_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    errors = []
    total_tokens = 0
    for idx, line in enumerate(lines, 1):
        tokens = line.strip().split()
        for token in tokens:
            total_tokens += 1
            if '/' not in token:
                errors.append(f"Line {idx}: Missing slash in '{token}'")
                continue
            word, tag = token.rsplit('/', 1)
            if tag not in valid_tags:
                errors.append(f"Line {idx}: Invalid tag '{tag}' for word '{word}'")

    print(f"Validation finished. Sentences: {len(lines)}, Total Tokens: {total_tokens}")
    if errors:
        print(f"Found {len(errors)} formatting/tag errors:")
        for err in errors[:10]:
            print(f" - {err}")
    else:
        print("Gold-standard dataset is 100% compliant with Penn Treebank schema.")


validate_gold_standard("daily_mirror_data.reviewed.pos")