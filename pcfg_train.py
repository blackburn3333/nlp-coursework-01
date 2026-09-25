"""
Filename: pcfg_train.py
Author: Jayendra Matarage
Created on: 9/25/2026 5:06 PM
Description: High-performance PCFG trainer and evaluation helper.
"""
import json
import collections
import nltk
import io
import contextlib
from nltk.corpus import treebank
from nltk.parse import ViterbiParser
from nltk.grammar import PCFG, ProbabilisticProduction, Nonterminal


def train_pcfg():
    """Extracts PCFG grammar rules and probabilities from the Penn Treebank dataset."""
    parsed_sents = treebank.parsed_sents()
    productions = []
    for tree in parsed_sents:
        productions.extend(tree.productions())

    lhs_counts = collections.Counter()
    rule_counts = collections.Counter()

    for prod in productions:
        lhs_counts[prod.lhs()] += 1
        rule_counts[prod] += 1

    pcfg_productions = []
    grammar_dict = {}

    for prod, count in rule_counts.items():
        prob = count / lhs_counts[prod.lhs()]
        pcfg_productions.append(ProbabilisticProduction(prod.lhs(), prod.rhs(), prob=prob))

        lhs_str = str(prod.lhs())
        rhs_str = [str(sym) for sym in prod.rhs()]
        if lhs_str not in grammar_dict:
            grammar_dict[lhs_str] = []
        grammar_dict[lhs_str].append({'rhs': rhs_str, 'prob': prob})

    start_symbol = Nonterminal('S')
    grammar = PCFG(start_symbol, pcfg_productions)

    with open('pcfg_grammar.json', 'w', encoding='utf-8') as f:
        json.dump(grammar_dict, f, indent=2)

    return grammar


def get_oov_tag_guesses(word):
    """Morphological heuristic tag assigner for unknown (OOV) words."""
    if word[0].isupper():
        return ['NNP', 'NN']
    elif word.isdigit() or any(char.isdigit() for char in word):
        return ['CD']
    elif word.endswith('ing'):
        return ['VBG', 'JJ']
    elif word.endswith('ed'):
        return ['VBD', 'VBN']
    elif word.endswith('ly'):
        return ['RB']
    elif word.endswith('s'):
        return ['NNS', 'VBZ']
    elif word.endswith('able') or word.endswith('ible') or word.endswith('al'):
        return ['JJ']
    else:
        return ['NN', 'JJ']


def parse_with_oov_fallback(pcfg_grammar, tokens, max_len=22):
    """Parses a word sequence using constrained Viterbi PCFG."""
    if len(tokens) > max_len:
        return None

    extra_rules = []
    for word in tokens:
        if not pcfg_grammar.productions(rhs=word):
            guessed_tags = get_oov_tag_guesses(word)
            for tag in set(guessed_tags):
                extra_rules.append(
                    ProbabilisticProduction(Nonterminal(tag), [word], prob=1e-5)
                )

    if extra_rules:
        active_grammar = PCFG(pcfg_grammar.start(), list(pcfg_grammar.productions()) + extra_rules)
    else:
        active_grammar = pcfg_grammar

    viterbi = ViterbiParser(active_grammar, max_time=0.4)

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            parses = list(viterbi.parse(tokens))
            return parses[0] if parses else None
    except Exception:
        return None


def parse_tag_sequence(pcfg_grammar, tags, words):
    """Fallback parser operating on POS tags while binding actual words to terminal leaves."""
    nonterminal_productions = [p for p in pcfg_grammar.productions() if not p.is_lexical()]

    unique_tags = set(tags)
    tag_lexical_rules = [
        ProbabilisticProduction(Nonterminal(tag), [tag], prob=1.0)
        for tag in unique_tags
    ]

    try:
        tag_grammar = PCFG(pcfg_grammar.start(), nonterminal_productions + tag_lexical_rules)
        viterbi_tag_parser = ViterbiParser(tag_grammar, max_time=0.2)

        tag_tokens = [str(tag) for tag in tags]
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            parses = list(viterbi_tag_parser.parse(tag_tokens))
            if parses:
                tag_tree = parses[0]
                for i, leaf_pos in enumerate(tag_tree.treepositions('leaves')):
                    tag_tree[leaf_pos] = words[i]
                return tag_tree
    except Exception:
        pass

    # Safety Fallback: Construct word-bound flat parse tree
    children = [nltk.Tree(tag, [word]) for tag, word in zip(tags, words)]
    return nltk.Tree('S', children)


def compute_parseval_constituents(tree):
    """Extracts labeled constituents: (Label, Start_Index, End_Index)."""
    constituents = []

    def get_leaf_count(node):
        if isinstance(node, nltk.Tree):
            return len(node.leaves())
        return 1

    def traverse(node, start_idx):
        if isinstance(node, nltk.Tree):
            length = len(node.leaves())
            end_idx = start_idx + length
            if length > 1:
                constituents.append((node.label(), start_idx, end_idx))
            curr = start_idx
            for child in node:
                traverse(child, curr)
                curr += get_leaf_count(child)

    traverse(tree, 0)
    return constituents


def evaluate_parse(pred_tree, gold_tree, gold_tags):
    """Computes genuine PARSEVAL Precision, Recall, and F1 against a gold reference tree."""
    if pred_tree is None or gold_tree is None:
        return 0.0, 0.0, 0.0, 0.0

    # 1. POS Tag Accuracy
    pred_pos_tags = [pos for _, pos in pred_tree.pos()]
    correct_pos = sum(1 for p, g in zip(pred_pos_tags, gold_tags) if p == g)
    pos_accuracy = correct_pos / len(gold_tags) if gold_tags else 0.0

    # 2. PARSEVAL Metrics against reference tree
    pred_consts = set(compute_parseval_constituents(pred_tree))
    gold_consts = set(compute_parseval_constituents(gold_tree))

    if not pred_consts:
        return 0.0, 0.0, 0.0, pos_accuracy

    matching = len(pred_consts.intersection(gold_consts))

    precision = matching / len(pred_consts) if pred_consts else 0.0
    recall = matching / len(gold_consts) if gold_consts else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, pos_accuracy