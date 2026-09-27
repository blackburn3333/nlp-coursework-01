"""
Author: Jayendra Matarage
Created on: 9/25/2026 1:23 PM
Description:
"""
from __future__ import annotations

import collections
import copy
import json
import math
import pickle
from pathlib import Path

from nltk.corpus import treebank
from nltk.grammar import Nonterminal, induce_pcfg
from nltk.tag import AffixTagger, BigramTagger, DefaultTagger, UnigramTagger
from nltk.tree import Tree


PROJECT_DIR = Path(__file__).resolve().parent
GRAMMAR_JSON_PATH = PROJECT_DIR / "pcfg_grammar.json"
GRAMMAR_PICKLE_PATH = PROJECT_DIR / "pcfg_grammar.pkl"
TAGGER_PICKLE_PATH = PROJECT_DIR / "pos_tagger.pkl"


def _strip_function_label(label: str) -> str:
    """Reduce phrase labels such as ``NP-SBJ`` to their base category."""
    if label.startswith("-"):
        return label
    return label.split("-", 1)[0].split("=", 1)[0]


def _clean_tree(node: Tree) -> Tree | None:
    """Copy a PTB tree while removing traces and normalizing phrase labels."""
    if node.label() == "-NONE-":
        return None

    children = []
    for child in node:
        if isinstance(child, Tree):
            cleaned_child = _clean_tree(child)
            if cleaned_child is not None and len(cleaned_child) > 0:
                children.append(cleaned_child)
        else:
            children.append(child)

    if not children:
        return None

    # Preserve POS labels on preterminals; normalize phrase labels only.
    label = node.label() if all(not isinstance(c, Tree) for c in children) else _strip_function_label(node.label())
    return Tree(label, children)


def _to_tag_tree(source_tree: Tree) -> Tree | None:
    """Convert lexical leaves to their gold POS symbols for grammar training."""
    cleaned = _clean_tree(copy.deepcopy(source_tree))
    if cleaned is None:
        return None

    for leaf_position in cleaned.treepositions("leaves"):
        preterminal = cleaned[leaf_position[:-1]]
        cleaned[leaf_position] = preterminal.label()

    # A shared root allows every training-tree root type to contribute.
    wrapped = Tree("TOP", [cleaned])
    wrapped.collapse_unary(
        collapsePOS=False,
        collapseRoot=False,
        joinChar="+",
    )
    wrapped.chomsky_normal_form(horzMarkov=2)
    return wrapped


def train_pos_tagger(tagged_sentences):
    """Train a statistical backoff tagger using only Penn Treebank labels."""
    tag_counts = collections.Counter(
        tag for sentence in tagged_sentences for _, tag in sentence
    )
    if not tag_counts:
        raise ValueError("Penn Treebank supplied no tagged training tokens")

    default = DefaultTagger(tag_counts.most_common(1)[0][0])
    affix = AffixTagger(tagged_sentences, backoff=default)
    unigram = UnigramTagger(tagged_sentences, backoff=affix)
    return BigramTagger(tagged_sentences, backoff=unigram)


def train_structural_pcfg(parsed_sentences):
    """Induce a binarized PCFG whose terminals are POS-tag symbols."""
    productions = []

    for source_tree in parsed_sentences:
        tag_tree = _to_tag_tree(source_tree)
        if tag_tree is not None:
            productions.extend(tag_tree.productions())

    if not productions:
        raise ValueError("Penn Treebank supplied no grammar productions")

    return induce_pcfg(Nonterminal("TOP"), productions)


def _save_grammar_json(grammar, output_path: Path = GRAMMAR_JSON_PATH) -> None:
    """Save human-readable PCFG rules and their learned probabilities."""
    serialized_rules = []

    for production in grammar.productions():
        rhs = [
            {
                "type": "nonterminal" if isinstance(symbol, Nonterminal) else "terminal",
                "value": str(symbol),
            }
            for symbol in production.rhs()
        ]
        serialized_rules.append(
            {
                "lhs": str(production.lhs()),
                "rhs": rhs,
                "probability": production.prob(),
            }
        )

    output_path.write_text(
        json.dumps(
            {
                "start_symbol": str(grammar.start()),
                "productions": serialized_rules,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def train_models():
    """Train and save the POS tagger and structural PCFG."""
    # PTB traces are parsing annotations, not observable POS tokens.
    tagged_sentences = [
        [(word, tag) for word, tag in sentence if tag != "-NONE-"]
        for sentence in treebank.tagged_sents()
    ]
    tagged_sentences = [sentence for sentence in tagged_sentences if sentence]
    parsed_sentences = list(treebank.parsed_sents())

    pos_tagger = train_pos_tagger(tagged_sentences)
    grammar = train_structural_pcfg(parsed_sentences)

    with TAGGER_PICKLE_PATH.open("wb") as output:
        pickle.dump(pos_tagger, output)

    with GRAMMAR_PICKLE_PATH.open("wb") as output:
        pickle.dump(grammar, output)

    _save_grammar_json(grammar)
    return pos_tagger, grammar


def predict_pos(pos_tagger, words: list[str]) -> list[str]:
    """Predict one Penn Treebank POS tag for every input word."""
    tagged_words = pos_tagger.tag(words)
    predicted_tags = [tag for _, tag in tagged_words]

    if len(predicted_tags) != len(words) or any(tag is None for tag in predicted_tags):
        raise RuntimeError("POS tagger did not produce one tag per input word")

    return predicted_tags


class ViterbiCkyParser:
    """Exact Viterbi CKY parser for the induced CNF PCFG."""

    def __init__(self, grammar):
        self.start = grammar.start()
        self.lexical_rules = collections.defaultdict(list)
        self.unary_rules = collections.defaultdict(list)
        self.binary_rules = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

        for production in grammar.productions():
            rhs = production.rhs()
            rule = (production.lhs(), math.log(production.prob()))

            if production.is_lexical():
                self.lexical_rules[rhs[0]].append(rule)
            elif len(rhs) == 1 and isinstance(rhs[0], Nonterminal):
                self.unary_rules[rhs[0]].append(rule)
            elif (
                len(rhs) == 2
                and isinstance(rhs[0], Nonterminal)
                and isinstance(rhs[1], Nonterminal)
            ):
                self.binary_rules[rhs[0]][rhs[1]].append(rule)
            else:
                raise ValueError(f"Grammar is not in supported CNF form: {production}")

    def _apply_unary_closure(self, cell):
        queue = collections.deque(cell)

        while queue:
            child_symbol = queue.popleft()
            child_score, child_tree = cell[child_symbol]

            for parent_symbol, rule_log_probability in self.unary_rules.get(
                child_symbol, ()
            ):
                candidate_score = child_score + rule_log_probability
                current = cell.get(parent_symbol)

                if current is None or candidate_score > current[0] + 1e-12:
                    cell[parent_symbol] = (
                        candidate_score,
                        Tree(str(parent_symbol), [child_tree]),
                    )
                    queue.append(parent_symbol)

    def parse(self, tokens: list[str]) -> Tree | None:
        token_count = len(tokens)
        if token_count == 0:
            return None

        chart = [
            [dict() for _ in range(token_count + 1)]
            for _ in range(token_count)
        ]

        for index, token in enumerate(tokens):
            cell = chart[index][index + 1]
            for lhs, rule_log_probability in self.lexical_rules.get(token, ()):
                current = cell.get(lhs)
                if current is None or rule_log_probability > current[0]:
                    cell[lhs] = (
                        rule_log_probability,
                        Tree(str(lhs), [token]),
                    )
            self._apply_unary_closure(cell)

        for span_length in range(2, token_count + 1):
            for start in range(token_count - span_length + 1):
                end = start + span_length
                cell = chart[start][end]

                for split in range(start + 1, end):
                    left_cell = chart[start][split]
                    right_cell = chart[split][end]

                    for left_symbol, (left_score, left_tree) in left_cell.items():
                        right_rule_groups = self.binary_rules.get(left_symbol, {})

                        for right_symbol, rules in right_rule_groups.items():
                            right_result = right_cell.get(right_symbol)
                            if right_result is None:
                                continue

                            right_score, right_tree = right_result
                            for lhs, rule_log_probability in rules:
                                candidate_score = (
                                    left_score
                                    + right_score
                                    + rule_log_probability
                                )
                                current = cell.get(lhs)

                                if current is None or candidate_score > current[0]:
                                    cell[lhs] = (
                                        candidate_score,
                                        Tree(str(lhs), [left_tree, right_tree]),
                                    )

                if cell:
                    self._apply_unary_closure(cell)

        result = chart[0][token_count].get(self.start)
        return result[1] if result is not None else None


_PARSER_CACHE = {}


def parse_predicted_tags(
    grammar,
    predicted_tags: list[str],
    words: list[str],
) -> tuple[Tree | None, str | None]:
    """Parse predicted tags, restore the original words, and report failures."""
    if len(predicted_tags) != len(words):
        return None, "tag/word length mismatch"

    try:
        cache_key = id(grammar)
        parser = _PARSER_CACHE.get(cache_key)
        if parser is None:
            parser = ViterbiCkyParser(grammar)
            _PARSER_CACHE[cache_key] = parser
        parsed_tree = parser.parse(predicted_tags)
    except ValueError as error:
        return None, f"grammar coverage error: {error}"
    except Exception as error:  # Preserve the failure reason in the report.
        return None, f"{type(error).__name__}: {error}"

    if parsed_tree is None:
        return None, "no complete TOP parse"

    parsed_tree.un_chomsky_normal_form(expandUnary=True)

    if parsed_tree.label() == "TOP" and len(parsed_tree) == 1:
        parsed_tree = parsed_tree[0]

    leaf_positions = parsed_tree.treepositions("leaves")
    if len(leaf_positions) != len(words):
        return None, "parsed leaf count does not match input word count"

    for position, word in zip(leaf_positions, words):
        parsed_tree[position] = word

    return parsed_tree, None


# Compatibility wrapper for older imports in the project.
def train_pcfg():
    """Train both models and return the structural grammar."""
    _, grammar = train_models()
    return grammar
