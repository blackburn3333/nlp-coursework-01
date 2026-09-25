"""
Filename: nltk_flow.py
Author: Jayendra Matarage
Created on: 9/25/2026 6:54 PM
Description: 
"""

import nltk
from nltk.corpus import treebank
from nltk.grammar import induce_pcfg, Nonterminal
from nltk.parse import ViterbiParser

# Download the Penn Treebank dataset
nltk.download('treebank')

# Load parsed sentences
parsed_sents = treebank.parsed_sents()

print(f"Total parsed sentences loaded: {len(parsed_sents)}")
print("\nSample Parse Tree (Sentence 1):")
print(parsed_sents[0])




# Collect all production rules from the training trees
productions = []
for tree in parsed_sents:
    # Ensure binary or clean tree structure where necessary
    tree.collapse_unary(collapsePOS=False, collapseRoot=False)
    productions.extend(tree.productions())

# Induce PCFG with maximum likelihood probabilities
pcfg_grammar = induce_pcfg(Nonterminal('S'), productions)

print(f"Total induced PCFG rules: {len(pcfg_grammar.productions())}")
print("\nSample Induced Rules:")
for rule in pcfg_grammar.productions()[:10]:
    print(rule)




# Instantiate the Viterbi probabilistic parser
parser = ViterbiParser(pcfg_grammar)

# Example test sentence (must contain words present in the grammar vocabulary)
sample_tokens = "Pierre".split()

try:
    for tree in parser.parse(sample_tokens):
        print("\nPredicted Most Probable Parse Tree:")
        print(tree)
        print(f"Log Probability: {tree.prob()}")
except ValueError as e:
    print(f"Parsing error (likely Out-of-Vocabulary word): {e}")


