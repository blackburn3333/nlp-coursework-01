# Manual POS Review

## Review status

- Source: `daily_mirror_data.pos` (preserved unchanged)
- Working copy: `daily_mirror_data.manual_review.pos`
- Tagset: Penn Treebank, matching the NLTK output already present
- Dataset size: 150 lines and 3,695 original whitespace-delimited tokens
- Manually reviewed: lines 1–25 (769 tokens in the reviewed representation)
- Pending: lines 26–150
- Changed in this batch: 18 lines

`daily_mirror_data.manual_review.pos` contains the whole dataset so source and
review line numbers remain aligned. Only the completed range (lines 1–25) is
approved; tags after line 25 must still be treated as unreviewed NLTK output.

## Review decisions for lines 1–25

| Line | Original | Reviewed | Reason |
|---:|---|---|---|
| 1 | `Former/NNP` | `Former/JJ` | Attributive adjective, not a proper noun. |
| 2 | `’/NNP s/NN` | `'s/POS` | Rejoined and tagged the possessive clitic. |
| 3 | `carried/VBD` | `carried/VBN` | Past participle in a reduced passive clause. |
| 3 | `suicide/JJ` | `suicide/NN` | Noun used attributively in `suicide bombers`. |
| 3 | `Islamist/NNP` | `Islamist/JJ` | Adjective modifying `extremist group`. |
| 3 | `linked/VBD` | `linked/VBN` | Past participle in a reduced passive clause. |
| 4 | `’/NNP s/VBD` | `'s/POS` | Rejoined and tagged the possessive clitic. |
| 5 | `’/NNP s/VBD` | `'s/POS` | Rejoined and tagged the possessive clitic. |
| 5 | malformed quote tags around `B-` and `CCC+` | Penn opening/closing quote tags | Restored quoted rating labels; labels are common nouns (`NN`). |
| 7 | `New/NNP` | `New/JJ` | Sentence-initial adjective modifying `regulations`. |
| 8 | `upper/NN`, `lower/JJR` | `upper/JJ`, `lower/JJ` | Coordinated adjectives modifying `range`; no comparison is expressed. |
| 12 | `de/IN` | `de/NNP` | Part of the personal name `Aravinda de Silva`. |
| 13 | `Former/NNP` | `Former/JJ` | Attributive adjective before a title. |
| 14 | `Certain/NN` | `Certain/JJ` | Adjective modifying `Individuals`. |
| 15 | `italian/JJ` | `Italian/JJ` | Corrected demonym capitalization; tag remains adjectival. |
| 15 | `down/RB` | `down/RP` | Particle in the phrasal verb `fell down`. |
| 16 | zero-width character before `oppose` | `oppose/VBP` | Removed an invisible token contaminant; retained the valid tag. |
| 16 | zero-width character + `President/NN` | `President/NNP` | Removed contaminant and tagged a title before a personal name. |
| 17 | `Civil/NNP` | `Civil/JJ` | Adjective in `civil society organisation`. |
| 18 | `MPs/NNP` | `MPs/NNPS` | Plural proper-name/acronym form. |
| 18 | `dressed/VBD` | `dressed/VBN` | Past participle describing the MPs' state. |
| 19 | `’/NNP s/VBD` | `'s/POS` | Rejoined and tagged the possessive clitic. |
| 22 | `AI-powered/NNP` | `AI-powered/JJ` | Compound adjective modifying `dashboard`. |
| 23 | `Cursed/JJ` | `Cursed/NNP` | Component of the work title. |
| 23 | `living/VBG` | `living/NN` | Noun in the fixed phrase `cost of living`. |
| 23 | duplicated malformed apostrophe tokens | Penn opening/closing quote tags | Normalized the quotation around `literally impossible`. |
| 23 | `’/NNP s/VBD` | `'s/POS` | Rejoined and tagged the possessive clitic. |
| 24 | `Justice/NNP` | `Justice/NN` | Sentence-initial common noun. |
| 24 | `remain/NN` | `remain/VBP` | Present-tense verb agreeing with the coordinated subject. |
| 24 | `’/NNP s/VBD` | `'s/POS` | Rejoined and tagged the possessive clitic. |

Multiple quote-token and tag repairs on line 5 and line 23 are summarized in
single rows above.

## Dataset-wide issues queued for later batches

- The original has 24 right-curly-apostrophe characters, mostly split
  possessives such as `’/NNP s/VBD`.
- It has three opening and four closing smart-quote characters with unreliable
  tags.
- Two zero-width Unicode characters occur on line 16; both were removed in this
  batch.
- Twenty-one source lines contain smart punctuation or zero-width characters.
- Line 36 appears to be a scraped website navigation/header line rather than
  news prose. It is retained pending a corpus-inclusion decision.
- Later text includes additional likely errors involving possessives, quotes,
  sentence-initial capitalization, participles, titles, acronyms, and list
  headings. These have not yet been approved or changed.

## Review convention

Each next batch should be reviewed in context, not by replacing every instance
of a word/tag pair globally. Preserve one sentence per line, use the Penn
Treebank tagset, normalize split possessive clitics to `'s/POS`, and use the
Penn opening-quote and closing-quote tags for quotations. Record corpus-content
decisions (for example, removing navigation text) separately from POS changes.
