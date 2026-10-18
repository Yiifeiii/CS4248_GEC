"""Split T5 output into tokens the way the dataset is tokenized.

T5 writes ordinary text ("It's ... minds."), but the source files and the other
systems use spaCy tokens ("It 's ... minds ."). GRECO compares sentences token
by token, so every such mismatch would become a fake edit.

    python t5/retokenize.py outputs/t5/dev30.tgt   # rewrite a file in place
"""
import sys

import spacy

# A blank English pipeline has the same tokenizer rules as en_core_web_sm
_tokenizer = spacy.blank("en").tokenizer


def retokenize(text):
    return " ".join(token.text for token in _tokenizer(text))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            lines = [retokenize(line.rstrip("\n")) for line in f]
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
        print(f"retokenized {len(lines)} lines in {path}")
