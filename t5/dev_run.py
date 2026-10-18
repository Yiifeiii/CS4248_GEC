from pathlib import Path
from itertools import islice

from retokenize import retokenize
from t5_gec import top_k

# Repo root (t5/dev_run.py -> CS4248_GEC/)
ROOT = Path(__file__).resolve().parents[1]
DEV_SRC = ROOT / "data" / "dev.src"
OUT_PATH = ROOT / "outputs" / "t5" / "dev30.tgt"
N = 30


# Read the first n sentences (one per line) from a source file
def load_sentences(path=DEV_SRC, n=N):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in islice(f, n)]


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for i, sentence in enumerate(load_sentences(), 1):
            candidates = top_k(sentence)
            print(f"[{i}] {sentence}")
            for text, score in candidates:
                print(f"    {score:.3f}  {text}")
            print()
            # Write all k candidates (best first), one per line: lines 5(i-1)+1 .. 5i belong to sentence i.
            # Split them into tokens like the source ("It's" -> "It 's") so GRECO sees only real edits.
            for text, _ in candidates:
                out.write(retokenize(text) + "\n")

    print(f"Wrote top-5 corrections for {N} sentences to {OUT_PATH}")
