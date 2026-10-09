from pathlib import Path
from itertools import islice

from t5_gec import top_k

# Repo root (dev/t5/dev_run.py -> CS4248_GEC/)
ROOT = Path(__file__).resolve().parents[2]
DEV_SRC = ROOT / "data" / "dev.src"
OUT_DIR = ROOT / "output" / "t5" / "dev30"
N = 30


# Read the first n sentences (one per line) from a source file
def load_sentences(path=DEV_SRC, n=N):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in islice(f, n)]


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "t5.tgt"

    with open(out_path, "w", encoding="utf-8") as out:
        for i, sentence in enumerate(load_sentences(), 1):
            candidates = top_k(sentence)
            print(f"[{i}] {sentence}")
            for text, score in candidates:
                print(f"    {score:.3f}  {text}")
            print()
            # Write all k candidates (best first), one per line: lines 5(i-1)+1 .. 5i belong to sentence i
            for text, _ in candidates:
                out.write(text + "\n")

    print(f"Wrote top-5 corrections for {N} sentences to {out_path}")
