"""Turn the dev_tune candidate pool into the M2 inputs that GRECO's combiner reads.

Reads greco_tuning/runs/pool/bart.tgt and t5.tgt (k = 5 lines per source, best first) and
gector.tgt (one line per source). Writes one ERRANT M2 file per ranked system output:

  greco_tuning/runs/pool/m2/gector-1.m2
  greco_tuning/runs/pool/m2/bart-<r>.m2 and t5-<r>.m2, for r = 1 to 5

This is the input order of the mini experiment: for k, the combiner reads gector-1, then
bart-r and t5-r for r = 1 to k. Run in the environment with ERRANT, after dev_pool.sh:

    python3 -m greco_tuning.make_m2
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from greco_setup.prepare import read_lines, read_system

POOL = Path('greco_tuning/runs/pool')
SRC = Path('data/curriculum/dev_tune.src')
K = 5


def ranked(path, rank, n_sources):
    """The rank-th hypothesis (0 = best) of every source, from a k-lines-per-source file."""
    _, per_source = read_system(f'system={path}', n_sources)
    return [hyps[rank] for hyps in per_source]


def to_m2(src, hyps, out):
    """Write the ERRANT M2 edits from src to hyps, one hypothesis per source line."""
    tools = Path(sys.executable).parent
    with tempfile.TemporaryDirectory() as tmp:
        cor = Path(tmp, 'hyp.txt')
        cor.write_text('\n'.join(hyps) + '\n', encoding='utf-8')
        subprocess.run([tools / 'errant_parallel', '-orig', str(src), '-cor', str(cor), '-out', str(out)],
                       check=True)


def main():
    n = len(read_lines(SRC))
    out_dir = POOL / 'm2'
    out_dir.mkdir(parents=True, exist_ok=True)
    to_m2(SRC, ranked(POOL / 'gector.tgt', 0, n), out_dir / 'gector-1.m2')
    for r in range(1, K + 1):
        to_m2(SRC, ranked(POOL / 'bart.tgt', r - 1, n), out_dir / f'bart-{r}.m2')
        to_m2(SRC, ranked(POOL / 't5.tgt', r - 1, n), out_dir / f't5-{r}.m2')
    print(f'wrote {1 + 2 * K} M2 files for {n} sentences to {out_dir}')


if __name__ == '__main__':
    main()
