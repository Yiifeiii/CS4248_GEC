"""Every setting of the GRECO + ESC pipeline, in one place.

The shell scripts read the runs from here, so a setting is changed only in this file:

    python3 -m greco_esc.settings runs        # one line per GRECO run: name, k, vote_coef, esc
    python3 -m greco_esc.settings inputs 3    # the input names for k = 3, in combination order
"""

import sys
from pathlib import Path

ROOT = Path('greco_esc/runs')
DATA = Path('data/curriculum')
TRAIN_SPLIT = 'dev_tune'  # ESC learns its edit weights here (the paper trains ESC on BEA-2019 dev)
TEST_SPLIT = 'dev_test'   # every combination is scored here

# Candidate systems: our fine-tuned BART and T5 (stage 1 then stage 2), and GECToR
NBEST = {'bart': 'CS4248-GEC/gec-bart-base-s1s2', 't5': 'CS4248-GEC/gec-t5-base-s1s2'}
MAX_RANK = 5  # BART and T5 hypotheses kept per source; GECToR gives one

KS = (1, 2, 3, 5)

# Voting coefficient per k: the best "hypothesis" alpha on dev_tune from the greco-voting
# branch (outputs/greco/tuning/dev_tune.errant/best.json). "hypothesis" is the vote that
# official run_combination.py computes; its "system" and "rank" modes are not in upstream.
VOTE = {1: 0.4, 2: 0.1, 3: 0.1, 5: 0.1}

SCORE_RATIO = 0.7  # weight of the ESC edit scores, as in the paper's GRECO + ESC command
BEAM_SIZE = 16     # paper and upstream README
BATCH_SIZE = 16    # paper and upstream README; scores do not depend on it with the attention-mask fix


def inputs(k):
    """Input names for k, in the mini experiment's order: gector-1, then bart-r and t5-r for r <= k."""
    names = ['gector-1']
    for r in range(1, k + 1):
        names += [f'bart-{r}', f't5-{r}']
    return names


def runs():
    """(name, k, vote_coef, esc) for every GRECO run: with and without ESC, with and without voting."""
    out = []
    for k in KS:
        for esc in (False, True):
            for vote in (VOTE[k], 0):
                name = f"{'greco+esc' if esc else 'greco'}.vote{vote:g}"
                out.append((name, k, vote, esc))
    return out


def pred_path(k, name):
    return ROOT / TEST_SPLIT / f'k{k}' / name / 'pred.txt'


def main(argv):
    if argv[:1] == ['runs']:
        for name, k, vote, esc in runs():
            print(name, k, f'{vote:g}', int(esc))
    elif argv[:1] == ['inputs'] and len(argv) == 2:
        print(' '.join(inputs(int(argv[1]))))
    else:
        sys.exit('usage: python3 -m greco_esc.settings runs | inputs <k>')


if __name__ == '__main__':
    main(sys.argv[1:])
