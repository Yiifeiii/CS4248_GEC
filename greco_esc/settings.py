"""Every setting of the GRECO + ESC pipeline, in one place.

The shell scripts read the runs from here, so a setting is changed only in this file:

    python3 -m greco_esc.settings runs        # one line per GRECO run: name, k, vote_coef, esc, vote_mode
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

# Voting coefficient per vote mode and k: the best alpha on dev_tune, tuned without ESC on
# plain beam search lists decoded as here (main, outputs/greco/tuning/dev_tune.errant.beam/
# best.json). Not the full-dev tuning (dev.errant.beam), which includes dev_test.
# "hypothesis" is what official run_combination.py counts; "system" and "rank" are counted by
# combine.py (see its docstring).
VOTES = {
    'hypothesis': {1: 0.4, 2: 0.1, 3: 0.1, 5: 0.1},
    'system': {1: 0.4, 2: 0.1, 3: 0.1, 5: 0.1},
    'rank': {1: 0.4, 2: 0.4, 3: 0.4, 5: 0.4},
}
VOTE = VOTES['hypothesis']
VOTE_MODES = tuple(VOTES)

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
    """(name, k, vote_coef, esc, vote_mode) for every GRECO run.

    For each k, with and without ESC: no voting, "hypothesis" voting (names greco[+esc].vote<a>,
    as before vote modes were added, so finished runs are reused), and for k > 1 "system" and
    "rank" voting (greco[+esc].<mode><a>). At k = 1 the three modes give the same output.
    """
    out = []
    for k in KS:
        for esc in (False, True):
            prefix = 'greco+esc' if esc else 'greco'
            for vote in (VOTE[k], 0):
                out.append((f'{prefix}.vote{vote:g}', k, vote, esc, 'hypothesis'))
            if k > 1:
                for mode in ('system', 'rank'):
                    vote = VOTES[mode][k]
                    out.append((f'{prefix}.{mode}{vote:g}', k, vote, esc, mode))
    return out


def pred_path(k, name):
    return ROOT / TEST_SPLIT / f'k{k}' / name / 'pred.txt'


def main(argv):
    if argv[:1] == ['runs']:
        for name, k, vote, esc, mode in runs():
            print(name, k, f'{vote:g}', int(esc), mode)
    elif argv[:1] == ['inputs'] and len(argv) == 2:
        print(' '.join(inputs(int(argv[1]))))
    else:
        sys.exit('usage: python3 -m greco_esc.settings runs | inputs <k>')


if __name__ == '__main__':
    main(sys.argv[1:])
