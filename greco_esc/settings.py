"""Every setting of the GRECO + ESC pipeline, in one place.

The shell scripts read the runs from here, so a setting is changed only in this file:

    python3 -m greco_esc.settings runs        # one line per GRECO run: name, k, vote_coef, esc, vote_mode
    python3 -m greco_esc.settings inputs 3    # the input names for k = 3, in combination order
    python3 -m greco_esc.settings splits      # each test split and the split ESC trains on for it
    python3 -m greco_esc.settings esc_home 3 dev_tune   # ESC's folder for that k and test split
    python3 -m greco_esc.settings test_runs   # the GRECO runs test.sh makes on each test set
    python3 -m greco_esc.settings source conll14        # a split's or test set's source file
"""

import sys
from pathlib import Path

ROOT = Path('greco_esc/runs')
DATA = Path('data/curriculum')
TRAIN_SPLIT = 'dev_tune'  # ESC learns its edit weights here (the paper trains ESC on BEA-2019 dev)
TEST_SPLIT = 'dev_test'   # every combination is scored here

# Full dev set (data/dev.src) = dev_tune + dev_test; data/curriculum/<half>.ids gives each line's
# position in it. For full-dev scores every combination is also run on dev_tune, with ESC
# trained on dev_test (cross-fitting), so no sentence is scored by an ESC that saw its gold edits.
FULL_SPLIT = 'dev'
HALVES = (TRAIN_SPLIT, TEST_SPLIT)
TEST_SPLITS = (TEST_SPLIT, TRAIN_SPLIT)               # the splits every combination is run on
ESC_TRAIN = {TEST_SPLIT: TRAIN_SPLIT, TRAIN_SPLIT: TEST_SPLIT}  # ESC's training split per test split

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

# Test sets: source, and gold M2 where it is public. CoNLL-2014's gold is in Git LFS (git lfs pull
# before scoring); BEA-2019 has no public gold, so its output is scored on Codabench.
TEST_SETS = {
    'conll14': (Path('data/test.conll14.src'), Path('data/test.conll14.m2')),
    'bea19': (Path('data/test.bea19.src'), None),
}

# What test.sh runs on the test sets: (name, k, vote_coef, esc, vote_mode). One system, chosen
# beforehand as the best ERRANT F0.5 on the full dev set (= BEA-2019 dev, as in the GRECO paper;
# runs/dev/results.md): GRECO + ESC at k = 3 with rank voting, alpha 0.4 (60.4). ESC alone at
# k = 3 and the three single systems are scored as well. ESC is trained once on the full dev set
# (dev_tune + dev_test), as the paper does on BEA-2019 dev.
TEST_RUNS = [
    ('greco+esc.k3.rank0.4', 3, 0.4, True, 'rank'),
]
FINAL_K = 3  # k of the full-dev ESC model, used by the final system and by ESC alone

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


def pred_path(k, name, split=TEST_SPLIT):
    return ROOT / split / f'k{k}' / name / 'pred.txt'


def source(split):
    """The source sentences of a dev split or a test set."""
    return TEST_SETS[split][0] if split in TEST_SETS else DATA / f'{split}.src'


def esc_home(k, test_split=TEST_SPLIT):
    """ESC's folders and model for scoring test_split at k. The dev_test ones keep their
    original place (runs/esc/k<k>), so a finished run is reused."""
    if test_split in TEST_SETS:
        return ROOT / 'esc' / 'final' / f'k{k}'  # trained on the full dev set, shared by the test sets
    home = ROOT / 'esc' / f'k{k}'
    return home if test_split == TEST_SPLIT else home / f'for-{test_split}'


def main(argv):
    if argv[:1] == ['runs']:
        for name, k, vote, esc, mode in runs():
            print(name, k, f'{vote:g}', int(esc), mode)
    elif argv[:1] == ['inputs'] and len(argv) == 2:
        print(' '.join(inputs(int(argv[1]))))
    elif argv == ['splits']:
        for split in TEST_SPLITS:
            print(split, ESC_TRAIN[split])
    elif argv[:1] == ['esc_home'] and len(argv) == 3:
        print(esc_home(int(argv[1]), argv[2]))
    elif argv == ['test_runs']:
        for name, k, vote, esc, mode in TEST_RUNS:
            print(name, k, f'{vote:g}', int(esc), mode)
    elif argv[:1] == ['source'] and len(argv) == 2:
        print(source(argv[1]))
    else:
        sys.exit('usage: python3 -m greco_esc.settings runs | inputs <k> | splits | esc_home <k> <split> '
                 '| test_runs | source <split>')


if __name__ == '__main__':
    main(sys.argv[1:])
