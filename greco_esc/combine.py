"""Upstream GRECO's run_combination.py, with the greco-voting branch's three ways of counting votes.

Upstream ranks a candidate by (1 - r) log Q + alpha log V + r (ESC term), where V averages the
support of the candidate's edits, and an edit's support is the share of input files proposing it
("hypothesis"). Upstream merges the inputs before counting, so it cannot tell which system or rank
an edit came from. This wrapper keeps upstream's beam search, GRECO scorer and ESC edit scores
unchanged and only replaces the support, as greco_setup/combine.py on the greco-voting branch
defines it:

    hypothesis  share of inputs proposing the edit (upstream, unchanged)
    system      share of systems with at least one input proposing it (the paper's count(e) / c)
    rank        sum over systems of 1 / (best rank proposing it), divided by the number of systems

As upstream, the unchanged sentence gets one voter's share (1 / inputs or 1 / systems). At k = 1
every system has one input, so the three modes give the same output.

Each input's system and rank come from its file name, <system>-<rank>.m2 (prepare.py writes
gector-1.m2, bart-1.m2, ..., t5-5.m2). Run from external/greco, like run_combination.py, with
its arguments plus --vote_mode:

    cd external/greco && python ../../greco_esc/combine.py --vote_mode rank --vote_coef 0.4 ...
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

MODES = ('hypothesis', 'system', 'rank')
_NAME = re.compile(r'^(?P<system>.+)-(?P<rank>[1-9][0-9]*)\.m2$')


def system_rank(path):
    """('bart', 2) for .../bart-2.m2."""
    match = _NAME.match(Path(path).name)
    if not match:
        raise ValueError(f'{path}: input names must be <system>-<rank>.m2 for --vote_mode system or rank')
    return match['system'], int(match['rank'])


def support(edit_sets, labels, mode):
    """Support of every edit in one sentence, and the unchanged sentence's share.

    edit_sets: one set of (start, end, correction) per input; labels: (system, rank) per input.
    """
    if mode == 'hypothesis':
        counts = defaultdict(int)
        for edits in edit_sets:
            for e in edits:
                counts[e] += 1
        return {e: n / len(edit_sets) for e, n in counts.items()}, 1 / len(edit_sets)
    systems = {s for s, _ in labels}
    best = defaultdict(dict)  # edit -> system -> best rank proposing it
    for edits, (s, r) in zip(edit_sets, labels):
        for e in edits:
            best[e][s] = min(r, best[e].get(s, r))
    c = len(systems)
    if mode == 'system':
        return {e: len(by_sys) / c for e, by_sys in best.items()}, 1 / c
    if mode == 'rank':
        return {e: sum(1 / r for r in by_sys.values()) / c for e, by_sys in best.items()}, 1 / c
    raise ValueError(f'unknown vote mode {mode!r}')


def voted_dataset(files, edit_scores, mode):
    """Upstream's SysCombDataset, with each edit's count replaced by the chosen support."""
    from utils.data import SysCombDataset
    from utils.edit import m2_to_edits, read_m2

    class VotedDataset(SysCombDataset):
        def __init__(self):
            super().__init__(files, False, edit_scores)
            if mode == 'hypothesis':
                return  # upstream's own counts
            labels = [system_rank(f) for f in files]
            per_file = [[{(e[0], e[1], e[3]) for e in m2_to_edits(m)['edits']} for m in read_m2(f)]
                        for f in files]
            self.supports = []
            for i in range(len(self.sources)):
                sup, baseline = support([edits[i] for edits in per_file], labels, mode)
                self.supports.append(sup)
            # beam_search starts V at 1 / num_hyp and adds count / num_hyp per applied edit:
            # with num_hyp = number of systems and count = support * num_hyp, V is the mean support
            self.num_hyp = round(1 / baseline)

        def __getitem__(self, index):
            data = super().__getitem__(index)
            if mode != 'hypothesis':
                sup = self.supports[index]
                data['edits'] = [(edit, sup[(edit[0], edit[1], edit[3])] * self.num_hyp)
                                 for edit, _ in data['edits']]
            return data

    return VotedDataset()


def main():
    sys.path[0] = os.getcwd()  # import upstream (external/greco), not this folder
    import torch
    import run_combination
    from models import get_model

    sys.argv, mode = _pop_vote_mode(sys.argv)
    args = run_combination.get_arguments()
    dataset = voted_dataset(args.data, args.edit_scores, mode)
    print(f'vote mode: {mode}, voters: {dataset.num_hyp}')
    model = get_model(args)
    model.eval()
    with torch.no_grad():
        result = run_combination.beam_search(dataset, model, args.beam_size, args.batch_size,
                                             args.score_ratio, args.vote_coef, args.verbose)
    with open(args.output_path, 'w', encoding='utf-8') as out:
        out.write('\n'.join(result))


def _pop_vote_mode(argv):
    """Remove --vote_mode from argv, so upstream's own argument parser accepts the rest."""
    argv = list(argv)
    if '--vote_mode' not in argv:
        return argv, 'hypothesis'
    i = argv.index('--vote_mode')
    mode = argv[i + 1] if i + 1 < len(argv) else ''
    if mode not in MODES:
        sys.exit(f'--vote_mode must be one of {", ".join(MODES)}')
    return argv[:i] + argv[i + 2:], mode


if __name__ == '__main__':
    main()
