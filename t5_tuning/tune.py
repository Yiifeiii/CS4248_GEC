"""Stage-2 tuning for T5: plan the runs, submit them, and collect the results.

    python t5_tuning/tune.py plan --phase 1                  # noise band and learning-rate sweep
    python t5_tuning/tune.py submit --phase 1                # prints the sbatch commands
    python t5_tuning/tune.py submit --phase 1 --run          # submits them, on xlogin
    python t5_tuning/tune.py collect                         # band, per-knob table, chosen values
    python t5_tuning/tune.py plan --phase 2 --lr 3e-5        # wd, dropout, label smoothing at lr*
    python t5_tuning/tune.py submit --phase 2 --run
    python t5_tuning/tune.py collect

Every run starts from the frozen stage-1 model (CS4248-GEC/gec-t5-base-s1) and uses
training/configs/t5-base.stage2.json, changed only in the keys its knob varies. Each run
is trained and then scored on dev_tune only (t5_tuning/run_one.sh). dev_test is never read.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'training' / 'configs' / 't5-base.stage2.json'
CONFIGS = HERE / 'configs'
RUNS = HERE / 'runs'
MANIFEST = HERE / 'manifest.json'
MANIFEST_TABLE = HERE / 'manifest.md'
RESULTS = HERE / 'results.md'

INIT = 'CS4248-GEC/gec-t5-base-s1'      # the frozen stage-1 model
SEEDS = [42, 123, 7]                     # the noise-floor seeds
DEFAULT = {'lr': '3e-5', 'wd': '0.01', 'dropout': '0.1', 'ls': '0.1'}
GRID = {'lr': ['1e-5', '3e-5', '5e-5'], 'wd': ['0', '0.01', '0.1'],
        'dropout': ['0', '0.1', '0.3'], 'ls': ['0', '0.1', '0.2']}
PHASE2_KNOBS = ['wd', 'dropout', 'ls']


def make_row(phase, knob, seed, **overrides):
    """One run: the knob it varies, its settings, and its run name."""
    params = {**DEFAULT, **overrides}
    tag = f'{knob}{overrides[knob]}' if knob in PHASE2_KNOBS else 'base'
    run = f"t5-tune.{tag}.lr{params['lr']}.seed{seed}"
    return {'phase': phase, 'knob': knob, 'run': run, 'seed': seed, **params,
            'config': f'{CONFIGS.relative_to(ROOT)}/{run}.json'}


def write_config(row):
    cfg = json.loads(BASE.read_text(encoding='utf-8'))
    cfg.update({
        'run': row['run'], 'init': INIT,
        'learning_rate': float(row['lr']), 'weight_decay': float(row['wd']),
        'dropout': float(row['dropout']), 'label_smoothing': float(row['ls']),
        'seed': row['seed'], 'output_dir': f"${{CKPT_ROOT}}/{row['run']}",
    })
    (ROOT / row['config']).write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}


def save_manifest(manifest):
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    lines = ['| run | phase | knob | lr | weight decay | dropout | label smoothing | seed | config |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in manifest.values():
        lines.append(f"| {r['run']} | {r['phase']} | {r['knob']} | {r['lr']} | {r['wd']} | {r['dropout']} "
                     f"| {r['ls']} | {r['seed']} | {r['config']} |")
    MANIFEST_TABLE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plan(args):
    if args.phase == 1:
        rows = [make_row(1, 'base', seed) for seed in SEEDS]
        rows += [make_row(1, 'lr', 42, lr=v) for v in GRID['lr'] if v != DEFAULT['lr']]
    else:
        if not args.lr:
            raise SystemExit('phase 2 needs --lr, the learning rate chosen in phase 1')
        # At the default learning rate the baseline is the phase-1 seed-42 run, already in the manifest.
        baseline = make_row(2, 'base', 42, lr=args.lr)
        rows = [] if baseline['run'] in load_manifest() else [baseline]
        for knob in PHASE2_KNOBS:
            rows += [make_row(2, knob, 42, lr=args.lr, **{knob: v})
                     for v in GRID[knob] if v != DEFAULT[knob]]
    CONFIGS.mkdir(exist_ok=True)
    manifest = load_manifest()
    for row in rows:
        write_config(row)
        manifest[row['run']] = row
    save_manifest(manifest)
    for row in rows:
        print(row['config'])


def submit(args):
    for row in load_manifest().values():
        if row['phase'] != args.phase:
            continue
        if (RUNS / row['run'] / 'dev_tune.scores.json').exists():
            print(f"skip {row['run']}: already scored")
            continue
        cmd = ['sbatch', '-J', row['run'], 'cluster/job.sbatch', 't5_tuning/run_one.sh', row['config']]
        if args.run:
            subprocess.run(cmd, check=True, cwd=ROOT)
        else:
            print(' '.join(cmd))


def f05(run):
    """Correction-level ERRANT F0.5 on dev_tune, as a percentage, or None if the run isn't scored."""
    path = RUNS / run / 'dev_tune.scores.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))['errant']['correction']['f0_5'] * 100


def judge(title, knob, rows, reference, scores, band):
    """Print one knob's candidates against its reference, and return the chosen value."""
    ref_score = scores[reference['run']]
    print(f'\n{title}: reference {knob}={reference[knob]}'
          + (f' (F0.5 {ref_score:.2f})' if ref_score is not None else ' (not finished)'))
    for r in sorted(rows, key=lambda r: float(r[knob])):
        score = scores[r['run']]
        if score is None:
            print(f'  {r[knob]:>8}  not finished')
        elif r is reference:
            print(f'  {r[knob]:>8}  {score:6.2f}  reference')
        else:
            delta = score - ref_score
            verdict = 'inside band' if band is None or abs(delta) <= band else 'outside band'
            print(f'  {r[knob]:>8}  {score:6.2f}  {delta:+6.2f}  {verdict}')

    if band is None or ref_score is None:
        print('  no choice yet: the band or the reference is missing')
        return None
    finished = [r for r in rows if scores[r['run']] is not None]
    best = max(finished, key=lambda r: scores[r['run']])
    delta = scores[best['run']] - ref_score
    if best is reference or delta <= 0:
        chosen = reference
        note = 'keep the default: no candidate is better'
    elif delta > band:
        chosen = best
        note = 'beats the default by more than the band'
    else:
        chosen = reference
        note = 'beats the default by less than the band: run a second seed before reporting it'
    print(f'  chosen {knob}={chosen[knob]} ({note})')
    if chosen[knob] in (GRID[knob][0], GRID[knob][-1]):
        print(f'  FLAG: {knob}={chosen[knob]} is an endpoint of the grid; the optimum may be outside it')
    return chosen[knob]


def collect(args):
    manifest = load_manifest()
    scores = {run: f05(run) for run in manifest}

    # The noise band: spread of the seed runs at the default learning rate.
    seed_rows = [r for r in manifest.values()
                 if r['knob'] == 'base' and r['lr'] == DEFAULT['lr'] and r['seed'] in SEEDS
                 and scores[r['run']] is not None]
    values = [scores[r['run']] for r in seed_rows]
    band = None
    if len(values) >= 2:
        band = max(values) - min(values)
        print(f'noise band from {len(values)} seeds at lr {DEFAULT["lr"]}: mean {statistics.mean(values):.2f}, '
              f'sd {statistics.stdev(values):.2f}, range {band:.2f}')
    else:
        print('noise band: not measured yet (needs at least two finished seed runs)')

    # Learning rate: all runs at seed 42 with the other knobs at default.
    lr_rows = [r for r in manifest.values() if r['knob'] in ('base', 'lr') and r['seed'] == 42
               and all(r[k] == DEFAULT[k] for k in ('wd', 'dropout', 'ls'))]
    lr_ref = next((r for r in lr_rows if r['lr'] == DEFAULT['lr']), None)
    chosen = {}
    if lr_ref:
        chosen['lr'] = judge('learning rate (seed 42)', 'lr', lr_rows, lr_ref, scores, band)

    # Phase 2: each knob at the chosen learning rate, against the baseline at that rate.
    # One group per phase-2 learning rate, so runs at a rate chosen later don't mix with earlier ones.
    phase2_lrs = sorted({r['lr'] for r in manifest.values() if r['phase'] == 2 and r['knob'] != 'base'}, key=float)
    for lr_star in phase2_lrs:
        p2_ref = next((r for r in manifest.values()
                       if r['knob'] == 'base' and r['lr'] == lr_star and r['seed'] == 42), None)
        if p2_ref is None:
            continue
        for knob in PHASE2_KNOBS:
            rows = [r for r in manifest.values() if r['lr'] == lr_star and r['seed'] == 42
                    and (r['knob'] == knob or r is p2_ref)]
            chosen[f'{knob}@lr{lr_star}'] = judge(f'{knob} (at lr {lr_star}, seed 42)', knob, rows, p2_ref,
                                                  scores, band)

    chosen = {k: v for k, v in chosen.items() if v is not None}
    if chosen:
        print('\nchosen: ' + ', '.join(f'{k}={v}' for k, v in chosen.items()))

    lines = ['| run | phase | knob | lr | wd | dropout | ls | seed | dev_tune F0.5 |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in manifest.values():
        s = scores[r['run']]
        lines.append(f"| {r['run']} | {r['phase']} | {r['knob']} | {r['lr']} | {r['wd']} | {r['dropout']} "
                     f"| {r['ls']} | {r['seed']} | {'not finished' if s is None else f'{s:.2f}'} |")
    RESULTS.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\ntable written to {RESULTS}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('plan', help='write the configs and update the manifest')
    p.add_argument('--phase', type=int, choices=[1, 2], required=True)
    p.add_argument('--lr', help='phase 2: the chosen learning rate, as written, e.g. 3e-5')
    p.set_defaults(func=plan)

    p = sub.add_parser('submit', help='submit one phase with sbatch')
    p.add_argument('--phase', type=int, choices=[1, 2], required=True)
    p.add_argument('--run', action='store_true', help='submit; without this, only print the commands')
    p.set_defaults(func=submit)

    p = sub.add_parser('collect', help='band, per-knob comparisons, chosen values')
    p.set_defaults(func=collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
