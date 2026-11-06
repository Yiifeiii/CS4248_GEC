"""T5 tuning: plan the runs, submit them, and collect the results, for stage 1 or stage 2.

Stage 2 (the default), from the frozen stage-1 model:
    python t5_tuning/tune.py plan --phase 1                  # noise band and learning-rate sweep
    python t5_tuning/tune.py submit --phase 1 [--run]        # prints the sbatch commands; --run submits
    python t5_tuning/tune.py collect                         # band, one results table per hyperparameter, chosen values
    python t5_tuning/tune.py plan --phase 2 --lr 3e-5        # wd, dropout, label smoothing at lr*
    python t5_tuning/tune.py submit --phase 2 [--run]
    python t5_tuning/tune.py collect

Stage 2 from a different stage-1 model, under its own tag (runs and results are kept apart):
    python t5_tuning/tune.py plan --phase 1 --tag s1lr3e-4 --init '${CKPT_ROOT}/t5-tune.s1.base.lr3e-4.seed42/best'
    python t5_tuning/tune.py submit --phase 1 --tag s1lr3e-4 [--run]
    python t5_tuning/tune.py collect --tag s1lr3e-4

Stage 1, from the public T5 base model (a learning-rate sweep only):
    python t5_tuning/tune.py plan --stage 1 --phase 1
    python t5_tuning/tune.py submit --stage 1 --phase 1 [--run]
    python t5_tuning/tune.py collect --stage 1

Every run is trained and then scored on dev_tune only (t5_tuning/run_one.sh). dev_test is never read.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIGS = HERE / 'configs'
RUNS = HERE / 'runs'
MANIFEST = HERE / 'manifest.json'
MANIFEST_TABLE = HERE / 'manifest.md'

SEEDS = [42, 123, 7]                     # the noise-floor seeds
GRID = {'wd': ['0', '0.01', '0.1'], 'dropout': ['0', '0.1', '0.3'], 'ls': ['0', '0.1', '0.2']}
PHASE2_KNOBS = ['wd', 'dropout', 'ls']

# Stage 1 starts from the public base model, so its learning rate is higher than stage 2's.
# Stage 2 starts from the frozen stage-1 model. Run names of stage 2 have no stage tag, so the
# runs already on the cluster keep their names.
STAGES = {
    1: {
        'base_config': ROOT / 'training' / 'configs' / 't5-base.stage1.json',
        'init': None,                    # keep the base config's init (google-t5/t5-base)
        'prefix': 't5-tune.s1',
        'default': {'lr': '1e-4', 'wd': '0.01', 'dropout': '0.1', 'ls': '0.1'},
        'lr_grid': ['3e-5', '1e-4', '3e-4'],
        'results': HERE / 'results-stage1.md',
    },
    2: {
        'base_config': ROOT / 'training' / 'configs' / 't5-base.stage2.json',
        'init': 'CS4248-GEC/gec-t5-base-s1',   # the frozen stage-1 model
        'prefix': 't5-tune',
        'default': {'lr': '3e-5', 'wd': '0.01', 'dropout': '0.1', 'ls': '0.1'},
        'lr_grid': ['1e-5', '3e-5', '5e-5'],
        'results': HERE / 'results.md',
    },
}


def make_row(stage, phase, knob, seed, tag=None, init=None, **overrides):
    """One run: its stage, tag, starting model, the knob it varies, its settings, and its run name."""
    cfg = STAGES[stage]
    params = {**cfg['default'], **overrides}
    knob_tag = f'{knob}{overrides[knob]}' if knob in PHASE2_KNOBS else 'base'
    prefix = cfg['prefix'] if tag is None else f't5-tune.{tag}'
    run = f"{prefix}.{knob_tag}.lr{params['lr']}.seed{seed}"
    return {'stage': stage, 'phase': phase, 'tag': tag, 'init': init, 'knob': knob, 'run': run,
            'seed': seed, **params, 'config': f'{CONFIGS.relative_to(ROOT)}/{run}.json'}


def write_config(row):
    cfg = STAGES[row['stage']]
    settings = json.loads(cfg['base_config'].read_text(encoding='utf-8'))
    settings.update({
        'run': row['run'],
        'learning_rate': float(row['lr']), 'weight_decay': float(row['wd']),
        'dropout': float(row['dropout']), 'label_smoothing': float(row['ls']),
        'seed': row['seed'], 'output_dir': f"${{CKPT_ROOT}}/{row['run']}",
    })
    init = row.get('init') or cfg['init']
    if init:
        settings['init'] = init
    (ROOT / row['config']).write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}


def save_manifest(manifest):
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    lines = ['| run | stage | tag | phase | knob | lr | weight decay | dropout | label smoothing | seed | config |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in manifest.values():
        lines.append(f"| {r['run']} | {r.get('stage', 2)} | {r.get('tag') or ''} | {r['phase']} | {r['knob']} "
                     f"| {r['lr']} | {r['wd']} | {r['dropout']} | {r['ls']} | {r['seed']} | {r['config']} |")
    MANIFEST_TABLE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plan(args):
    stage = args.stage
    if args.init and not args.tag:
        raise SystemExit('--init needs --tag, so the new runs get their own names')
    if args.init and stage != 2:
        raise SystemExit('--init is only for stage 2')
    default = STAGES[stage]['default']
    common = {'tag': args.tag, 'init': args.init}
    if args.phase == 1:
        rows = [make_row(stage, 1, 'base', seed, **common) for seed in SEEDS]
        rows += [make_row(stage, 1, 'lr', 42, lr=v, **common)
                 for v in STAGES[stage]['lr_grid'] if v != default['lr']]
    else:
        if stage != 2:
            raise SystemExit('phase 2 (weight decay, dropout, label smoothing) is only set up for stage 2')
        if not args.lr:
            raise SystemExit('phase 2 needs --lr, the learning rate chosen in phase 1')
        # At the default learning rate the baseline is the phase-1 seed-42 run, already in the manifest.
        baseline = make_row(2, 2, 'base', 42, lr=args.lr, **common)
        rows = [] if baseline['run'] in load_manifest() else [baseline]
        for knob in PHASE2_KNOBS:
            rows += [make_row(2, 2, knob, 42, lr=args.lr, **common, **{knob: v})
                     for v in GRID[knob] if v != default[knob]]
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
        if row.get('stage', 2) != args.stage or row.get('tag') != args.tag or row['phase'] != args.phase:
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


def judge(title, knob, rows, reference, scores, band, grid):
    """Print one hyperparameter's candidates against its reference, and return the chosen value."""
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
    if chosen[knob] in (grid[0], grid[-1]):
        print(f'  FLAG: {knob}={chosen[knob]} is an endpoint of the grid; the optimum may be outside it')
    return chosen[knob]


def collect(args):
    stage = args.stage
    cfg = STAGES[stage]
    default = cfg['default']
    manifest = load_manifest()
    rows = [r for r in manifest.values() if r.get('stage', 2) == stage and r.get('tag') == args.tag]
    scores = {r['run']: f05(r['run']) for r in rows}
    print(f'stage {stage}' + (f', tag {args.tag}' if args.tag else ''))

    # The noise band: spread of the seed runs at the default learning rate.
    seed_rows = [r for r in rows if r['knob'] == 'base' and r['lr'] == default['lr'] and r['seed'] in SEEDS
                 and scores[r['run']] is not None]
    values = [scores[r['run']] for r in seed_rows]
    band = None
    if len(values) >= 2:
        band = max(values) - min(values)
        print(f'noise band from {len(values)} seeds at lr {default["lr"]}: mean {statistics.mean(values):.2f}, '
              f'sd {statistics.stdev(values):.2f}, range {band:.2f}')
    else:
        print('noise band: not measured yet (needs at least two finished seed runs)')

    # Learning rate: all runs at seed 42 with the other knobs at default.
    lr_rows = [r for r in rows if r['knob'] in ('base', 'lr') and r['seed'] == 42
               and all(r[k] == default[k] for k in ('wd', 'dropout', 'ls'))]
    lr_ref = next((r for r in lr_rows if r['lr'] == default['lr']), None)
    chosen = {}
    if lr_ref:
        chosen['lr'] = judge('learning rate (seed 42)', 'lr', lr_rows, lr_ref, scores, band, cfg['lr_grid'])

    # Phase 2 (stage 2 only): each hyperparameter at a phase-2 learning rate, against the baseline at that rate.
    # One group per learning rate, so runs at different rates are never compared.
    if stage == 2:
        phase2_lrs = sorted({r['lr'] for r in rows if r['phase'] == 2 and r['knob'] != 'base'}, key=float)
        for lr_star in phase2_lrs:
            p2_ref = next((r for r in rows
                           if r['knob'] == 'base' and r['lr'] == lr_star and r['seed'] == 42), None)
            if p2_ref is None:
                continue
            for knob in PHASE2_KNOBS:
                group = [r for r in rows if r['lr'] == lr_star and r['seed'] == 42
                         and (r['knob'] == knob or r is p2_ref)]
                chosen[f'{knob}@lr{lr_star}'] = judge(f'{knob} (at lr {lr_star}, seed 42)', knob, group,
                                                      p2_ref, scores, band, GRID[knob])

    chosen = {k: v for k, v in chosen.items() if v is not None}
    if chosen:
        print('\nchosen: ' + ', '.join(f'{k}={v}' for k, v in chosen.items()))

    lines = ['| run | phase | knob | lr | wd | dropout | ls | seed | dev_tune F0.5 |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        s = scores[r['run']]
        lines.append(f"| {r['run']} | {r['phase']} | {r['knob']} | {r['lr']} | {r['wd']} | {r['dropout']} "
                     f"| {r['ls']} | {r['seed']} | {'not finished' if s is None else f'{s:.2f}'} |")
    results_path = cfg['results'] if args.tag is None else HERE / f'results-{args.tag}.md'
    results_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\ntable written to {results_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('plan', help='write the configs and update the manifest')
    p.add_argument('--stage', type=int, choices=[1, 2], default=2)
    p.add_argument('--phase', type=int, choices=[1, 2], required=True)
    p.add_argument('--lr', help='phase 2: the chosen learning rate, as written, e.g. 3e-5')
    p.add_argument('--tag', help='name the runs under this tag, e.g. s1lr3e-4, and keep their results apart')
    p.add_argument('--init', help='stage 2: the starting model, e.g. ${CKPT_ROOT}/<stage-1 run>/best (needs --tag)')
    p.set_defaults(func=plan)

    p = sub.add_parser('submit', help='submit one phase with sbatch')
    p.add_argument('--stage', type=int, choices=[1, 2], default=2)
    p.add_argument('--phase', type=int, choices=[1, 2], required=True)
    p.add_argument('--tag', help='submit only the runs planned under this tag')
    p.add_argument('--run', action='store_true', help='submit; without this, only print the commands')
    p.set_defaults(func=submit)

    p = sub.add_parser('collect', help='band, comparisons and chosen values')
    p.add_argument('--stage', type=int, choices=[1, 2], default=2)
    p.add_argument('--tag', help='collect only the runs planned under this tag')
    p.set_defaults(func=collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
