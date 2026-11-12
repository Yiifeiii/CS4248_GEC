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

Every run is trained and then scored on dev_tune only (t5_tuning/run_one.sh). dev_test is never read
until the final configuration is fixed.

Checks and the final dev_test scoring:
    python t5_tuning/tune.py preflight [--stage 1] [--phase 1] [--tag T]   # CPU, before any GPU run; submit runs it
    python t5_tuning/tune.py verify --run <run> [--run ...]               # on a finished run; run_one.sh calls it
    python t5_tuning/tune.py test [--stage 1] [--tag T] [--lr 3e-5] [--run]   # dev_test once per seed of the final config
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIGS = ROOT / 'configs' / 't5_tuning'
RUNS = HERE / 'runs'
MANIFEST = HERE / 'manifest.json'
MANIFEST_TABLE = HERE / 'manifest.md'

PREFLIGHT = HERE / 'preflight.json'
DATA_MANIFEST = ROOT / 'data' / 'curriculum' / 'manifest.json'
REFERENCE_RUN = ROOT / 'experiments' / 'training' / 't5-base.s1s2' / 'run_info.json'  # versions of the recorded runs
CHECKSUMS = ROOT / 'experiments' / 'training' / 'checkpoints.sha256'
STAGE1_REPO, STAGE1_HASH_KEY = 'CS4248-GEC/gec-t5-base-s1', 't5-base.s1/best/model.safetensors'
SPLIT_SIZES = {'dev_tune': 2000, 'dev_test': 2384}
CONFIG_FIELDS = {'learning_rate': 'lr', 'weight_decay': 'wd', 'dropout': 'dropout', 'label_smoothing': 'ls'}

SEEDS = [42, 123, 7]                     # the noise-floor seeds
GRID = {'wd': ['0', '0.01', '0.1'], 'dropout': ['0', '0.1', '0.3'], 'ls': ['0', '0.1', '0.2']}
PHASE2_PARAMS = ['wd', 'dropout', 'ls']

# Stage 1 starts from the public base model, so its learning rate is higher than stage 2's.
# Stage 2 starts from the frozen stage-1 model. Run names of stage 2 have no stage tag, so the
# runs already on the cluster keep their names.
STAGES = {
    1: {
        'base_config': ROOT / 'configs' / 'training' / 't5-base.stage1.json',
        'init': None,                    # keep the base config's init (google-t5/t5-base)
        'prefix': 't5-tune.s1',
        'default': {'lr': '1e-4', 'wd': '0.01', 'dropout': '0.1', 'ls': '0.1'},
        'lr_grid': ['3e-5', '1e-4', '3e-4'],
        'results': HERE / 'results-stage1.md',
        'data': 'stage1',                # its key in data/curriculum/manifest.json
    },
    2: {
        'base_config': ROOT / 'configs' / 'training' / 't5-base.stage2.json',
        'init': 'CS4248-GEC/gec-t5-base-s1',   # the frozen stage-1 model
        'prefix': 't5-tune',
        'default': {'lr': '3e-5', 'wd': '0.01', 'dropout': '0.1', 'ls': '0.1'},
        'lr_grid': ['1e-5', '3e-5', '5e-5'],
        'results': HERE / 'results.md',
        'data': 'stage2',
    },
}


def make_row(stage, phase, param, seed, tag=None, init=None, **overrides):
    """One run: its stage, tag, starting model, the hyperparameter it varies, its settings, and its run name."""
    cfg = STAGES[stage]
    params = {**cfg['default'], **overrides}
    param_tag = f'{param}{overrides[param]}' if param in PHASE2_PARAMS else 'base'
    prefix = cfg['prefix'] if tag is None else f't5-tune.{tag}'
    run = f"{prefix}.{param_tag}.lr{params['lr']}.seed{seed}"
    return {'stage': stage, 'phase': phase, 'tag': tag, 'init': init, 'param': param, 'run': run,
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
    lines = ['| run | stage | tag | phase | param | lr | weight decay | dropout | label smoothing | seed | config |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in manifest.values():
        lines.append(f"| {r['run']} | {r.get('stage', 2)} | {r.get('tag') or ''} | {r['phase']} | {r['param']} "
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
        for param in PHASE2_PARAMS:
            rows += [make_row(2, 2, param, 42, lr=args.lr, **common, **{param: v})
                     for v in GRID[param] if v != default[param]]
    CONFIGS.mkdir(exist_ok=True)
    manifest = load_manifest()
    for row in rows:
        write_config(row)
        manifest[row['run']] = row
    save_manifest(manifest)
    for row in rows:
        print(row['config'])


def submit(args):
    """Submit the CPU preflight, then every unscored run of the phase, each depending on the preflight."""
    pending = []
    for row in selected(args.stage, args.phase, args.tag):
        if (RUNS / row['run'] / 'dev_tune.scores.json').exists():
            print(f"skip {row['run']}: already scored")
        else:
            pending.append(row)
    if not pending:
        print('nothing to submit')
        return

    scope = ['--stage', str(args.stage), '--phase', str(args.phase)] + (['--tag', args.tag] if args.tag else [])
    check = ['sbatch', '--parsable', '-J', 't5-tune-preflight', '--partition=normal', '--gpus=0',
             '--cpus-per-task=4', '--mem-per-cpu=6000', '--time=00:45:00',
             'cluster/job.sbatch', 't5_tuning/preflight.sh', *scope]
    if args.run:
        job = subprocess.run(check, check=True, cwd=ROOT, capture_output=True, text=True).stdout.strip().split(';')[0]
        print(f'preflight job {job}')
    else:
        print(' '.join(check))
        job = '<preflight job id>'
    for row in pending:
        cmd = ['sbatch', f'--dependency=afterok:{job}', '--kill-on-invalid-dep=yes',
               '-J', row['run'], 'cluster/job.sbatch', 't5_tuning/run_one.sh', row['config']]
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


def judge(title, param, rows, reference, scores, band, grid):
    """Print one hyperparameter's candidates against its reference, and return the chosen value."""
    ref_score = scores[reference['run']]
    print(f'\n{title}: reference {param}={reference[param]}'
          + (f' (F0.5 {ref_score:.2f})' if ref_score is not None else ' (not finished)'))
    for r in sorted(rows, key=lambda r: float(r[param])):
        score = scores[r['run']]
        if score is None:
            print(f'  {r[param]:>8}  not finished')
        elif r is reference:
            print(f'  {r[param]:>8}  {score:6.2f}  reference')
        else:
            delta = score - ref_score
            verdict = 'inside band' if band is None or abs(delta) <= band else 'outside band'
            print(f'  {r[param]:>8}  {score:6.2f}  {delta:+6.2f}  {verdict}')

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
    print(f'  chosen {param}={chosen[param]} ({note})')
    if chosen[param] in (grid[0], grid[-1]):
        print(f'  FLAG: {param}={chosen[param]} is an endpoint of the grid; the optimum may be outside it')
    return chosen[param]


def collect(args):
    stage = args.stage
    cfg = STAGES[stage]
    default = cfg['default']
    manifest = load_manifest()
    rows = [r for r in manifest.values() if r.get('stage', 2) == stage and r.get('tag') == args.tag]
    scores = {r['run']: f05(r['run']) for r in rows}
    print(f'stage {stage}' + (f', tag {args.tag}' if args.tag else ''))

    # The noise band: spread of the seed runs at the default learning rate.
    seed_rows = [r for r in rows if r['param'] == 'base' and r['lr'] == default['lr'] and r['seed'] in SEEDS
                 and scores[r['run']] is not None]
    values = [scores[r['run']] for r in seed_rows]
    band = None
    if len(values) >= 2:
        band = max(values) - min(values)
        print(f'noise band from {len(values)} seeds at lr {default["lr"]}: mean {statistics.mean(values):.2f}, '
              f'sd {statistics.stdev(values):.2f}, range {band:.2f}')
    else:
        print('noise band: not measured yet (needs at least two finished seed runs)')

    # Learning rate: all runs at seed 42 with the other hyperparameters at default.
    lr_rows = [r for r in rows if r['param'] in ('base', 'lr') and r['seed'] == 42
               and all(r[k] == default[k] for k in ('wd', 'dropout', 'ls'))]
    lr_ref = next((r for r in lr_rows if r['lr'] == default['lr']), None)
    chosen = {}
    if lr_ref:
        chosen['lr'] = judge('learning rate (seed 42)', 'lr', lr_rows, lr_ref, scores, band, cfg['lr_grid'])

    # Phase 2 (stage 2 only): each hyperparameter at a phase-2 learning rate, against the baseline at that rate.
    # One group per learning rate, so runs at different rates are never compared.
    if stage == 2:
        phase2_lrs = sorted({r['lr'] for r in rows if r['phase'] == 2 and r['param'] != 'base'}, key=float)
        for lr_star in phase2_lrs:
            p2_ref = next((r for r in rows
                           if r['param'] == 'base' and r['lr'] == lr_star and r['seed'] == 42), None)
            if p2_ref is None:
                continue
            for param in PHASE2_PARAMS:
                group = [r for r in rows if r['lr'] == lr_star and r['seed'] == 42
                         and (r['param'] == param or r is p2_ref)]
                chosen[f'{param}@lr{lr_star}'] = judge(f'{param} (at lr {lr_star}, seed 42)', param, group,
                                                      p2_ref, scores, band, GRID[param])

    chosen = {k: v for k, v in chosen.items() if v is not None}
    if chosen:
        print('\nchosen: ' + ', '.join(f'{k}={v}' for k, v in chosen.items()))

    lines = ['| run | phase | param | lr | wd | dropout | ls | seed | dev_tune F0.5 |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        s = scores[r['run']]
        lines.append(f"| {r['run']} | {r['phase']} | {r['param']} | {r['lr']} | {r['wd']} | {r['dropout']} "
                     f"| {r['ls']} | {r['seed']} | {'not finished' if s is None else f'{s:.2f}'} |")
    results_path = cfg['results'] if args.tag is None else HERE / f'results-{args.tag}.md'
    results_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\ntable written to {results_path}')


def selected(stage=None, phase=None, tag=None, runs=None):
    """Manifest rows matching the filters; runs, if given, picks rows by name instead."""
    manifest = load_manifest()
    if runs:
        missing = [r for r in runs if r not in manifest]
        if missing:
            raise SystemExit(f'not in the manifest: {", ".join(missing)}')
        return [manifest[r] for r in runs]
    return [r for r in manifest.values()
            if (stage is None or r.get('stage', 2) == stage) and r.get('tag') == tag
            and (phase is None or r['phase'] == phase)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def recorded_versions():
    return json.loads(REFERENCE_RUN.read_text(encoding='utf-8'))['versions']


def start_model(row):
    """The model a run starts from: its own init, its stage's, or the base config's."""
    cfg = STAGES[row['stage']]
    return (row.get('init') or cfg['init']
            or json.loads(cfg['base_config'].read_text(encoding='utf-8'))['init'])


def same(a, b):
    """Config values match: numbers by value (1e-05 == 0.00001), everything else exactly."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def config_problems(row, path=None):
    """Differences between a run's config file and its manifest row."""
    path = path or ROOT / row['config']
    if not path.exists():
        return [f"{row['config']} is missing"]
    cfg = json.loads(path.read_text(encoding='utf-8'))
    problems = [f'{key} is {cfg.get(key)}, the manifest says {row[field]}'
                for key, field in CONFIG_FIELDS.items() if not same(cfg.get(key), row[field])]
    expected = {'run': row['run'], 'seed': row['seed'], 'init': start_model(row),
                'output_dir': f"${{CKPT_ROOT}}/{row['run']}"}
    problems += [f'{key} is {cfg.get(key)!r}, expected {value!r}'
                 for key, value in expected.items() if cfg.get(key) != value]
    return problems


def model_problems(model):
    """Problems finding a starting model offline, and checking the stage-1 model's recorded hash."""
    if model.startswith(('$', '/')):
        path = Path(os.path.expandvars(model))
        return [] if (path / 'config.json').exists() else [f'starting model {model} not found ({path})']
    from huggingface_hub import snapshot_download
    try:
        folder = Path(snapshot_download(model, local_files_only=True))
    except Exception as error:
        return [f'{model} is not in the Hugging Face cache ({type(error).__name__}); download it first']
    if model != STAGE1_REPO:
        return []
    expected = next((line.split()[0] for line in CHECKSUMS.read_text(encoding='utf-8').splitlines()
                     if line.split()[1] == STAGE1_HASH_KEY), None)
    actual = sha256(folder / 'model.safetensors')
    return [] if actual == expected else [f'{model}: sha256 {actual} differs from the recorded {expected}']


def preflight(args):
    """Checks before any GPU run: configs, software versions and starting models. Writes preflight.json."""
    import torch
    import transformers
    rows = selected(args.stage, args.phase, args.tag)
    if not rows:
        raise SystemExit('no planned runs match; run tune.py plan first')
    failures = [f"{row['run']}: {p}" for row in rows for p in config_problems(row)]
    versions = {'torch': torch.__version__, 'transformers': transformers.__version__}
    if versions != recorded_versions():
        failures.append(f'software versions {versions} differ from the recorded runs {recorded_versions()} '
                        f'({REFERENCE_RUN.relative_to(ROOT)})')
    for model in sorted({start_model(row) for row in rows}):
        failures += model_problems(model)
    report('preflight', failures, [])
    PREFLIGHT.write_text(json.dumps({
        'checked': time.strftime('%Y-%m-%d %H:%M:%S'), 'runs': [r['run'] for r in rows], 'versions': versions,
        'data_manifest_sha256': sha256(DATA_MANIFEST), 'start_models': sorted({start_model(r) for r in rows}),
    }, indent=2) + '\n', encoding='utf-8')
    print(f'preflight passed for {len(rows)} runs; recorded in {PREFLIGHT.relative_to(ROOT)}')


def stability(log_history, spike=2.0):
    """The trainer_state.json check. Failures: a non-finite loss or gradient norm.
    Warnings: after the first tenth of the run, a loss above spike x the lowest so far, or no decrease overall."""
    losses = [(entry['step'], entry['loss']) for entry in log_history if 'loss' in entry]
    grads = [entry['grad_norm'] for entry in log_history if 'grad_norm' in entry]
    failures, warnings = [], []
    if not losses:
        return ['no training loss in trainer_state.json'], []
    bad = [step for step, loss in losses if not math.isfinite(loss)]
    if bad:
        failures.append(f'non-finite training loss at step {bad[0]}')
    if any(not math.isfinite(g) for g in grads):
        failures.append('non-finite gradient norm')
    lowest = math.inf
    for i, (step, loss) in enumerate(losses):
        if not math.isfinite(loss):
            continue
        if i >= len(losses) // 10 and loss > spike * lowest:
            warnings.append(f'loss spike at step {step}: {loss:.3f}, over {spike}x the lowest so far ({lowest:.3f})')
            break
        lowest = min(lowest, loss)
    if math.isfinite(losses[-1][1]) and losses[-1][1] >= losses[0][1]:
        warnings.append(f'training loss did not decrease ({losses[0][1]:.3f} to {losses[-1][1]:.3f})')
    return failures, warnings


def count_lines(path):
    with open(path, encoding='utf-8') as f:
        return sum(1 for _ in f)


def run_problems(row, ckpt_root, results=RUNS):
    """Checks on one finished run against its config. Returns (failures, warnings)."""
    failures = []
    out = results / row['run']
    if not (out / 'dev_tune.scores.json').exists():
        failures.append('dev_tune.scores.json is missing')
    pred = out / 'dev_tune.pred'
    if not pred.exists():
        failures.append('dev_tune.pred is missing')
    elif count_lines(pred) != SPLIT_SIZES['dev_tune']:
        failures.append(f"dev_tune.pred has {count_lines(pred)} lines, expected {SPLIT_SIZES['dev_tune']}")

    info_path = ckpt_root / row['run'] / 'run_info.json'
    if not info_path.exists():
        return failures + [f'{info_path} is missing'], []
    info = json.loads(info_path.read_text(encoding='utf-8'))
    if 'finished' not in info:
        failures.append('training did not finish (run_info.json has no "finished" time)')
    config = json.loads((ROOT / row['config']).read_text(encoding='utf-8'))
    trained = info.get('config', {})
    failures += [f'trained with {key}={trained.get(key)!r}, the config says {value!r}'
                 for key, value in config.items() if key != 'output_dir' and trained.get(key) != value]
    pairs = json.loads(DATA_MANIFEST.read_text(encoding='utf-8'))[STAGES[row['stage']]['data']]['pairs']
    if info.get('training_pairs') != pairs:
        failures.append(f"trained on {info.get('training_pairs')} pairs, the data manifest has {pairs}")
    if info.get('data_manifest_sha256') != sha256(DATA_MANIFEST):
        failures.append('trained on a different build of data/curriculum (data manifest hash differs)')
    if info.get('versions') != recorded_versions():
        failures.append(f"trained with {info.get('versions')}, the recorded runs used {recorded_versions()}")

    state = ckpt_root / row['run'] / 'trainer_state.json'
    if not state.exists():
        return failures + ['trainer_state.json is missing'], []
    stable_failures, warnings = stability(json.loads(state.read_text(encoding='utf-8'))['log_history'])
    return failures + stable_failures, warnings


def report(name, failures, warnings):
    for w in warnings:
        print(f'  warning: {w}')
    for f in failures:
        print(f'  FAIL: {f}')
    if failures:
        sys.exit(f'{name}: {len(failures)} check(s) failed')


def ckpt_root():
    if not os.environ.get('CKPT_ROOT'):
        raise SystemExit('CKPT_ROOT is not set; run this on the cluster, where cluster/job.sbatch sets it')
    return Path(os.environ['CKPT_ROOT'])


def verify(args):
    """Check finished runs against their configs; writes runs/<run>/verified.json for each."""
    rows = selected(args.stage, None, args.tag, args.run)
    root, failed = ckpt_root(), []
    for row in rows:
        failures, warnings = run_problems(row, root)
        (RUNS / row['run']).mkdir(parents=True, exist_ok=True)
        (RUNS / row['run'] / 'verified.json').write_text(json.dumps({
            'checked': time.strftime('%Y-%m-%d %H:%M:%S'), 'failures': failures, 'warnings': warnings,
        }, indent=2) + '\n', encoding='utf-8')
        print(f"{row['run']}: {'FAILED' if failures else 'ok'}")
        for w in warnings:
            print(f'  warning: {w}')
        for f in failures:
            print(f'  FAIL: {f}')
        if failures:
            failed.append(row['run'])
    if failed:
        sys.exit(f'verify: {len(failed)} run(s) failed: {", ".join(failed)}')


def final_dir(stage, tag):
    """Where the final dev_test scores go; untagged stage 2 keeps the original final-dev_test/."""
    if tag:
        return RUNS / f'final-dev_test-{tag}'
    return RUNS / ('final-dev_test' if stage == 2 else 'final-dev_test-s1')


def test(args):
    """Score the final configuration's seed runs on dev_test, once each, only if they passed verify."""
    lr = args.lr or STAGES[args.stage]['default']['lr']
    rows = [r for r in selected(args.stage, None, args.tag)
            if r['param'] == 'base' and r['lr'] == lr and r['seed'] in SEEDS]
    if not rows:
        raise SystemExit(f'no seed runs at lr {lr}; the final configuration is the default settings at a learning rate')
    for row in sorted(rows, key=lambda r: SEEDS.index(r['seed'])):
        out = final_dir(args.stage, args.tag) / f"seed{row['seed']}"
        if (out / 'dev_test.scores.json').exists():
            print(f"skip {row['run']}: {out.relative_to(ROOT)} already scored; dev_test is scored once")
            continue
        verified = RUNS / row['run'] / 'verified.json'
        if not verified.exists() or json.loads(verified.read_text(encoding='utf-8'))['failures']:
            print(f"skip {row['run']}: it has not passed tune.py verify")
            continue
        cmd = ['sbatch', '-J', f"test-{row['run']}", 'cluster/job.sbatch', 't5_tuning/run_test.sh',
               row['run'], str(out.relative_to(ROOT))]
        if args.run:
            subprocess.run(cmd, check=True, cwd=ROOT)
        else:
            print(' '.join(cmd))


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

    p = sub.add_parser('preflight', help='check configs, software versions and starting models before GPU runs')
    p.add_argument('--stage', type=int, choices=[1, 2])
    p.add_argument('--phase', type=int, choices=[1, 2])
    p.add_argument('--tag')
    p.set_defaults(func=preflight)

    p = sub.add_parser('verify', help='check finished runs against their configs (on the cluster)')
    p.add_argument('--run', action='append', help='a run name; repeat for several (default: all matching runs)')
    p.add_argument('--stage', type=int, choices=[1, 2])
    p.add_argument('--tag')
    p.set_defaults(func=verify)

    p = sub.add_parser('test', help='score the final configuration on dev_test, once per seed')
    p.add_argument('--stage', type=int, choices=[1, 2], default=2)
    p.add_argument('--tag')
    p.add_argument('--lr', help='the learning rate of the final configuration (default: the stage default)')
    p.add_argument('--run', action='store_true', help='submit; without this, only print the commands')
    p.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
