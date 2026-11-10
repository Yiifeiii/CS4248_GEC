"""Frozen BART candidate-quality ablations on dev_tune; no T5/GRECO or training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from training.candidate_quality import pool_metrics


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('training/configs/bart-decoding-20261106.json'))
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/bart-decoding-20261106'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not 0 <= args.index < len(cfg['setups']):
        raise ValueError('Unknown decoding setup')
    setup = cfg['setups'][args.index]
    if cfg['split'] != 'dev_tune':
        raise ValueError('This exploratory sweep is restricted to dev_tune')

    import errant
    import torch
    from transformers.utils import cached_file
    from greco_setup.evaluate_errant import read_gold_edits
    from training.common import CURRICULUM, write_lines
    from training.metrics import m2_subset, score

    if not torch.cuda.is_available():
        raise ValueError('A CUDA GPU is required for the submitted decoding job')
    path = cached_file(cfg['model'], 'model.safetensors', local_files_only=True)
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != cfg['checkpoint_sha256']:
        raise ValueError('Frozen baseline checkpoint hash mismatch')

    out = args.out / setup['name']
    out.mkdir(parents=True, exist_ok=True)
    sources = (CURRICULUM / 'dev_tune.src').read_text().splitlines()
    if len(sources) != cfg['sentences']:
        raise ValueError('Unexpected dev_tune size')
    src = out / 'source.src'
    write_lines(src, sources)
    gold_path = m2_subset(CURRICULUM / 'dev_tune.m2', len(sources), out / 'gold.m2')
    gold_sources = [line[2:] for line in gold_path.read_text().splitlines() if line.startswith('S ')]
    if gold_sources != sources:
        raise ValueError('Gold/source alignment mismatch')
    # Coverage currently requires single-annotator references rather than merging alternatives.
    annotators = {line.split('|||')[-1] for line in gold_path.read_text().splitlines()
                  if line.startswith('A ')}
    if annotators != {'0'}:
        raise ValueError('Candidate coverage requires single-annotator gold')
    manifest = dict(config=cfg, setup=setup, checkpoint_sha256=digest.hexdigest(),
        sources_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        gold_sha256=hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        ranking='teacher-forced average token log probability; top k overall',
        weights_updated=False, combination=False, device=torch.cuda.get_device_name())
    save(out / 'manifest.json', manifest)

    nbest = out / 'candidates.jsonl'
    command = [sys.executable, 'bart/generate_nbest.py', '--model', cfg['model'],
        '--input', str(src), '--output', str(nbest), '--mode', setup['mode'],
        '--num_beams', str(setup['beams']), '--num_return', str(cfg['num_return']),
        '--batch_size', str(cfg['batch_size']), '--max_batch_tokens', str(cfg['max_batch_tokens']),
        '--length_penalty', str(cfg['length_penalty']),
        '--no_repeat_ngram_size', str(cfg['no_repeat_ngram_size'])]
    if setup['mode'] == 'diverse':
        command += ['--num_beam_groups', str(setup['groups']),
                    '--diversity_penalty', str(setup['penalty'])]
    started = time.perf_counter()
    reused = nbest.exists()
    if not reused:
        subprocess.run(command, check=True)
    generation_seconds = None if reused else round(time.perf_counter()-started, 2)
    entries = [json.loads(line) for line in nbest.read_text().splitlines()]
    header, rows = entries[0], entries[1:]
    if not header.get('header') or header['num_beams'] != setup['beams']:
        raise ValueError('Cached candidate settings mismatch')
    expected = dict(model=cfg['model'], mode=setup['mode'], num_return=cfg['num_return'],
                    length_penalty=cfg['length_penalty'], no_repeat_ngram_size=cfg['no_repeat_ngram_size'])
    if setup['mode'] == 'diverse':
        expected.update(num_beam_groups=setup['groups'], diversity_penalty=setup['penalty'])
    if any(header.get(key) != value for key, value in expected.items()):
        raise ValueError('Cached candidates were generated with other settings')
    if len(rows) != len(sources) or any(row['id'] != i or row['src'] != source
            for i, (row, source) in enumerate(zip(rows, sources))):
        raise ValueError('Candidate/source alignment mismatch')

    texts = [[cand['text'] for cand in row['cands']] for row in rows]
    top1 = [choices[0] for choices in texts]
    write_lines(out / 'candidate-top1.pred', top1)
    top1_scores = score(src, top1, gold_path)
    save(out / 'candidate-top1.scores.json', top1_scores)

    annotator = errant.load('en')
    gold_edits = read_gold_edits(gold_path)
    edits = []
    for i, (source, choices) in enumerate(zip(sources, texts)):
        orig = annotator.parse(source)
        per_source = []
        for text in choices:
            extracted = [] if text == source else annotator.annotate(orig, annotator.parse(text))
            per_source.append({(e.o_start, e.o_end, ' '.join(e.c_str.split())) for e in extracted})
        edits.append(per_source)
        if (i+1) % 200 == 0:
            print(f"{setup['name']}: candidate edits {i+1}/{len(sources)}", flush=True)
    metrics = [pool_metrics(texts, edits, gold_edits, k) for k in cfg['k']]
    save(out / 'candidate-pool-metrics.json', metrics)
    for k in cfg['k']:
        subprocess.run([sys.executable, 'bart/export_candidates.py', '--nbest', str(nbest),
                        '--out', str(out / f'candidates-k{k}.tgt'), '--k', str(k)], check=True)

    # Conventional one-best decoding is a separate control from likelihood-ranked n-best top1.
    if setup['mode'] == 'beam':
        subprocess.run([sys.executable, '-m', 'training.evaluate', '--model', cfg['model'],
            '--split', cfg['split'], '--beams', str(setup['beams']), '--batch-size', '8',
            '--out', str(out / 'standard-one-best')], check=True)
    summary = dict(setup=setup, sentences=len(sources), generation_wall_seconds=generation_seconds,
                   generation_reused=reused, candidate_top1=top1_scores, candidate_pools=metrics)
    save(out / 'summary.json', summary)
    print(f"BART_DECODING_COMPLETE: {setup['name']}; candidate top1 "
          f"F0.5={100*top1_scores['errant']['correction']['f0_5']:.2f}", flush=True)


if __name__ == '__main__':
    main()
