# Manual GRECO combination

This standalone entry point accepts manual BART/T5 proposals. Neither generator needs to be installed. It pools their edits, searches compatible combinations, uses the official GRECO model to score each beam expansion, and returns the best surviving sentence.

## Try candidate generation now

From the repository root, using Python 3 (no third-party dependencies):

```bash
python3 -m greco_setup.combine --input greco_setup/examples/manual.json --dry-run
```

The first example includes these combinations:

```text
She go to school yesterday .
She went to school yesterday .
She go to the school yesterday .
She went to the school yesterday .
```

Dry-run mode enumerates combinations, sets scores and `best_sentence` to null, and does not pretend to run GRECO. It fails if more than `--max-candidates` combinations are generated (default 4096).

## Install the real scorer

Use a separate **Python 3.10** environment. The official dependency versions date from 2022 and do not support the Python 3.14 interpreter found in this checkout. A Linux machine with a CUDA GPU is recommended for DeBERTa-v3-large; CPU inference is supported but resource intensive. Installation also downloads the large GRECO checkpoint and, on first scoring, the DeBERTa backbone/tokenizer. Allow several GB of disk and memory.

```bash
python3.10 -m venv .venv-greco
source .venv-greco/bin/activate
bash greco_setup/setup.sh
python -m greco_setup.combine \
  --input greco_setup/examples/manual.json \
  --output outputs/greco/manual-results.json \
  --device cpu
```

For a compatible CUDA installation, use `--device cuda`. Match the PyTorch installation to the host CUDA environment. The setup script uses `pip install -r` on the upstream requirements, pins the upstream Git revision, and stores code and weights in ignored directories. It does not install BART/T5 or alter dataset files.

If the checkpoint download is interrupted, rerun `bash greco_setup/setup.sh` in the same environment. The download resumes from `models/greco/checkpoint.bin.partial`; keep that file. The script renames it to `checkpoint.bin` only after curl completes successfully.

The real scorer setup and pretrained inference have not been validated on the local Python 3.14 environment. Core edit combination and CLI behavior are covered by the tests below. If dependencies are already installed elsewhere, pass `--greco-repo /path/to/greco --checkpoint /path/to/checkpoint.bin`.

## Provide manual input

The input is one JSON object or a list of objects, each with `source` and a nonempty `proposals` list. Use tokenized text consistently (including punctuation). Each proposal contains either a corrected `sentence` or an explicit `edits` list, plus optional metadata such as `system` and `rank`.

```json
{
  "source": "He eat apple .",
  "proposals": [
    {"system": "bart", "rank": 1, "sentence": "He eats apple ."},
    {"system": "t5", "rank": 1, "edits": [
      {"start": 2, "end": 2, "replacement": "an"}
    ]}
  ]
}
```

Offsets refer to the **original source tokens**, starting at zero, with an exclusive end. Equal start/end means insertion; an empty replacement means deletion. Conflicting edits within one proposal are rejected. Conflicting alternatives from different proposals remain in the pool but cannot be applied together. Insertions on a replacement boundary are allowed; insertions strictly inside a replaced span are rejected.

Sentence proposals are converted with Python's token-level `SequenceMatcher`. This is a simple alignment for manual development, **not ERRANT linguistic edit extraction**. Adjacent differences may form a single edit; use explicit offsets when you need precise pooling boundaries. Later, the generators can produce this same input schema, ideally with ERRANT-derived edits.

Multiple BART/T5 hypotheses are supported by adding more proposal objects. The `system` and `rank` fields are descriptive metadata; this version does not add voting/agreement weights or GECToR-specific behavior.

## From system output files

`greco_setup/prepare.py` builds the input JSON from prediction files, so nobody writes edits by hand. Every system writes `outputs/<system>/<split>.tgt` in the shared format described in the [main README](../README.md#outputs): **k consecutive lines per source sentence**, best hypothesis first, k = 1 for GECToR and k = 5 for BART/T5. Line counts must be an exact multiple of the source file's, and hypotheses must use the source's tokenization (`apple .`, not `apple.`), otherwise every token difference becomes a spurious edit.

```bash
python3 -m greco_setup.prepare build --source outputs/dev30.src \
  --system gector=outputs/gector/dev30.tgt \
  --system bart=outputs/bart/dev30.beam.tgt \
  --system t5=outputs/t5/dev30.tgt \
  --k 3 --output outputs/greco/dev30.input.json
python -m greco_setup.combine --input outputs/greco/dev30.input.json --output outputs/greco/dev30.output.json --device cuda
python3 -m greco_setup.prepare collect --input outputs/greco/dev30.output.json --output outputs/greco/dev30.tgt
```

`--k` keeps only the top-k hypotheses per system (the k = 1, 3, 5 ablation) and leaves 1-best systems unchanged. Empty hypotheses are skipped. `collect` writes one best sentence per line, aligned with the source, ready for ERRANT or the M² scorer.

## Search and output

For each pooled edit, search considers skipping it or adding it to a compatible state. GRECO scores the resulting sentences against the original source; the highest-scoring `--beam-size` states survive (default 16). Scoring is batched with `--batch-size` (default 8) and cached per sentence. The unchanged sentence starts in the beam and is eligible for selection, but can be pruned like other states. Equal scores favor fewer edits.

The output contains `best_sentence`, pooled edits, and the surviving candidates with raw GRECO quality scores and selected edits. These scores are quality estimates, not measured F₀.₅ values. Beam search is approximate: it returns the best surviving candidate, not a guarantee of the global optimum. This is our own search wrapper over the official scoring API, not a reproduction of GRECO's full published combination configuration (which can include edit scores and voting).

```bash
python3 -m unittest discover -s tests -v
```

## Sources

- [Official GRECO code](https://github.com/nusnlp/greco), pinned to `8e509aaacf4e5ec3276d3427371471fd82a7f58b`; the adapter uses `models.GRECO.score(source, hypotheses)` and loads the released state dictionary strictly.
- [Released checkpoint](https://huggingface.co/nusnlp/GRECO).
- Qorib & Ng (2023), [System Combination via Quality Estimation for Grammatical Error Correction](https://aclanthology.org/2023.emnlp-main.785/).

Upstream GRECO code has its own GPL license in `external/greco/LICENSE.txt`.

## BART/T5/GECToR dev30 pilot

Run `python -m greco_setup.run_pilot` in the GRECO environment from the repo root.
This combines the saved BART beam (`outputs/bart/dev30.beam.tgt`) and T5
(`outputs/t5/dev30.tgt`) predictions for the first 30 W&I+LOCNESS development
sentences at k=1, 3, and 5. Add `--with-gector` to also pool GECToR's 1-best
correction (`outputs/gector/dev30.tgt`) at every k. Each run uses the same beam
size (16), batch size (8), and CPU scorer. Gold evaluation is a separate step.

Outputs go to `outputs/greco/bart-t5-dev30/` (or `bart-t5-gector-dev30/` with
`--with-gector`): `manifest.json` records settings,
and each `k1/`, `k3/`, `k5/` folder contains `input.json`, `results.json`, and
`greco.pred`. Basic punctuation/contraction spacing is normalized before the
token-level edit alignment; this is not ERRANT alignment. Raw GRECO scores are
quality estimates, not F0.5 measurements.

Score a pilot with the M² scorer under Python 3.10, passing its folder. The
folder needs the 30 gold records, which the BART/T5 pilot already has:

```bash
cp outputs/greco/bart-t5-dev30/gold.m2 outputs/greco/bart-t5-gector-dev30/
python -m greco_setup.evaluate_pilot --output-dir outputs/greco/bart-t5-gector-dev30
```

The existing `bart-t5-dev30` results were produced before the outputs were
unified. Rerunning reproduces the BART input exactly; 2 of 150 T5 lines differ
because spaCy now separates a `?"` the old normalization left joined.

Commit the runner, documentation, and pilot outputs for reproducibility.
The output directory is not ignored by Git; prediction files, input records,
full results, and the manifest can be versioned together.
