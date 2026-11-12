# GRECO + ESC system combination

This combines our three systems the way the GRECO paper's strongest setup does (Qorib & Ng, 2023): GRECO's quality score plus ESC's edit scores. It runs with and without ESC, and with and without voting. The three systems are the fine-tuned BART and T5 (`s1s2`) and GECToR. Combinations are chosen on BEA-2019 dev and tested on CoNLL-2014 and BEA-2019 test.

**Status: complete.** Every dev and test run is done and scored, including all 35 BEA-2019 outputs on Codabench.

## Results

**Metrics.** On BEA-2019 dev and test we report **ERRANT F0.5**, the BEA-2019 metric; BEA-2019 test is scored by its Codabench task. On CoNLL-2014 we report **M2 F0.5**, its official metric.

**The three models to report:**

| Model | What it is | BEA-2019 dev (ERRANT F0.5) | BEA-2019 test (ERRANT F0.5) | CoNLL-2014 test (M2 F0.5) |
|---|---|---|---|---|
| **Best dev model** (main result) | GRECO + ESC, k = 3, `rank` vote, α 0.4 | **60.35** | 77.24 | 66.14 |
| **Best on BEA-2019 test** | GRECO + ESC, k = 2, `system` vote, α 0.1 | 60.22 | **77.45** | 66.13 |
| **Best on CoNLL-2014 test** | GRECO alone (no ESC), k = 2, `rank` vote, α 0.2 | 58.59 | 74.83 | **67.98** |
| Best single system | GECToR | 54.77 | 71.57 | 64.03 |

- **Main result: GRECO + ESC, k = 3, `rank` vote 0.4.** It is the highest of the 28 combinations on BEA-2019 dev (60.35), chosen before any test score was seen. It scores **77.24 on BEA-2019 test** and **66.14 on CoNLL-2014**, 5.7 and 2.1 points above GECToR. These are the held-out numbers.
- **Best on BEA-2019 test: GRECO + ESC, k = 2, `system` vote 0.1, at 77.45.** It is 0.21 above the main result (see below for why).
- **Best on CoNLL-2014 test: GRECO alone, without ESC, k = 2, `rank` vote 0.2, at 67.98 M2.** ESC, trained on BEA-2019 dev, lowers CoNLL-2014's M2 by trading recall for precision. The GRECO paper likewise uses ESC only for BEA-2019.
- The two "best on test" rows were picked by looking at test scores. They show the best each setup reaches, not held-out estimates.

**What the results show:**

- **ESC is what makes the combination work on BEA-2019.** Every GRECO + ESC run beats GRECO alone, and the gain grows with k: +0.3 to +1.0 at k = 1, and up to +6.6 at k = 3 and +9.5 at k = 5 on BEA-2019 test.
- **GRECO alone gets worse as k grows**, from 76.82 at k = 1 to at most 73.67 at k = 5 on BEA-2019 test. Without `rank` voting its k = 5 runs fall below GECToR (67.26–69.58 against 71.57). With ESC, every k stays between 76.73 and 77.45.
- **Every GRECO + ESC run beats every single system** on dev, on BEA-2019 test (by more than 5 points) and on CoNLL-2014.

**Why the best dev model is not the best on BEA-2019 test.** The two runs are 0.21 apart on test and were 0.13 apart the other way on dev, a difference we attribute to the two sets' different sentences and references. The best dev model stays the main result because it was chosen before any test score was seen. With ESC, using more than each system's 1-best does no harm (every k scores 76.73–77.45 on test); without ESC, larger k lowers the score. See [`results/bea19.md`](results/bea19.md#why-k--2-beats-k--3-on-bea-2019-test) for the full discussion.

**Full results, every run, in [`results/`](results/):**

- [`results/dev.md`](results/dev.md): BEA-2019 dev (ERRANT), where the best dev model is chosen. It also compares every run on dev and both test sets, and explains why they can differ.
- [`results/bea19.md`](results/bea19.md): BEA-2019 test, every Codabench submission.
- [`results/conll14.md`](results/conll14.md): CoNLL-2014 test, M2 (official) and ERRANT, with P / R.
- `results/{dev,conll14}.csv`: the raw scores, one row per output, copied from `runs/` (which is not tracked).

To reproduce every number, follow [section 3](#3-how-to-run) from the top.

## 1. The method

**ESC** (Qorib, Na & Ng, 2022) is a logistic regression over edits. For each edit any system proposes, the features are which system proposed it and its ERRANT type, one binary feature per (system, type) pair. ESC is trained on a development set whose gold edits label each proposed edit right or wrong. Its output is a probability that the edit is correct.

**GRECO + ESC** runs GRECO's beam search over the pooled edits, as in the [GRECO dev30 pilot](../greco_setup/README.md#bartt5gector-dev30-pilot), with one change: each candidate sentence is ranked by

    (1 − r) · log Q  +  α · log V  +  r · mean over the edits considered of log p (edit applied) or log(1 − p) (edit skipped)

- Q is GRECO's quality score of the sentence.
- V is the vote: the average share of inputs that propose the sentence's edits.
- p is ESC's probability for each edit.
- r is `--score_ratio`, and α is `--vote_coef`.

This is upstream's `run_combination.py --edit_scores <ESC scores> --score_ratio 0.7`, the paper's command for BEA-2019. Upstream ESC writes the scores with `run.py --score`, which its README added for GRECO.

GRECO runs through `combine.py`, a thin wrapper around upstream's `run_combination.py`. It keeps upstream's beam search, scorer and ESC term, and adds `--vote_mode`, which sets how V counts an edit's support:

| Mode | Support of an edit | Voters (the unchanged sentence's share) |
|---|---|---|
| `hypothesis` | share of the 1 + 2k inputs proposing it; upstream's own count | 1 + 2k |
| `system` | share of systems with at least one input proposing it; the paper's count(e) / c | c = 3 |
| `rank` | Σ over systems of 1 / (best rank proposing it), ÷ c | c = 3 |

These are the same definitions as in [`greco_setup/combine.py`](../greco_setup/README.md#voting). At k = 1 each system has one input, so the three modes give the same output and only `hypothesis` is run.

**What this pipeline does:**

| Step | What | Where |
|---|---|---|
| 1 | BART and T5 `s1s2` 5-best (beam 10), GECToR 1-best, on `dev_tune` and `dev_test` | `runs/<split>/pool/` |
| 2 | One ERRANT M2 file per ranked output: `gector-1`, `bart-1..5`, `t5-1..5` | `runs/<split>/m2/` |
| 3 | For each k: train ESC on `dev_tune` with the inputs GRECO combines at that k; score the `dev_test` edits; ESC's own output too | `runs/esc/k<k>/`, `runs/dev_test/k<k>/` |
| 4 | For each k: GRECO with and without ESC, each with α = 0 and with each vote mode at its tuned α (k = 1: `hypothesis` only) | `runs/dev_test/k<k>/<run>/pred.txt` |
| 5 | ERRANT and M2 scores on `dev_test`, and on the full dev set (steps 3–4 also run on `dev_tune`, with ESC trained on `dev_test`) | `runs/dev_test/results.{csv,md}`, `runs/dev/results.{csv,md}` |

At k, GRECO and ESC both read the 1 + 2k inputs `gector-1, bart-1, t5-1, …, bart-k, t5-k`, the same order as the GRECO dev30 pilot. GRECO and ESC read the same M2 files, so ESC has a score for every edit GRECO considers. The job stops if any edit lacks one.

**Splits:** ESC learns its weights on `dev_tune`, which has gold edits. The main results are on `dev_test`. Nothing is tuned on `dev_test`.

**Full dev set (cross-fitting).** The full dev set (`data/dev.src`, 4,384 sentences) is `dev_tune` + `dev_test`, and `data/curriculum/<half>.ids` gives each line's place in it. To score it without ESC seeing a sentence's gold edits, the job also runs every combination on `dev_tune` with ESC trained on `dev_test` (`runs/esc/k<k>/for-dev_tune/`). `evaluate.py dev` then merges each output's two halves back into `dev.src` order and scores them against a reference rebuilt the same way from the halves' M2 files, because `data/dev.m2` is a Git LFS pointer on the cluster. GRECO and ESC handle each sentence on its own, so the merged output is what one run on the full set would give, apart from ESC's training data. The α values are tuned on the full dev set itself (`main`'s `dev.errant.beam`), so they are in-sample on both halves: compare full-dev results with that table, and treat neither dev number as a held-out estimate. Only the test sets are held out.

*Why the split is needed.* ESC is trained on gold corrections: it sees our systems' edits on dev sentences and learns which of them the gold accepts. The dev set is the only labelled data we have system outputs for. An ESC trained on all of dev and scored on all of dev would be marked on sentences whose answers it has already seen, and its score would be inflated. So the full dev set is scored in two pieces:

| Half | Sentences | ESC trained on |
|---|---|---|
| `dev_test` | 2,384 | `dev_tune` |
| `dev_tune` | 2,000 | `dev_test` |

Every sentence is then scored by an ESC that never saw its gold edits. That is why the job log shows each run twice, once per half. It is one experiment on 4,384 sentences, done in two pieces. GRECO is not trained here, so only ESC needs this. The test sets need no split, because ESC never sees them: there it is trained once on the whole dev set.

*How the halves become one score.* The two scores are not averaged. The two halves' predictions are put back into `dev.src` order using the `.ids` files, giving one 4,384-line output. ERRANT then scores that output once against the full reference. It counts true positives, false positives and false negatives over all sentences and computes one P, R and F0.5 from those totals. The result is the same as if one system had produced all 4,384 outputs, and each half counts in proportion to its number of edits. As a check, subtracting the `dev_tune` counts from the full-dev counts gives exactly the `dev_test` score (58.2 at k = 1).

### What ESC does, and which ESC each run uses

**Purpose.** GRECO judges a whole candidate sentence: how good does it read? It cannot know which of our systems is reliable for which kind of error. ESC adds that, one edit at a time. It learns how far to trust each input for each error type, and gives every proposed edit a probability of being correct. GRECO + ESC ranks candidate sentences by both: GRECO's sentence score, and ESC's probabilities for the edits the sentence applies or skips.

**What ESC learns.** One weight for each (input, ERRANT error type) pair, by logistic regression. Each ranked output is its own input, so `bart-1` and `bart-4` are weighed separately. ESC therefore learns things like:

- GECToR's spelling fixes (`R:SPELL`) are usually right.
- An article insertion (`M:DET`) that only `t5-3` proposes is usually wrong.

For a new edit, ESC adds up the weights of the inputs that proposed it, for that edit's type, and turns the sum into a probability. It does not read the words or the rest of the sentence: it knows who proposed an edit and what kind of edit it is, not whether the edit fits. That is GRECO's job.

**How ESC is trained.** On our systems' outputs, not on the data they were trained on. For each sentence of a dev split, every edit any input proposes becomes one example, labelled right if it is in the gold M2 and wrong if not. The log shows these counts as `Label distribution: [wrong, right]`. ESC holds back a fifth of the examples to choose the number of epochs (`Best checkpoint at epoch N`), then retrains on all of them for that many epochs.

**Why dev and not the training set.** BART and T5 were fine-tuned on the training data, and GECToR's released model was trained on W&I+LOCNESS train. On those sentences the systems are unrealistically accurate, so an ESC trained there would trust them too much. ESC has to learn from sentences the systems have not seen, as on the test sets. The ESC paper trains on BEA-2019 dev for the same reason.

**One ESC per k.** At each k, GRECO combines a different set of 1 + 2k inputs, so ESC has 162, 270, 378 or 594 features (inputs × 54 error types). An edit that `bart-2` to `bart-5` did not propose means something at k = 5 (the lower beams disagreed) and nothing at k = 1 (they are not inputs). A model trained for one k would be wrong for another, so each k has its own.

**Never scored on its own training sentences.** ESC learns from gold edits, so scoring a sentence with an ESC that was trained on it would be scoring with the answers. Each set of results therefore uses an ESC trained on other sentences:

| Results on | ESC trained on | ESC model | Trained by |
|---|---|---|---|
| `dev_test` (2,384) | `dev_tune` (2,000) | `runs/esc/k<k>/model` | `run.sh` |
| `dev_tune` half of the full dev set | `dev_test` (2,384) | `runs/esc/k<k>/for-dev_tune/model` | `run.sh` |
| Full dev set (4,384) | each half by the model trained on the other half (cross-fitting; the two rows above) | | |
| CoNLL-2014 and BEA-2019 test | the full dev set (4,384) | `runs/esc/final/k<k>/model` | `test.sh` |

The test sets can use all of dev because none of their sentences is in it; the full-dev model is the one the paper's setup corresponds to (ESC trained on BEA-2019 dev). That makes 4 × 3 = 12 small ESC models in all, each trained in about a minute.

## 2. Settings

All settings are in `settings.py`, and the scripts read them from there.

| Setting | Value | Source |
|---|---|---|
| Systems | `CS4248-GEC/gec-bart-base-s1s2`, `CS4248-GEC/gec-t5-base-s1s2`, GECToR | Our final models |
| BART/T5 decoding | **Plain** beam search, 10 beams, 10 returned; re-ranked by length-normalised teacher-forced log-probability, duplicates removed, top 5 kept | `bart/generate_nbest.py` default `--mode beam`; `bart/export_candidates.py --k 5` |
| GECToR | Grammarly's `roberta_1_gectorv2.th`, no further training; keep-confidence 0.2, min error probability 0.5, up to 5 iterations | `gector/predict.py` defaults |
| k (hypotheses per BART/T5) | 1, 2, 3, 5 | As in the GRECO voting tuning |
| α (`vote_coef`), `hypothesis` | k1 0.1, k2 0.1, k3 0.1, k5 0.1 | Best α per mode on the full dev set (BEA-2019 dev), on plain beam search lists: `outputs/greco/tuning/dev.errant.beam/best.json` |
| α, `system` | k2 0.1, k3 0.1, k5 0.1 | same file |
| α, `rank` | k2 0.2, k3 0.4, k5 0.6 | same file |
| Control α | 0 | No voting |
| r (`score_ratio`) | 0.7 | Paper (BEA-2019 command) |
| Beam size, batch size | 16, 16 | Paper; with the attention-mask fix the batch size does not change scores |
| ESC training | upstream defaults: SGD lr 0.1, batch 16, epochs chosen on a 1/5 split (up to 100), then retrained on all of `dev_tune` | ESC paper |
| GRECO checkpoint | `models/greco/checkpoint.bin`, the released model | `greco_setup/setup.sh` |

**Where the α values come from.** The GRECO voting tuning ([`greco_setup/`](../greco_setup/README.md#tuning-α-on-dev_tune)) tried α ∈ {0, 0.1, …, 0.9} for three ways of counting votes:

- **hypothesis:** the share of all hypotheses proposing the edit.
- **system:** the share of systems proposing it.
- **rank:** weighted by the best rank that proposes it.

Upstream `run_combination.py` computes only "hypothesis"; `combine.py` adds "system" and "rank" (section 1), so all three are run with their own α. The α come from the tuning on **plain** beam search lists (`cluster/nbest_beam.sh`: the same models, 10 beams, 10 returned, top 5 after re-ranking, as this pipeline), on the **full dev set** (BEA-2019 dev, `dev.errant.beam`), without ESC. That is the paper's practice and keeps the test sets untouched, but it makes the α in-sample on `dev_test` and the full dev set, so dev scores with them are not held-out estimates.

**Why α was reused, not retuned with ESC.** The α come from the earlier GRECO voting tuning (plain beam search lists, full dev set). They were not tuned again with ESC switched on, and r = 0.7 was not tuned at all. There are three reasons:

- **The tuning already answers the question α is for.** α sets how much GRECO trusts agreement between inputs. It was tuned for exactly these candidate lists (same models, same decoding, same k) on the same dev set the paper tunes on. The GRECO paper likewise reports one vote coefficient and takes r = 0.7 as given rather than retuning them together.
- **With ESC on, α barely matters.** At r = 0.7, ESC's edit term carries 70 % of the score, and the vote is a small extra term. On the full dev set, every GRECO + ESC run, whatever its α and vote mode, falls between 59.59 and 60.35: within 0.8 points. Without ESC, the same α choices move GRECO by up to 3.7 points (k = 5). So a retuned α could gain only a fraction of a point.
- **Retuning with ESC would be costly to do cleanly.** It would mean 10 α values × 3 vote modes × 4 k, each run on both dev halves for cross-fitting: about 240 GRECO runs of 3–8 minutes each, or roughly 20 GPU hours, for the small gain above. Tuning α on the same dev sentences ESC is trained on would also need nested cross-fitting to stay fair. Given the time left in the project, we kept the existing α and record this as a limitation (section 4).

**Earlier runs used the `dev_tune`-only α** (`dev_tune.errant.beam`: k1 0.4, rank 0.4 at every k; hypothesis and system as now). They kept `dev_test` held out. The runs whose α changed (k1 voting, k2 `rank` 0.2, k5 `rank` 0.6) are new runs with new names. Job 1101538 made their dev outputs with the new α; the old outputs stay on disk. The changes were small (−0.25 to +0.48, see [`results/dev.md`](results/dev.md)), and the best dev model (k3, `rank` 0.4) has the same α under both tunings.

The difflib-edit tuning in `outputs/greco/tuning/dev_tune/` is an earlier run, replaced by the ERRANT-edit tuning used here.

## 3. How to run

Run everything from the repository root (`~/CS4248_GEC` on the cluster). Jobs go through `cluster/job.sbatch`, which runs offline on an H100.

### Reproduce every number, in order

Every setting (systems, k, α, r, beam size) is in `settings.py`, so these commands need no arguments to reproduce the results above. Each job skips outputs that are already complete, so a job cut off by its time limit can be resubmitted as is.

| Step | Where | Command | Time | Produces |
|---|---|---|---|---|
| 1. One-time setup | compute node | [One-time setup](#one-time-setup) below | about 1 h | environments, models, ESC |
| 2. Dev runs | `xlogin` | `sbatch -J greco-esc --time=10:00:00 cluster/job.sbatch greco_esc/run.sh` | about 7 h | `runs/dev/results.{csv,md}`: the dev table, and the best dev model |
| 3. CoNLL-2014 gold | compute node | `git lfs pull --include data/test.conll14.m2` | 1 min | the gold M2 that step 4 scores against |
| 4. Test runs | `xlogin`, after step 2 | `sbatch -J greco-test --time=12:00:00 cluster/job.sbatch greco_esc/test.sh` | about 8 h | `runs/conll14/results.{csv,md}`, and the BEA-2019 outputs |
| 5. BEA-2019 zips | `xlogin`, after step 4 | `python3 -m greco_esc.codabench` | seconds | `codabench/bea19_*.zip`, one per output |
| 6. BEA-2019 scores | browser | upload each zip under **My Submissions** on the BEA-2019 Codabench task | a few minutes each | the scores in `results/bea19.md` |
| 7. Copy to your laptop | laptop | the `rsync` command under [The job](#the-job) | minutes | `runs/` locally; `results/*.csv` are copies of its `results.csv` files |

The α are not tuned here: they come from the GRECO voting tuning (`outputs/greco/tuning/dev.errant.beam/best.json`, section 2). ESC is trained inside steps 2 and 4.

**What to expect when reproducing.**

- **Deterministic by design.** Beam search decoding, GECToR and GRECO's scoring have no sampling, and upstream ESC fixes its seed (`--seed 0`) for its train/validation split and training. Floating-point differences between GPUs can still move a score by a few hundredths.
- **Needs access to:**
  - the public `CS4248-GEC` Hugging Face repositories, for the BART and T5 models;
  - the SoC cluster, or an equivalent Slurm GPU node: `cluster/job.sbatch` and the `HF_HOME` path are written for SoC;
  - a Codabench account on the BEA-2019 task, for BEA-2019 test scores.
- **Pinned dependencies.** Upstream GRECO is at `8e509aa` and ESC at `cdefd3b`; the GRECO checkpoint is the released one. The α come from `outputs/greco/tuning/dev.errant.beam/best.json` and are copied into `settings.py`.
- **Not yet verified from scratch.** The whole pipeline has not been rerun end to end in a clean checkout. The α rerun (job 1101538) reran only the six changed runs, and the other outputs were reused.

### One-time setup

If you have already set up GRECO for [`greco_setup/`](../greco_setup/README.md)'s tuning, steps 1–4 are done; only step 5 is new.

Do these on a compute node: `salloc -t 60`, then `srun --jobid=<id> --pty bash`. Check that `hostname` is not `xlogin`. Then run:

```bash
cd ~/CS4248_GEC
source ~/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/mnt/scratch/${USER:0:1}/$USER/cs4248-gec/hf_cache

# 1. gec-train: ERRANT for the M2 files and scoring
conda activate gec-train
pip install errant && python -m spacy download en_core_web_sm

# 2. GECToR (Python 3.11 or newer)
PYTHON=python3.12 bash gector/setup.sh

# 3. greco environment: upstream GRECO, its checkpoint, and two version fixes
conda create -n greco -c conda-forge --override-channels python=3.10 -y
conda activate greco
bash greco_setup/setup.sh
pip install "pyarrow==14.0.2"
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 4. Models in the cache (public repositories; no login required)
conda activate gec-train
hf download CS4248-GEC/gec-bart-base-s1s2
hf download CS4248-GEC/gec-t5-base-s1s2
hf download microsoft/deberta-v3-large
gector/.venv/bin/python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('roberta-base'); AutoTokenizer.from_pretrained('roberta-base')"

# 5. ESC, in the greco environment
conda activate greco
bash greco_esc/setup.sh                        # prints: esc ok: scikit-learn 1.3.2 torch 2.1.2...
exit; exit
```

### The job

From `xlogin`:

```bash
sbatch -J greco-esc --time=10:00:00 cluster/job.sbatch greco_esc/run.sh
tail -f logs/slurm/greco-esc-<jobid>.out       # "== <time> <step>" marks each step
```

The job skips any step whose output is already complete. If it fails or runs out of time, fix the cause and submit the same command again. Don't `git pull` while it runs, because bash reads the script as it goes.

**Expected time:** about 7 hours from scratch. If the `dev_test` runs already finished, resubmitting runs only the `dev_tune` half (ESC trained on `dev_test`, 4 ESC models and 28 GRECO runs on 2,000 sentences) and the scoring: about 3 hours. The command above allows 10 hours.

- Candidates: about 30 min for both splits.
- GRECO: 28 runs of about 3–8 min each, longer at larger k (16 `hypothesis` and no-vote runs, 12 `system` and `rank` runs).
- Scoring: about 25 min.

**Check it when it finishes:**

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed        # COMPLETED 0:0
wc -l greco_esc/runs/dev_tune/pool/*.tgt     # bart, t5 10000; gector 2000
wc -l greco_esc/runs/dev_test/pool/*.tgt     # bart, t5 11920; gector 2384
ls greco_esc/runs/dev_test/k*/*/pred.txt | wc -l               # 32: 28 GRECO runs and 4 ESC
ls greco_esc/runs/dev_tune/k*/*/pred.txt | wc -l               # 32 as well
cat greco_esc/runs/dev/results.md                              # full dev set
grep -h "Best checkpoint at epoch" logs/slurm/greco-esc-<jobid>.out   # ESC's chosen epoch per k; should not be 0
cat greco_esc/runs/dev_test/results.md
```

`results.md` has two tables. The first has one line per k: ERRANT F0.5 (P / R) for GRECO alone, GRECO with `hypothesis` voting, GRECO + ESC, GRECO + ESC with voting, and ESC alone, plus the three single systems. The second compares the three vote modes at their tuned α, without and with ESC. `results.csv` adds the M2 scores. To rescore without rerunning, use `conda activate gec-train && python -m greco_esc.evaluate` on a compute node.

**Bring the results to your laptop.** The runs folder is small apart from the n-best JSONL files:

```bash
rsync -av -e "ssh -J <user>@stujump.comp.nus.edu.sg" --exclude '*.nbest.jsonl' \
  <user>@xlogin.comp.nus.edu.sg:~/CS4248_GEC/greco_esc/runs/ greco_esc/runs/
```

### Test sets

`test.sh` runs every combination on CoNLL-2014 (1,312 sentences) and BEA-2019 (4,477): the same 28 GRECO runs and ESC alone at each k as on dev, plus the single systems, in the same tables. Nothing is tuned or trained on a test set. ESC is trained once per k on the full dev set (`dev_tune` + `dev_test`, 4,384 sentences), as the paper trains ESC on BEA-2019 dev.

**The best dev model is chosen on dev.** As in the GRECO paper, it is the run with the highest ERRANT F0.5 on BEA-2019 dev (our full dev set). It is GRECO + ESC, k = 3, `rank` vote 0.4 (`settings.BEST_DEV_RUN`, 60.35). Its test scores are the held-out result. The best model on each test set is also reported, labelled as picked on that test set. The paper used ESC only on BEA-2019 and reports GRECO with voting alone on CoNLL-2014; on dev, ESC helps ERRANT F0.5 (best 60.35 with ESC vs 59.85 without).

Run it after `run.sh` has finished (it trains ESC on `run.sh`'s dev candidates). About 8 hours for both sets, mostly the 56 GRECO runs:

```bash
sbatch -J greco-test --time=12:00:00 cluster/job.sbatch greco_esc/test.sh            # both test sets
sbatch -J greco-test --time=6:00:00 cluster/job.sbatch greco_esc/test.sh conll14     # one only
```

**Scoring.**

- **CoNLL-2014** is scored into `runs/conll14/results.{csv,md}` with ERRANT and the M2 scorer; M2 is its official metric. `data/test.conll14.m2` is in Git LFS, so pull it first (`conda install -c conda-forge git-lfs && git lfs install && git lfs pull --include data/test.conll14.m2`). Without it, the job still writes the outputs and lists them. Rescore later with `python -m greco_esc.evaluate conll14`.
- **BEA-2019** has no public gold. `python3 -m greco_esc.codabench` zips every finished output (`runs/bea19/k<k>/<run>/pred.txt` and the single systems) as `codabench/bea19_k<k>_<run>.zip`, one 4,477-line `.txt` each. Submit them to the BEA-2019 Codabench task. Rerunning it packs only outputs without a zip. Submitted scores are in [`results/bea19.md`](results/bea19.md).

**Fine-tuned GRECO.** `GRECO_CKPT` sets another GRECO checkpoint, and `GRECO_TAG` names its outputs (`runs/<test set>/<tag>/k<k>/<run>`; default `released`). Choose the checkpoint on dev first: run `run.sh` with it, then test the one that does better on the full dev set.

### Tests

```bash
python3 -m unittest greco_esc.test_greco_esc
```

The end-to-end test runs only with `external/esc` and `external/greco` in place and torch, scikit-learn and transformers installed. It trains upstream ESC on a toy pool and runs upstream GRECO's beam search with ESC's scores, using a stand-in for DeBERTa. It checks two things:

- ESC scores the right edit above 0.5 and the wrong one below.
- GRECO finds a score for every edit and picks the right sentence.

## 4. Differences from the paper and known limits

- **α was tuned without ESC,** on plain beam search lists, with `greco_setup/`'s own combiner rather than upstream's beam search. It was not retuned with ESC, so the best α with ESC may differ. r = 0.7 is the paper's value and was not tuned either.
- **Compare with the plain beam tuning, not the diverse one.** The `dev_test.errant` tuning results are on diverse beam search lists; the plain beam tuning (`dev.errant.beam`, whose α are used here) uses the same candidates as this pipeline.
- **On dev, ESC is trained on half of dev.** For dev scores, each ESC learns from one half (2,000 or 2,384 sentences) so it never scores its own training sentences. For the test sets, ESC learns from all 4,384 sentences of BEA-2019 dev, as in the paper. With 11 inputs at k = 5, ESC has 11 × (number of edit types) features.
- **ESC is fitted to BEA-2019.** Its labels are BEA-2019 dev's gold edits, extracted by ERRANT. It helps on BEA-2019 test (ERRANT) but lowers CoNLL-2014's official M2 at k ≤ 3 ([`results/conll14.md`](results/conll14.md)). No CoNLL-style ESC was trained.
- **GRECO's released checkpoint is used**, not one fine-tuned on our data.
- **Each ranked hypothesis is one ESC "system".** `bart-2` gets its own weights, separate from `bart-1`. That is how upstream ESC treats input files. The paper combined 1-best outputs only.
- **Upstream ESC quirk.** ESC retrains on all the data for as many epochs as the best validation epoch. Epochs are counted from 0, so if the best is epoch 0, the final model is untrained. The log line `Best checkpoint at epoch N` shows the chosen epoch. None of the dev runs chose epoch 0:

  | ESC trained on | k = 1 | k = 2 | k = 3 | k = 5 | Slurm log |
  |---|---|---|---|---|---|
  | `dev_tune` (scores `dev_test`) | 87 | 90 | 84 | 48 | `greco-esc-1098818.out` |
  | `dev_test` (scores `dev_tune`) | 47 | 80 | 70 | 47 | `greco-esc-1099227.out` |
- **Clipping.** ESC probabilities of exactly 0 or 1 are clipped to [10⁻⁶, 1 − 10⁻⁶], because GRECO takes their logarithm (`prepare.py scores`).

## 5. Code

| File | What it does |
|---|---|
| `settings.py` | Every setting; `python3 -m greco_esc.settings runs` lists the GRECO runs for `run.sh` |
| `setup.sh` | Clones ESC at `cdefd3b` into `external/esc` and installs scikit-learn |
| `run.sh` | The cluster job: candidates, M2 files, ESC, GRECO, scores |
| `combine.py` | Upstream `run_combination.py` with `--vote_mode hypothesis/system/rank` |
| `test.sh` | Every combination on CoNLL-2014 and BEA-2019, with ESC trained on the full dev set |
| `prepare.py` | `m2 <split>`: text and M2 per ranked output; `esc`: ESC folders per k; `scores <k>`: checks and clips ESC's scores |
| `codabench.py` | Zips each finished BEA-2019 output for Codabench |
| `evaluate.py` | Scores every output on `dev_test` (default), `dev_tune`, `dev` (both halves merged), `conll14` or `bea19`; writes `runs/<split>/results.{csv,md}` |
| `test_greco_esc.py` | Tests, including the vote modes on upstream's dataset and the end-to-end ESC → GRECO test |

## Sources

- Qorib & Ng (2023), [System Combination via Quality Estimation for Grammatical Error Correction](https://aclanthology.org/2023.emnlp-main.785/). Code: [nusnlp/greco](https://github.com/nusnlp/greco) at `8e509aa`.
- Qorib, Na & Ng (2022), [Frustratingly Easy System Combination for Grammatical Error Correction](https://aclanthology.org/2022.naacl-main.143/). Code: [nusnlp/esc](https://github.com/nusnlp/esc) at `cdefd3b`.

Both upstream repositories are GPL-3.0 and are cloned into the git-ignored `external/`, not copied into this repository.
