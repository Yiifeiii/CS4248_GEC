# Running the fine-tuning on the SoC compute cluster

These instructions follow the course's `SoC_Cluster_Guide.pdf` and the [Unofficial SoC Cluster Guide](https://github.com/isaacNingLee/Unofficial_SoC_Cluster_Guide). The official pages (hardware, job priority) need an SoC login. They take precedence wherever they differ from this guide.

**Rules:**
- Connect with a plain terminal SSH client. VSCode Remote-SSH is not allowed.
- Use `xlogin` only for `git`, downloads and `sbatch`. Install and compute on a compute node.

## One-time setup

```bash
ssh -J <user>@stujump.comp.nus.edu.sg <user>@xlogin.comp.nus.edu.sg   # from outside SoC
git clone https://github.com/Yiifeiii/CS4248_GEC.git ~/CS4248_GEC && cd ~/CS4248_GEC

salloc -t 120 && srun --pty bash      # a compute node for installing
bash cluster/setup_env.sh             # Miniconda (if missing) + conda env gec-train
exit; exit                            # back on xlogin

conda activate gec-train
bash cluster/download_models.sh       # bart-base and t5-base into the scratch HF cache
```

**Training data:**
- `data/curriculum/` holds the committed stage-2 data and dev split.
- Stage 1 (78 MB) is not in git. Build it on your own machine with `python -m training.build_data`, then copy it over:

  ```bash
  rsync -av data/curriculum/stage1.* <user>@xlogin.comp.nus.edu.sg:~/CS4248_GEC/data/curriculum/
  ```

  Then run `python -m training.check_data` in a job or on a compute node. It compares the hashes with the committed `manifest.json`.
- **Git LFS:** `data/dev.m2` and `data/test.conll14.m2` are Git LFS files. Without `git lfs` on the cluster they stay pointer files. Training and dev scoring don't need them: `data/curriculum/*.m2` is plain git. To score CoNLL-14 on the cluster, either run `conda install -c conda-forge git-lfs && git lfs install && git lfs pull`, or score the `.pred` file locally.

## Storage

| What | Where |
|---|---|
| Code, logs (`logs/slurm/`), results (`experiments/training/`) | `~/CS4248_GEC` (home, 500 GB) |
| Checkpoints (`$CKPT_ROOT`) | `/mnt/scratch/${USER:0:1}/$USER/cs4248-gec/checkpoints` (scratch, 500 GB) |
| Hugging Face cache (`$HF_HOME`) | `/mnt/scratch/${USER:0:1}/$USER/cs4248-gec/hf_cache` |

A checkpoint with optimizer state is about 1.7 GB for bart-base and 2.7 GB for t5-base. Each run keeps 2 of those plus `best/` and `final/`.

Check usage with `cat ~/.diskusage` and `cat /mnt/scratch/${USER:0:1}/$USER/.diskusage`.

## Submitting jobs

`cluster/job.sbatch` is the guide's job template with these settings:
- 1 × `h100-47` GPU on `gpu-long`
- 8 CPUs, 12 h
- `CKPT_ROOT` and `HF_HOME` on scratch, offline Hugging Face
- the log in `logs/slurm/<job name>-<id>.out`

Its first argument is the pipeline script to run.

Before the first submission, check which GPUs are free and the time limit of each partition:

```bash
gpufree                      # helper from the guide's bashrc_helpers.sh
sinfo -o "%P %l %G"          # partition, time limit, GPUs
```

Set `--time` below the partition limit. If a job hits it, submit it again: `finetune.py` resumes from the last checkpoint.

**1. Pilot (about 15 minutes each):** 200 steps with the real batch size, to measure speed and memory.

```bash
sbatch -J pilot-bart --time=00:30:00 cluster/job.sbatch cluster/pilot.sh training/configs/bart-base.stage1.json
sbatch -J pilot-t5   --time=00:30:00 cluster/job.sbatch cluster/pilot.sh training/configs/t5-base.stage1.json
```

The logs show `train_steps_per_second` and the peak GPU memory. Stage 1 is about 13,200 steps (3 epochs). Choose `--time` from those numbers, or lower `epochs` in the config.

**2. Stage 1, both models in parallel, then stage 2 once its stage 1 has succeeded:**

```bash
b1=$(sbatch --parsable -J bart-base.s1 cluster/job.sbatch cluster/train.sh training/configs/bart-base.stage1.json)
t1=$(sbatch --parsable -J t5-base.s1   cluster/job.sbatch cluster/train.sh training/configs/t5-base.stage1.json)
sbatch -J bart-base.s1s2 --dependency=afterok:$b1 cluster/job.sbatch cluster/train.sh training/configs/bart-base.stage2.json
sbatch -J t5-base.s1s2   --dependency=afterok:$t1 cluster/job.sbatch cluster/train.sh training/configs/t5-base.stage2.json
```

**3. N-best lists for GRECO (top 5 of 10 beams):**

```bash
sbatch -J nbest-bart --time=02:00:00 cluster/job.sbatch cluster/nbest.sh bart-base.s1s2
```

This writes `outputs/bart-base.s1s2/dev.nbest.jsonl` and `dev.tgt`, with 5 lines per source sentence.

**Monitoring:**

```bash
squeue --me                              # PD = pending, R = running
show_job                                 # helper from the guide: current and recent jobs
tail -f logs/slurm/bart-base.s1-<id>.out # [errant] lines show dev_tune P/R/F0.5
squeue --start -j <id>; sprio -j <id>    # why a job is still pending
```

## Results

`cluster/train.sh` writes each run's results to `experiments/training/<run>/`:
- `dev_tune` and `dev_test` predictions and their scores (`*.scores.json`, ERRANT and M2)
- the config
- `run_info.json`, `trainer_state.json` (the loss curve) and `errant_eval.json` (dev_tune F0.5 during training)
- `best_checkpoint.json`

Commit these files. The checkpoints themselves stay on scratch.
