# Model references

The team's released model repositories are public, so users can download them
without a Hugging Face account or token. The canonical IDs are listed in
[`public_models.json`](public_models.json).

Model weights do not belong in Git. Store the public Hugging Face repository ID
in configs and let `transformers` download the files into its local cache.

Training configs already support environment-variable expansion for `init`, so
a local model override can use:

```json
{
  "init": "${HF_BART_MODEL}"
}
```

Then set the value at runtime:

```bash
export HF_BART_MODEL="CS4248-GEC/gec-bart-base-s1s2"
python -m training.finetune --config configs/training/my-bart.json
```

Public repositories do not need a token. Never commit a token, a local
Hugging Face cache, or the checkpoint files. See
[`docs/running.md`](../../docs/running.md#9-use-the-public-team-models).
