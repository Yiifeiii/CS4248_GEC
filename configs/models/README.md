# Model references

Model weights do not belong in Git. Store only public or private Hugging Face
repository IDs in configs, and authenticate separately on each machine.

Training configs already support environment-variable expansion for `init`, so
a private config can use:

```json
{
  "init": "${HF_BART_MODEL}"
}
```

Then set the value at runtime:

```bash
export HF_BART_MODEL="your-team/gec-bart-base-s1s2"
python -m training.finetune --config configs/training/my-private-bart.json
```

Use a read-only Hugging Face token through `hf auth login` or the `HF_TOKEN`
environment variable. Never commit the token, a local Hugging Face cache, or
the checkpoint files. See [`docs/running.md`](../../docs/running.md#9-use-private-hugging-face-models).
