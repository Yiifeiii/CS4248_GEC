# Released models

The project's five main BART and T5 models are published on Hugging Face in the
[CS4248-GEC](https://huggingface.co/CS4248-GEC) organization. The repositories
are public: downloading them needs no Hugging Face account or token. Their IDs
are also listed in [`public_models.json`](public_models.json).

| Hugging Face repository | Training | dev_test F0.5 |
|---|---|---|
| [`CS4248-GEC/gec-bart-base-s1`](https://huggingface.co/CS4248-GEC/gec-bart-base-s1) | BART-base, stage 1 only | 43.8 |
| [`CS4248-GEC/gec-bart-base-s1s2`](https://huggingface.co/CS4248-GEC/gec-bart-base-s1s2) | BART-base, stage 1 + stage 2 | 52.3 |
| [`CS4248-GEC/gec-t5-base-s1`](https://huggingface.co/CS4248-GEC/gec-t5-base-s1) | T5-base, stage 1 only | 46.4 |
| [`CS4248-GEC/gec-t5-base-s1s2`](https://huggingface.co/CS4248-GEC/gec-t5-base-s1s2) | T5-base, stage 1 + stage 2 | 52.5 |
| [`CS4248-GEC/gec-t5-base-long-s1s2`](https://huggingface.co/CS4248-GEC/gec-t5-base-long-s1s2) | T5-base, longer stage 1 + longer stage 2 | 53.2 (best) |

Stage 1 trains on BEA-2019 data; stage 2 continues on W&I train. Each model
card lists the training details and scores. The runs that produced them are
recorded in [`experiments/training/`](../../experiments/training/README.md),
and their configs are in [`configs/training/`](../training/).

To download and run them, see
[`docs/running.md`](../../docs/running.md#9-use-the-public-team-models).
