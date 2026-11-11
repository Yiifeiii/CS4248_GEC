# Configuration layout

Configurations now have a canonical home under this directory:

- `configs/training/`: BART/T5 training, controls, and decoding configs.
- `configs/t5_tuning/`: T5 tuning configs and manifests.

Compatibility links remain at `training/configs/` and `t5_tuning/configs/`, so
existing commands, tests, and cluster jobs continue to work. New configs should
be added here and referenced by their canonical path where possible.
