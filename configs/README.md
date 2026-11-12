# Configuration notes

Training configurations live in `configs/training/`, and T5 tuning
configurations live in `configs/t5_tuning/`. The old scattered config
directories have been removed; scripts, tests, and documentation now use these
canonical paths. Public model identifiers are recorded in
`configs/models/public_models.json`; do not commit model weights, caches, or
credentials.
