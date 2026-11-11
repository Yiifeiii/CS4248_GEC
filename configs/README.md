# Configuration notes

Training configurations live in `configs/training/`, and T5 tuning
configurations live in `configs/t5_tuning/`. The old scattered config
directories have been removed; scripts, tests, and documentation now use these
canonical paths. Use environment variables for private Hugging Face model
identifiers rather than committing credentials or checkpoint files.
