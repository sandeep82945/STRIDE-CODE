# Run settings

The editable [run_settings.toml](../run_settings.toml) and the main
[RUN_SETTINGS.md guide](../RUN_SETTINGS.md) are now in the package root.

After preparing the model/data and placing the W&B key in root `.env`, edit the
root settings file, inspect with `python run.py train --dry-run`, then submit
`sbatch run.sbatch` from the package root.
