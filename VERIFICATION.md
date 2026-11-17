# Archive package verification

All 28 Python modules under `training/` match the archived source byte for byte.
The source hashes are recorded in `RELEASE_MANIFEST.json`.

Packaging checks passed for Python syntax, shell syntax, the explicit 1.7B
all-reference launch dry run, and the recorded 2,048-question training-data hash.
The launch requests eight GPUs and uses the archived local-positive all-rollout
configuration.

The four targeted CPU modules completed 54 tests successfully. These cover
credit, mathematical properties, disabled structural reward, and training
configuration. This is a targeted packaging check, separate from historical
suite reports retained under `docs/`. No new GPU training was performed.

Machine-specific paths and personal account defaults were anonymized in
non-training files. Credentials, environments, bytecode and model weights are
excluded. The added launch script and verification helper are identified in the
manifest. Details are in `docs/ARCHIVE_PACKAGE_VERIFICATION.json`.
