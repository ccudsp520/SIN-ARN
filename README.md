# Two-Stage Skeleton Imputation and Action Recognition

Source code organized into two stages:

- **Stage 1: SIN (Skeleton Imputation Network).** Reconstruct missing skeleton coordinates.
- **Stage 2: ARN (Action Recognition Network).** Recognize actions from repaired skeleton sequences, including teacher and knowledge-distillation experiment scripts.

## Structure

| Directory | Contents |
| --- | --- |
| `stage1/` | SIN source, encoders, decoders, graph definitions, dataset utilities, and Table I experiment scripts |
| `stage2/` | ARN source, teacher/KD scripts, Table II evaluation scripts, configurations, and data preparation utilities |

Source filenames and model implementations are preserved from the supplied MSAG3D and hong_MSG3D projects. No model refactoring is included in this package.

## Data and checkpoints

Datasets, pretrained weights, training outputs, virtual environments, and logs are excluded. Prepare the data and checkpoints separately and configure their paths in the corresponding scripts.

## Running the code

Install PyTorch for your system and the dependencies listed in `requirements.txt`. Some historical scripts use additional packages.

Review each experiment script and its data, checkpoint, teacher-score, and output paths before running it. The source retains machine-specific paths from the original experiments. Update references to the MSAG3D and hong_MSG3D project directories to the locations of `stage1` and `stage2` as appropriate.

The experiment scripts retain their original filenames rather than introducing a unified training entry point.

## Source availability

Some included historical scripts import `table1_row3_model.py` or `table1_row5_model.py`. These files were not present in the supplied source archive. Restore the original files before running experiments that depend on them. Checkpoint structure alone does not establish the missing training-mask policy.

Some dataset utilities also import the external `DLNest` package. The relevant original dependencies must be restored or the unused imports adapted before using those utilities.

This package provides the available research source code. A complete reproduction of every reported experiment has not been verified from this package.

## Attribution

Retain the copyright and attribution notices already present in source files. Several modules derive from existing graph-convolution and view-adaptation implementations. No new repository-wide license has been assigned in this package.
