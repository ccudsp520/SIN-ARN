# MSAG3D: Skeleton Imputation Network

Selected source code from the original MSAG3D project: the MSA-G3D skeleton
imputation network (SIN), MSA-AAGCN ablations, evaluation scripts, and viewers.
The separate ARN project, `hong_MSG3D`, is not bundled here.

## Code map

| Purpose | Files |
| --- | --- |
| Original MSA-G3D SIN training | `train_action_pre_atu_msa.py` |
| Original SIN wrapper and coordinate decoder | `unsupervised_globalgcn_3layer_action_KD_atu.py` |
| MSA-G3D feature encoder | `model/msag3d.py` |
| MSG3D classification modules used by existing code | `model/msg3d.py`, `model/msg3d2.py` |
| MSA-AAGCN Table I ablations | `table1_row1_model.py`, `table1_row2_model.py`, `table1_row3_model.py`, corresponding training scripts |
| Evaluation and epoch sweeps | `test_pose2.py`, `test_post_all.py`, `test_table1_*.py` |
| Approximate Row 1 continuation | `train_continue_table1_row1_to80.py` |
| Data loading and historical preprocessing | `Dataset/` |
| Graphs, transforms, and encoder dependencies | `graph/`, `src/`, `GCN_Unsupervised/` |
| Complete / incomplete / predicted / repaired viewer | `visualize_msag3d_four_panels_3.py` |
| Lightweight mask viewer | `visualize_msag3d_mask.py` |
| Supplied historical CSV outputs | `results/` |

## Environment and local inputs

`requirements.txt` lists third-party packages found in the retained imports.
It is not a pinned or validated reconstruction of the training environment.
Use a PyTorch installation compatible with your GPU and existing research setup.

`Dataset/Datasets.py` also imports `DLNest.Common.DatasetBase`, which is absent
from the supplied ZIP. Keep the original DLNest installation available when
running the dataset-based training and evaluation scripts.

Existing scripts retain their original settings. Before running them, adjust
dataset, checkpoint, save, and CSV output paths for your machine. Several
scripts still contain absolute Linux paths. Some also require CUDA.
Training and evaluation scripts can execute work at module import time.

Datasets, checkpoints, exported NPZ samples, logs, caches, and CUDA build
outputs are stored outside this source package. Keep checkpoints E61 through
E80 in `MSAG3D_CHECKPOINTS/weight_hong/` for the full checkpoint sweep.

The four-panel viewer accepts explicit paths, so the checkpoint can stay
outside the repository. From this folder, use the following command after
adjusting the paths to your data:

```powershell
python visualize_msag3d_four_panels_3.py --checkpoint "../MSAG3D_CHECKPOINTS/weight_hong/msag3d_epoch73.pth" --data "../hong_MSG3D/data/xsub/val_data_joint.npy" --labels "../hong_MSG3D/data/xsub/val_label.pkl"
```

This is a visualization example using an explicitly named existing checkpoint;
it does not select the best epoch. To open a previously exported four-panel
sample without loading the model:

```powershell
python visualize_msag3d_four_panels_3.py --view "../MSAG3D_ARCHIVE/ntu60_sin_four_panels.npz"
```

## Reproduction notes

The original MSA-G3D `Model_teacher` currently trains with fixed P1 masking.
The MSA-AAGCN ablation files implement different training policies: Row 1
uses random joint removal, Row 2 uses Original plus P1-P7, and Row 3 uses a
permutation of Original, P1-P7, and MR0.2-MR0.6. Their code is retained separately.

Historical sweep scripts contain their original epoch lists, including
shortlists. For a new Table I/SIN checkpoint selection, evaluate every epoch
E61 through E80 on P1-P7 and MR0.2-MR0.6, then select one epoch by the lowest
mean MPJPE across the 12 conditions. The historical CSVs are not that full sweep.

`train_continue_table1_row1_to80.py` starts a new optimizer from E60 weights;
its own output describes this as approximate continuation.
`Dataset/data_gen/gen_data_ntu60.py` and `gen_data_ntu120.py` currently generate
150-frame data, whereas the retained MSA-G3D model and viewers use 300 frames.
Use the original experiment's matching preprocessing and data.

The original `mpjpe()` in the main trainer and two older evaluation scripts
reduces the last tensor axis. The Table I scripts also contain XYZ-based
metrics reducing the coordinate axis. Check the metric definition when
interpreting historical results.

## Packaging checks

All retained source files are byte-for-byte copies of the supplied ZIP.
Internal imports and the dynamic graph/viewer references were checked for
retained-file coverage, and the retained Python sources compile.
Training, GPU inference, accuracy, MPJPE evaluation, and environment installation
were not executed during this file cleanup.

See `CLEANUP.md` for the file selection and archive details.
