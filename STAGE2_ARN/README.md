# STAGE2_ARN: Action Recognition Network

Selected source from the supplied hong_MSG3D project. The original Python
files and selected YAML/CSV files are preserved byte for byte. This folder
contains the historical NTU RGB+D 60 cross-subject SIN-to-ARN pipeline.

## Code map

| Purpose | Files |
| --- | --- |
| ARN training and evaluation launcher | `main_TS.py` |
| SIN reconstruction plus action classifier wrapper | `model/imputation_network.py` |
| SIN MSA-G3D encoder and temporal attention | `model/msag3d.py`, `model/MSA_ATnet.py` |
| Action classifier with concatenated joint, edge, surface, motion and acceleration features | `model/msg3d.py` |
| Local graph and convolution dependencies | `model/`, `graph/`, `feeders2/`, `utils.py` |
| Historical teacher-score ensemble | `ensemble_TS.py` |
| Original NTU60 CS ARN settings | `config/nturgbd-cross-subject/TS/` |
| Different training and test source snapshots | `experiment_snapshots/ntu60_cs/` |
| Historical per-epoch accuracy CSV | `results/B_TS_MSA_list_0521.csv` |
| Original skeleton viewer and MATLAB sample exporter | `visualize_ntu60_skeleton.py`, `export_ntu60_to_mat.py` |
| File decisions and model differences | `docs/FILE_SELECTION.json`, `docs/SNAPSHOT_DIFFERENCES.diff` |

## What the supplied code does

`main_TS.py` imports `model.imputation_network.Model_student` directly, loads
`imputation_weight/msag3d_finetune_epoch3.pth`, and freezes parameters outside
`MSG3D`. Its SGD optimizer updates the classifier. The loss is cross-entropy
plus KL divergence to precomputed teacher scores, with no explicit temperature.
The classifier constructs five feature groups and concatenates them into 15
input channels. This concatenation is separate from teacher-score fusion.

The trainer opens five teacher-score dictionaries, but the active fusion is
`0.3*S1 + 0.3*S2 + 0.3*S3 + 0.1*T2`. T3 is read but omitted from the sum.
The active sum in `ensemble_TS.py` also omits T3. These formulas have been
preserved. This package should not be described as a verified five-teacher
reproduction without checking the matching final experiment scripts.

## Source snapshots and masking

| Model source | Active masking |
| --- | --- |
| `model/imputation_network.py` | P1 only |
| `experiment_snapshots/ntu60_cs/train/imputation_network.py` | A permutation of Original, P1-P7 and MR0.2-MR0.6 across calls |
| `experiment_snapshots/ntu60_cs/test/imputation_network.py` | P1 plus random removal of 60% of joints, shared across frames in a forward call |

The training snapshot cycles through 13 conditions across sample calls.
Its MR branches draw joints separately for each frame. The test snapshot
combines P1 with a second joint mask; it is not a standalone MR0.6 test.

Snapshots are historical evidence. They retain imports from the main `model`
package, so running a snapshot launcher does not automatically select the
snapshot model. Select the matching implementation explicitly before any
reproduction. The saved training config also records `forward_batch_size: 1`,
while the reusable training YAML has 20. Logs contain more than one run and
are retained separately in the archive.

## Environment and required local inputs

`requirements.txt` lists packages imported by the selected code. Versions
and installation have not been validated. Use the matching research training
environment. The supplied Windows metadata records Python 3.9.13,
`torch-2.7.1+cpu` and `torchvision-0.22.1+cpu`; it does not establish the original
CUDA training environment. See `environment/REVIEW_ENVIRONMENT_PACKAGES.txt`.

The launcher imports NVIDIA Apex even when `half` is false. Apex is an external
dependency. The supplied Apex source, README and LICENSE are retained in
`hong_MSG3D_ARCHIVE.zip`; its Python implementation was not rewritten or
replaced with native AMP during this cleanup.

Keep the following inputs outside Git, and adjust their original paths before
running:

| Input | Original code location |
| --- | --- |
| NTU60 CS training and validation arrays and labels | `data/xsub/` |
| Initial combined model checkpoint | `imputation_weight/msag3d_finetune_epoch3.pth` |
| Five precomputed teacher-score dictionaries | `S1_epoch1_test_score.pkl`, `S2_epoch1_test_score.pkl`, `S3_epoch1_test_score.pkl`, `T2_epoch1_test_score.pkl`, `T3_epoch1_test_score.pkl` |
| Evaluation checkpoint | `work_dir/ntu/xsub/TS/MSAG3D_with_warm_up/train/weights/weights-57-57114.pt` |
| Teacher scores for the standalone ensemble utility | Relative `../teacher_weight/` paths in `ensemble_TS.py` |

Teacher dictionaries are paired by iteration order in the trainer, then
indexed alongside training labels. Their sample-name alignment cannot be
verified from this source-only upload. Preserve all five score files on the PC.
The trainer requires T3 to exist even though its score is omitted from fusion.

## Original entry points

Run scripts from inside `STAGE2_ARN`, because imports and data paths are
relative to this folder. Once the matching masking implementation, inputs,
paths and dependencies have been configured, the existing launcher accepts:

```bash
python main_TS.py --config config/nturgbd-cross-subject/TS/train_joint.yaml
python main_TS.py --config config/nturgbd-cross-subject/TS/test_joint.yaml
```

Its default config is the test config. It hardcodes both model and feeder,
so changing the YAML `model` or `feeder` strings alone does not change them.
The wrappers construct 60-class models and use fixed 300-frame tensors.
Several original model modules allocate tensors on CUDA directly.

The older CV, NTU120, Kinetics, LF and imputation YAMLs are retained in the
archive. They are not verified entry points for this launcher. Ten original
baseline test YAML files contain binary bytes and cannot be parsed as YAML.
Newer CV/NTU120 training and evaluation scripts from the later research runs
were not present in this supplied Windows copy.

## Checks performed

Selected Python files compile. File hashes match the uploaded source, selected
YAML files parse, and local dependencies used by the current ARN path were
checked for presence. The retained legacy MSAAGCN methods contain a deferred
`from Model.va import ViewAdaptive` import with uppercase `Model`; the active
MSA-G3D path uses lowercase `model.va`. This pre-existing issue is documented.
No training, GPU inference, checkpoint loading or accuracy evaluation was run.

See `CLEANUP.md` for the selection and original-folder instructions.
