# SIN-ARN

A two-stage GCN framework for human action recognition from incomplete 3D skeleton sequences, based on the manuscript **A Two-Stage GCN-Based Framework for Human Action Recognition from Incomplete Visual Skeleton Sensing**.

## Method

1. **Skeleton Imputation Network (SIN):** a GCN backbone with multi-head self-attention estimates missing 3D joint coordinates. A visibility mask combines these estimates with the input, preserving all observed joints.
2. **Action Recognition Network (ARN):** a one-stream GCN classifies the repaired skeleton using a 15-channel early-fused RICH representation: joint, bone, angle, joint-motion, and successive-motion features.

The paper considers AAGCN and MS-G3D backbones. SIN and ARN are trained separately. A five-stream teacher trained on complete skeletons transfers its class distributions to the student ARN through knowledge distillation. Student training combines cross-entropy and KL divergence. During inference, only SIN and the student ARN are required.

## Repository

| Folder                             | Contents                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| [STAGE1_SIN] | Skeleton imputation, ablations, evaluation, and repair visualization           |
| [STAGE2_ARN] | Action recognition, teacher-score KD, configurations, and experiment snapshots |

## Evaluation

The paper evaluates synthetic occlusion using seven fixed-part masks (P1-P7) and five random-joint masking ratios (0.2, 0.3, 0.4, 0.5, 0.6). Fixed-part masks remove the same body part throughout a sequence; random-joint masks select missing joints independently in each frame.

Main results reported for the MSA-G3D configuration on **NTU RGB+D 60 Cross-Subject**:

| Metric                                                | Reported result |
| ----------------------------------------------------- | --------------: |
| SIN average MPJPE                                     |         24.2 mm |
| SIN-ARN with five-stream teacher KD, average accuracy |           87.1% |

These averages cover the 12 occlusion conditions and exclude unmasked test sequences. They are manuscript results, not a new evaluation of this source release.

