"""
table1_row2_model.py

Table I Row 2:
    SIN (MSA-AAGCN)
    Training occlusion = Original + P1-P7

Purpose
-------
This file is a Row-2-only version of the Stage-1 SIN model used for the
manual Table I reproduction.

Compared with table1_row1_model.py:
    - Backbone is unchanged: 3-layer MSA-AAGCN.
    - Decoder is unchanged: 64 -> 3 coordinate regression.
    - Stage-1 output format is unchanged.
    - The ONLY experimental change is the training masking policy:
          Original + fixed-part P1-P7
      with one condition selected independently for each training sample.

Condition mapping
-----------------
    1 = P1 = left arm
    2 = P2 = right arm
    3 = P3 = two hands
    4 = P4 = two legs
    5 = P5 = trunk
    6 = P6 = left body
    7 = P7 = right body
    8 = Original / no occlusion

Notes
-----
The historical source contains several legacy modules and random-number draws.
For closer continuity with the Row 1 reproduction, this Row 2 version keeps:
    - the unused Encoder module,
    - the unused AGCNClassifier module,
    - initialization of legacy parts 9-13,
    - the legacy unused random-joint draw at the start of forward().

No ARN / MSG3D action-classification branch is executed. Table I Stage 1 is
trained with the reconstruction output only.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn

import GCN_Unsupervised.GCNEncoder.AGCNEncoder_3layer_action_atu as AGCNEncoder


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

random.seed(0)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Coordinate decoder for the 3-layer MSA-AAGCN SIN.

    Input:
        [N, M, 64, T, V]

    Output:
        [N, 3, T, V, M]
    """

    def __init__(self):
        super(Decoder, self).__init__()

        self.l3 = nn.Linear(
            64,
            3
        )

    def forward(self, x):

        N, M, C, T, V = x.size()

        if C != 64:
            raise RuntimeError(
                f"Decoder expected 64 feature channels, got {C}."
            )

        # [N, M, 64, T, V]
        x = x.reshape(
            N * M,
            C,
            T,
            V
        )

        # [N, M, T, V, 64]
        x = x.view(
            N,
            M,
            64,
            T,
            V
        ).permute(
            0,
            1,
            3,
            4,
            2
        )

        # [N, M, T, V, 3]
        x = self.l3(x)

        # [N, 3, T, V, M]
        x = x.permute(
            0,
            4,
            2,
            3,
            1
        )

        return x


# ---------------------------------------------------------------------------
# Legacy classifier
# ---------------------------------------------------------------------------

class AGCNClassifier(nn.Module):
    """
    Kept because the Row 1 historical Model_teacher also instantiated it.

    It is NOT used in Table I Stage-1 training.
    """

    def __init__(self):
        super(AGCNClassifier, self).__init__()

        num_class = 60

        self.fc = nn.Linear(
            64,
            num_class
        )

        nn.init.normal_(
            self.fc.weight,
            0,
            math.sqrt(
                2.0 / num_class
            )
        )

    def forward(self, x):

        # x: [N, M, 64, T, V]
        N, M, C, T, V = x.size()

        x = x.reshape(
            N,
            M,
            C,
            T * V
        )

        x = x.mean(3).mean(1)

        return self.fc(x)


# ---------------------------------------------------------------------------
# Row 2 SIN
# ---------------------------------------------------------------------------

class Model_teacher(nn.Module):

    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        if_rotation=False,
        seg_num=1,
        if_vibrate=False,
        prediction_mask=0,
        GCNEncoder="AGCN",
        ATU_layer=2,
        T=300,
        predict_seg=1
    ):

        super(Model_teacher, self).__init__()

        # -------------------------------------------------------------------
        # Main SIN backbone
        #
        # Same as Row 1:
        # 3-layer MSA-AAGCN encoder.
        # -------------------------------------------------------------------

        self.full_body = AGCNEncoder.AGCNEncoder(
            num_class,
            25,
            num_person,
            graph,
            graph_args,
            in_channels,
            adaptive=True,
            attention=True,
            ATU_layer=ATU_layer
        )

        self.FULL_BODY = np.asarray(
            [
                1, 2, 3, 4, 5,
                6, 7, 8, 9, 10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20,
                21, 22, 23, 24, 25
            ]
        )

        # -------------------------------------------------------------------
        # Legacy unused encoder
        #
        # Kept in the model so the structure remains aligned with the
        # manually reproduced Row 1 implementation.
        # -------------------------------------------------------------------

        if GCNEncoder == "AGCN":

            self.Encoder = AGCNEncoder.AGCNEncoder(
                num_class,
                num_point,
                num_person,
                graph,
                graph_args,
                64,
                ATU_layer=ATU_layer
            )

        # -------------------------------------------------------------------
        # Coordinate regression decoder
        # -------------------------------------------------------------------

        self.Decoder = Decoder()

        self.if_rotation = if_rotation
        self.if_vibrate = if_vibrate
        self.predict_seg = predict_seg

        # -------------------------------------------------------------------
        # Legacy classifier
        #
        # Instantiated but not executed in Stage-1 Table I training.
        # -------------------------------------------------------------------

        self.Classifier = AGCNClassifier()

        # -------------------------------------------------------------------
        # Fixed-part definitions
        #
        # All indices are converted from NTU's 1-based joint numbering
        # to Python's 0-based indexing.
        # -------------------------------------------------------------------

        self.parts = dict()

        # P1: left arm
        self.parts[1] = (
            np.array(
                [5, 6, 7, 8, 22, 23]
            ) - 1
        )

        # P2: right arm
        self.parts[2] = (
            np.array(
                [9, 10, 11, 12, 24, 25]
            ) - 1
        )

        # P3: two hands
        self.parts[3] = (
            np.array(
                [22, 23, 24, 25]
            ) - 1
        )

        # P4: two legs
        self.parts[4] = (
            np.array(
                [
                    13, 14, 15, 16,
                    17, 18, 19, 20
                ]
            ) - 1
        )

        # P5: trunk
        self.parts[5] = (
            np.array(
                [1, 2, 3, 4, 21]
            ) - 1
        )

        # P6: entire left side
        self.parts[6] = (
            np.array(
                [
                    5, 6, 7, 8,
                    13, 14, 15, 16,
                    22, 23
                ]
            ) - 1
        )

        # P7: entire right side
        self.parts[7] = (
            np.array(
                [
                    9, 10, 11, 12,
                    17, 18, 19, 20,
                    24, 25
                ]
            ) - 1
        )

        # Condition 8: Original / no occlusion
        self.parts[8] = np.array(
            [],
            dtype=np.int32
        )

        # -------------------------------------------------------------------
        # Legacy random-joint definitions
        #
        # Row 2 does NOT train with these conditions.
        #
        # They are retained only because the Row 1 historical model creates
        # them during initialization, which consumes NumPy RNG calls.
        # -------------------------------------------------------------------

        self.joint_list = (
            np.array(
                [
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25
                ]
            ) - 1
        )

        miss_rate = 0.6

        self.joint_occlusion_num = int(
            len(self.joint_list)
            * miss_rate
        )

        self.parts_history = []

        self.init_parts_9_13()


    # -----------------------------------------------------------------------
    # Legacy random-part initialization
    # -----------------------------------------------------------------------

    def init_parts_9_13(self):

        self.parts[9] = np.array(
            [],
            dtype=np.int32
        )

        self.parts[10] = np.array(
            [],
            dtype=np.int32
        )

        self.parts[11] = np.array(
            [],
            dtype=np.int32
        )

        self.parts[12] = np.array(
            [],
            dtype=np.int32
        )

        self.parts[13] = np.array(
            [],
            dtype=np.int32
        )

        # These are not used by Row 2.
        #
        # They intentionally remain here to keep initialization behavior
        # close to the historical Row 1 source.

        self.parts[9] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list)
                    * 0.2
                ),
                replace=False
            )
            - 1
        )

        self.parts[10] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list)
                    * 0.3
                ),
                replace=False
            )
            - 1
        )

        self.parts[11] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list)
                    * 0.4
                ),
                replace=False
            )
            - 1
        )

        self.parts[12] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list)
                    * 0.5
                ),
                replace=False
            )
            - 1
        )

        self.parts[13] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list)
                    * 0.6
                ),
                replace=False
            )
            - 1
        )


    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x,
        epoch,
        is_test=False
    ):

        N, C, T, V, M = x.size()

        if C != 3:
            raise RuntimeError(
                f"Expected C=3 XYZ input, got C={C}."
            )

        if V != 25:
            raise RuntimeError(
                f"Expected V=25 NTU joints, got V={V}."
            )

        if M != 2:
            raise RuntimeError(
                f"Expected M=2 persons, got M={M}."
            )

        ori_data = x.clone()

        # -------------------------------------------------------------------
        # Preserve one legacy NumPy draw used at the beginning of Row 1
        # forward(). The resulting joints are NOT used for Row 2 masking.
        # -------------------------------------------------------------------

        _legacy_joint_occlusion = np.random.choice(
            self.joint_list,
            self.joint_occlusion_num,
            replace=False
        )

        # 1 = observed
        # 0 = occluded
        x_mask = np.ones(
            (
                N,
                C,
                T,
                V,
                M
            ),
            dtype=np.float32
        )

        # -------------------------------------------------------------------
        # ROW 2 TRAINING MASK
        #
        # Original + P1-P7 only.
        #
        # For every training sample choose one of 8 conditions:
        #
        #     0 -> P1
        #     1 -> P2
        #     2 -> P3
        #     3 -> P4
        #     4 -> P5
        #     5 -> P6
        #     6 -> P7
        #     7 -> Original / no occlusion
        #
        # The selected fixed part is removed for the whole sequence
        # and for both persons.
        # -------------------------------------------------------------------

        num_parts = 8

        part_indices = np.random.choice(
            num_parts,
            size=N,
            replace=True
        )

        for i_b in range(N):

            part_index = int(
                part_indices[i_b]
            )

            part = self.parts[
                part_index + 1
            ]

            # parts[8] is empty, therefore condition 8 means
            # original / no occlusion.

            x[
                i_b,
                :,
                :,
                part,
                :
            ] = 0

            x_mask[
                i_b,
                :,
                :,
                part,
                :
            ] = 0


        # -------------------------------------------------------------------
        # MSA-AAGCN feature extraction
        # -------------------------------------------------------------------

        GCN_feature = torch.empty(
            N,
            M,
            64,
            T,
            V,
            device=x.device,
            dtype=torch.float32
        )

        GCN_feature[
            :,
            :,
            :,
            :,
            self.FULL_BODY - 1
        ] = self.full_body(
            x[
                :,
                :,
                :,
                self.FULL_BODY - 1,
                :
            ]
        )


        # -------------------------------------------------------------------
        # Coordinate regression
        # -------------------------------------------------------------------

        output = self.Decoder(
            GCN_feature
        )


        # -------------------------------------------------------------------
        # Selective replacement
        #
        # This is computed for consistency with the Row 1 implementation,
        # although the Stage-1 training script optimizes `output` with MSE.
        # -------------------------------------------------------------------

        x_mask = torch.from_numpy(
            x_mask
        ).to(
            device=x.device,
            dtype=torch.float32
        )

        output_mask = (
            output
            * (1.0 - x_mask)
            +
            ori_data
            * x_mask
        )

        # Keep variable alive for source readability.
        # It is not returned separately because Row 1 returned the raw
        # decoder output as element 0.
        _ = output_mask


        # -------------------------------------------------------------------
        # Same tuple structure expected by train_table1_row1.py:
        #
        # [0] raw reconstructed coordinates
        # [1] GCN features
        # [2] action output, unused for Table I Stage 1
        # [3] visibility mask
        # [4] unused
        # -------------------------------------------------------------------

        return (
            output,
            GCN_feature,
            None,
            x_mask,
            None
        )
