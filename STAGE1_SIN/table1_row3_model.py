"""
table1_row3_model.py

Table I Row 3:
    SIN (MSA-AAGCN)
    Training = Mixed:
        Original + P1-P7 + MR0.2-MR0.6

This follows You-Young's tutorial for the 13-condition mixed training mode.

Important detail:
    The historical mixed mode does NOT simply draw one of the 13 conditions
    independently for every sample.

    It uses select_part_for_epoch(), which behaves like a random permutation:
    every block of 13 calls contains each condition exactly once, but in a
    random order.

Condition IDs:
    1  = P1 left arm
    2  = P2 right arm
    3  = P3 two hands
    4  = P4 two legs
    5  = P5 trunk
    6  = P6 left body
    7  = P7 right body
    8  = Original / no occlusion
    9  = MR0.2
    10 = MR0.3
    11 = MR0.4
    12 = MR0.5
    13 = MR0.6

Architecture is unchanged from Rows 1 and 2:
    3-layer MSA-AAGCN -> 64-channel features -> Linear(64, 3) decoder.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn

import GCN_Unsupervised.GCNEncoder.AGCNEncoder_3layer_action_atu as AGCNEncoder


random.seed(0)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):

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

        x = x.reshape(
            N * M,
            C,
            T,
            V
        )

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

        x = self.l3(x)

        x = x.permute(
            0,
            4,
            2,
            3,
            1
        )

        return x


# ============================================================
# Legacy classifier, instantiated but unused for Stage 1
# ============================================================

class AGCNClassifier(nn.Module):

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

        N, M, C, T, V = x.size()

        x = x.reshape(
            N,
            M,
            C,
            T * V
        )

        x = x.mean(3).mean(1)

        return self.fc(x)


# ============================================================
# Row 3 model
# ============================================================

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

        # ----------------------------------------------------
        # Same 3-layer MSA-AAGCN SIN backbone as Rows 1 and 2
        # ----------------------------------------------------

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

        # Legacy unused encoder kept for architectural continuity
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

        self.Decoder = Decoder()

        self.if_rotation = if_rotation
        self.if_vibrate = if_vibrate
        self.predict_seg = predict_seg

        self.Classifier = AGCNClassifier()

        # ----------------------------------------------------
        # Fixed-part definitions
        # ----------------------------------------------------

        self.parts = dict()

        self.parts[1] = (
            np.array(
                [5, 6, 7, 8, 22, 23]
            ) - 1
        )

        self.parts[2] = (
            np.array(
                [9, 10, 11, 12, 24, 25]
            ) - 1
        )

        self.parts[3] = (
            np.array(
                [22, 23, 24, 25]
            ) - 1
        )

        self.parts[4] = (
            np.array(
                [
                    13, 14, 15, 16,
                    17, 18, 19, 20
                ]
            ) - 1
        )

        self.parts[5] = (
            np.array(
                [1, 2, 3, 4, 21]
            ) - 1
        )

        self.parts[6] = (
            np.array(
                [
                    5, 6, 7, 8,
                    13, 14, 15, 16,
                    22, 23
                ]
            ) - 1
        )

        self.parts[7] = (
            np.array(
                [
                    9, 10, 11, 12,
                    17, 18, 19, 20,
                    24, 25
                ]
            ) - 1
        )

        # Original / no occlusion
        self.parts[8] = np.array(
            [],
            dtype=np.int32
        )

        # ----------------------------------------------------
        # Random-joint definitions
        # ----------------------------------------------------

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

        # Exact 13-condition mapping used by the historical
        # permutation-based mixed training code.
        self.index_list = np.array(
            [
                1, 2, 3, 4, 5, 6, 7, 8,
                0.2, 0.3, 0.4, 0.5, 0.6
            ]
        )

        self.count = 0
        self.parts_history = []

        # Kept because historical initialization consumes NumPy RNG.
        self.init_parts_9_13()


    # ========================================================
    # Historical initialization of parts 9-13
    #
    # These arrays are not used by the active permutation mixed
    # branch, but preserving these calls keeps RNG behavior closer
    # to the historical source.
    # ========================================================

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

        self.parts[9] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list) * 0.2
                ),
                replace=False
            )
            - 1
        )

        self.parts[10] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list) * 0.3
                ),
                replace=False
            )
            - 1
        )

        self.parts[11] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list) * 0.4
                ),
                replace=False
            )
            - 1
        )

        self.parts[12] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list) * 0.5
                ),
                replace=False
            )
            - 1
        )

        self.parts[13] = (
            np.random.choice(
                self.joint_list,
                int(
                    len(self.joint_list) * 0.6
                ),
                replace=False
            )
            - 1
        )


    # ========================================================
    # You-Young mixed-condition selector
    #
    # Each block of 13 calls is a random permutation of 1..13.
    # ========================================================

    def select_part_for_epoch(self):

        num_parts = 13

        if (
            self.count == 0
            or
            self.count % 13 == 0
        ):

            last_selected_part = {}

        else:

            last_selected_part = (
                self.parts_history[
                    -(self.count % 13):
                ]
            )

        available_parts = [

            part

            for part
            in range(
                1,
                num_parts + 1
            )

            if part
            not in last_selected_part
        ]

        selected_part = np.random.choice(
            available_parts
        )

        self.parts_history.append(
            selected_part
        )

        self.count += 1

        return selected_part


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        epoch,
        is_test=False
    ):

        N, C, T, V, M = x.size()

        if C != 3:
            raise RuntimeError(
                f"Expected C=3, got C={C}."
            )

        if V != 25:
            raise RuntimeError(
                f"Expected V=25, got V={V}."
            )

        if M != 2:
            raise RuntimeError(
                f"Expected M=2, got M={M}."
            )

        ori_data = x.clone()

        # Historical unused draw at the beginning of forward().
        # It consumes NumPy RNG and is therefore intentionally kept.
        _legacy_joint_occlusion = np.random.choice(
            self.joint_list,
            self.joint_occlusion_num,
            replace=False
        )

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

        # ----------------------------------------------------
        # ROW 3 MIXED TRAINING
        #
        # 13 balanced conditions by permutation:
        #
        # 1-7  : fixed P1-P7
        # 8    : Original
        # 9-13 : MR0.2-MR0.6
        # ----------------------------------------------------

        for i_b in range(N):

            part_index = int(
                self.select_part_for_epoch()
            )

            if part_index <= 8:

                part = self.parts[
                    part_index
                ]

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

            else:

                # 9->0.2, 10->0.3, ..., 13->0.6
                miss_rate = float(
                    self.index_list[
                        part_index - 1
                    ]
                )

                joint_occlusion_num = int(
                    len(self.joint_list)
                    * miss_rate
                )

                for t in range(T):

                    joint_occlusion = np.random.choice(
                        self.joint_list,
                        joint_occlusion_num,
                        replace=False
                    )

                    x[
                        i_b,
                        :,
                        t,
                        joint_occlusion,
                        :
                    ] = 0

                    x_mask[
                        i_b,
                        :,
                        t,
                        joint_occlusion,
                        :
                    ] = 0


        # ----------------------------------------------------
        # 3-layer MSA-AAGCN
        # ----------------------------------------------------

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

        output = self.Decoder(
            GCN_feature
        )

        x_mask = torch.from_numpy(
            x_mask
        ).to(
            device=x.device,
            dtype=torch.float32
        )

        # Historical selective replacement, computed for consistency.
        output_mask = (
            output * (1.0 - x_mask)
            +
            ori_data * x_mask
        )

        _ = output_mask

        return (
            output,
            GCN_feature,
            None,
            x_mask,
            None
        )
