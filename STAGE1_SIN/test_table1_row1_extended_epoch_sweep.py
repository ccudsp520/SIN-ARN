import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.Datasets import NTUDataSet
from src.graph import Graph
from table1_row1_model import Model_teacher


# ============================================================
# 1. SETTINGS
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DATA_ROOT = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "hong_MSG3D/data/"
)


# Original Row 1 checkpoints
WEIGHT_DIR_ORIGINAL = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/weight_table1_row1"
)


# Extended epoch 61-80 checkpoints
WEIGHT_DIR_EXTENDED = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/weight_table1_row1_extended"
)


OUTPUT_CSV = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/table1_row1_extended_epoch_sweep.csv"
)


TEST_BATCH_SIZE = 24
FRAME = 300


# Checkpoints that we want to compare
EPOCHS = [
    60,
    65,
    70,
    73,
    75,
    80
]


# ============================================================
# 2. DATASET
# ============================================================

val_dataset_config = {

    "data_path":
        DATA_ROOT + "xsub/val_data_joint.npy",

    "label_path":
        DATA_ROOT + "xsub/val_label.pkl",

    "use_mmap": True,

    "debug": False
}


test_dataset = NTUDataSet(
    val_dataset_config
)


test_dataloader = DataLoader(

    dataset=test_dataset,

    batch_size=TEST_BATCH_SIZE,

    shuffle=False,

    num_workers=4
)


print()
print("=" * 80)
print("TABLE I ROW 1 EXTENDED EPOCH SWEEP")
print("=" * 80)

print(
    "Validation samples:",
    len(test_dataset)
)

print(
    "Device:",
    device
)

print(
    "Epochs:",
    EPOCHS
)

print("=" * 80)


# ============================================================
# 3. GRAPH
# ============================================================

gcn_kernel_size = [5, 2]


graph = Graph(
    max_hop=gcn_kernel_size[1]
)


A = torch.tensor(

    graph.A,

    dtype=torch.float32,

    requires_grad=False

).to(device)


# ============================================================
# 4. CREATE ROW 1 MODEL
# ============================================================

model = Model_teacher(

    num_class=60,

    num_point=25,

    num_person=2,

    graph=A,

    graph_args=dict(),

    in_channels=3,

    if_rotation=False,

    seg_num=1,

    ATU_layer=2,

    if_vibrate=False,

    prediction_mask=0,

    GCNEncoder="AGCN",

    T=300,

    predict_seg=1
)


model.to(device)

model.eval()


# ============================================================
# 5. FIXED-PART OCCLUSION DEFINITIONS
# ============================================================

parts = {

    # P1: left arm
    1:
        np.array(
            [5, 6, 7, 8, 22, 23]
        ) - 1,


    # P2: right arm
    2:
        np.array(
            [9, 10, 11, 12, 24, 25]
        ) - 1,


    # P3: two hands
    3:
        np.array(
            [22, 23, 24, 25]
        ) - 1,


    # P4: two legs
    4:
        np.array(
            [
                13, 14, 15, 16,
                17, 18, 19, 20
            ]
        ) - 1,


    # P5: trunk
    5:
        np.array(
            [1, 2, 3, 4, 21]
        ) - 1,


    # P6: left body
    6:
        np.array(
            [
                5, 6, 7, 8,
                13, 14, 15, 16,
                22, 23
            ]
        ) - 1,


    # P7: right body
    7:
        np.array(
            [
                9, 10, 11, 12,
                17, 18, 19, 20,
                24, 25
            ]
        ) - 1
}


joint_list = np.arange(25)


# ============================================================
# 6. TEST CONDITIONS
# ============================================================

conditions = [

    ("P1", 1),

    ("P2", 2),

    ("P3", 3),

    ("P4", 4),

    ("P5", 5),

    ("P6", 6),

    ("P7", 7),

    ("MR0.2", 0.2),

    ("MR0.3", 0.3),

    ("MR0.4", 0.4),

    ("MR0.5", 0.5),

    ("MR0.6", 0.6)
]


# ============================================================
# 7. PAPER TABLE I ROW 1 TARGET
# ============================================================

paper_target = {

    "P1": 55.4,

    "P2": 58.0,

    "P3": 34.9,

    "P4": 118.8,

    "P5": 29.4,

    "P6": 118.4,

    "P7": 120.5,

    "MR0.2": 10.2,

    "MR0.3": 14.0,

    "MR0.4": 20.0,

    "MR0.5": 24.0,

    "MR0.6": 29.8
}


# ============================================================
# 8. RANDOM SEED FOR EACH RANDOM-OCCLUSION CONDITION
#
# Important:
# Every epoch must receive exactly the same random occlusion.
#
# Therefore we reset the seed before every MR condition.
# ============================================================

condition_seed = {

    "MR0.2": 200,

    "MR0.3": 300,

    "MR0.4": 400,

    "MR0.5": 500,

    "MR0.6": 600
}


# ============================================================
# 9. APPLY OCCLUSION
# ============================================================

def apply_occlusion(data, occ):

    N, C, T, V, M = data.shape


    mask_np = np.ones(

        (N, C, T, V, M),

        dtype=np.float32
    )


    # ========================================================
    # FIXED-PART OCCLUSION
    # ========================================================

    if isinstance(occ, int):

        missing_joints = parts[occ]


        mask_np[
            :, :, :, missing_joints, :
        ] = 0.0


    # ========================================================
    # RANDOM-JOINT OCCLUSION
    # ========================================================

    else:

        miss_rate = float(occ)


        joint_occlusion_num = int(

            len(joint_list)

            * miss_rate
        )


        for i_b in range(N):


            for t in range(T):


                joint_occlusion = np.random.choice(

                    joint_list,

                    joint_occlusion_num,

                    replace=False
                )


                mask_np[

                    i_b,

                    :,

                    t,

                    joint_occlusion,

                    :

                ] = 0.0


    x_mask = torch.from_numpy(

        mask_np

    ).to(

        device=data.device,

        dtype=data.dtype
    )


    masked_data = (

        data

        * x_mask
    )


    return masked_data, x_mask


# ============================================================
# 10. STANDARD XYZ MPJPE
#
# Tensor shape:
#
# [N, C, T, V, M]
#
# C = XYZ coordinate dimension
#
# Therefore Euclidean distance is calculated over dim=1.
# ============================================================

def xyz_mpjpe(predicted, target):

    error = torch.norm(

        predicted - target,

        dim=1
    )


    return torch.mean(
        error
    )


# ============================================================
# 11. SIN FORWARD
#
# Important:
#
# We DO NOT call:
#
# model(masked_data, epoch)
#
# because Model_teacher.forward() contains the training-time
# random masking mechanism.
#
# For evaluation, masking has already been generated above.
# ============================================================

def sin_forward(masked_data):


    gcn_feature = model.full_body(
        masked_data
    )


    output = model.Decoder(
        gcn_feature
    )


    return output


# ============================================================
# 12. SELECT CHECKPOINT PATH
# ============================================================

def get_checkpoint(epoch_num):


    # Epoch 60 comes from the original 60-epoch training
    if epoch_num == 60:

        checkpoint = os.path.join(

            WEIGHT_DIR_ORIGINAL,

            f"row1_epoch{epoch_num}.pth"
        )


    # Epoch 61-80 come from approximate continuation
    else:

        checkpoint = os.path.join(

            WEIGHT_DIR_EXTENDED,

            f"row1_epoch{epoch_num}.pth"
        )


    return checkpoint


# ============================================================
# 13. EVALUATE ALL CHECKPOINTS
# ============================================================

all_results = []


for epoch_num in EPOCHS:


    checkpoint = get_checkpoint(
        epoch_num
    )


    print()

    print("#" * 80)

    print(
        "EPOCH:",
        epoch_num
    )

    print(
        "Checkpoint:",
        checkpoint
    )

    print("#" * 80)


    # ========================================================
    # CHECK FILE
    # ========================================================

    if not os.path.exists(checkpoint):

        print(
            "ERROR: CHECKPOINT NOT FOUND"
        )

        print(
            checkpoint
        )

        continue


    # ========================================================
    # LOAD MODEL WEIGHTS
    # ========================================================

    state_dict = torch.load(

        checkpoint,

        map_location=device
    )


    model.load_state_dict(

        state_dict,

        strict=True
    )


    model.eval()


    print(
        "Checkpoint loaded successfully."
    )


    epoch_results = []


    # ========================================================
    # TEST ALL 12 OCCLUSION CONDITIONS
    # ========================================================

    for condition_name, occ in conditions:


        print()

        print("=" * 70)

        print(
            f"Epoch {epoch_num}"
            f" | Testing {condition_name}"
        )

        print("=" * 70)


        # ====================================================
        # RESET RANDOM SEED
        #
        # This ensures that epoch 60, 65, 70, 73, 75 and 80
        # see exactly the same random masks.
        # ====================================================

        if condition_name in condition_seed:


            seed = condition_seed[
                condition_name
            ]


            np.random.seed(
                seed
            )


            random.seed(
                seed
            )


            torch.manual_seed(
                seed
            )


        # ====================================================
        # ACCUMULATORS
        # ====================================================

        total_error = 0.0

        total_samples = 0


        # ====================================================
        # TEST
        # ====================================================

        with torch.no_grad():


            for batch in tqdm(

                test_dataloader,

                dynamic_ncols=True
            ):


                data = batch[
                    "data"
                ].to(device)


                data = data[
                    :, :, :FRAME, :, :
                ]


                batch_size = data.size(0)


                # ============================================
                # CREATE OCCLUDED INPUT
                # ============================================

                masked_data, x_mask = apply_occlusion(

                    data,

                    occ
                )


                # ============================================
                # SIN PREDICTION
                # ============================================

                output = sin_forward(
                    masked_data
                )


                # ============================================
                # SELECTIVE REPLACEMENT
                #
                # Missing joints:
                # use SIN prediction
                #
                # Observed joints:
                # preserve original skeleton
                # ============================================

                repaired = (

                    output
                    * (1.0 - x_mask)

                    +

                    data
                    * x_mask
                )


                # ============================================
                # STANDARD XYZ MPJPE
                # ============================================

                error = xyz_mpjpe(

                    repaired,

                    data
                )


                # ============================================
                # WEIGHT BY BATCH SIZE
                #
                # The last validation batch may contain fewer
                # than 24 samples.
                # ============================================

                total_error += (

                    error.item()

                    * batch_size
                )


                total_samples += (
                    batch_size
                )


        # ====================================================
        # CONVERT METERS TO MILLIMETERS
        # ====================================================

        mpjpe_mm = (

            total_error

            / total_samples

            * 1000.0
        )


        target = paper_target[
            condition_name
        ]


        difference = (

            mpjpe_mm

            - target
        )


        # ====================================================
        # SAVE CONDITION RESULT
        # ====================================================

        row = {

            "Epoch":
                epoch_num,

            "Condition":
                condition_name,

            "XYZ_MPJPE_mm":
                mpjpe_mm,

            "Paper_mm":
                target,

            "Difference_mm":
                difference
        }


        all_results.append(
            row
        )


        epoch_results.append(
            row
        )


        # ====================================================
        # PRINT CONDITION RESULT
        # ====================================================

        print()

        print(
            "XYZ MPJPE :",
            f"{mpjpe_mm:.3f} mm"
        )


        print(
            "Paper     :",
            f"{target:.3f} mm"
        )


        print(
            "Difference:",
            f"{difference:+.3f} mm"
        )


    # ========================================================
    # 14. SUMMARY FOR CURRENT EPOCH
    # ========================================================

    epoch_values = [

        item["XYZ_MPJPE_mm"]

        for item in epoch_results
    ]


    target_values = [

        item["Paper_mm"]

        for item in epoch_results
    ]


    differences = [

        item["Difference_mm"]

        for item in epoch_results
    ]


    avg_mpjpe = np.mean(
        epoch_values
    )


    paper_avg = np.mean(
        target_values
    )


    mean_absolute_difference = np.mean(

        np.abs(
            differences
        )
    )


    print()

    print("-" * 80)


    print(
        f"Epoch {epoch_num} Avg XYZ MPJPE:"
    )

    print(
        f"{avg_mpjpe:.3f} mm"
    )


    print()

    print(
        "Paper Avg:"
    )

    print(
        f"{paper_avg:.3f} mm"
    )


    print()

    print(
        "Difference from Paper Avg:"
    )

    print(
        f"{avg_mpjpe - paper_avg:+.3f} mm"
    )


    print()

    print(
        "Mean Absolute Difference "
        "across 12 conditions:"
    )

    print(
        f"{mean_absolute_difference:.3f} mm"
    )


    print("-" * 80)


# ============================================================
# 15. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_results
)


# ============================================================
# 16. SAVE FULL RESULTS
# ============================================================

df.to_csv(

    OUTPUT_CSV,

    index=False
)


# ============================================================
# 17. FINAL SUMMARY BY EPOCH
# ============================================================

summary = (

    df.groupby("Epoch")

    .agg(

        Avg_XYZ_MPJPE_mm=(

            "XYZ_MPJPE_mm",

            "mean"
        ),


        Mean_Absolute_Difference_mm=(

            "Difference_mm",

            lambda x:
                np.mean(
                    np.abs(x)
                )
        )

    )

    .reset_index()
)


paper_average = np.mean(

    list(
        paper_target.values()
    )
)


summary[
    "Difference_from_Paper_Avg_mm"
] = (

    summary[
        "Avg_XYZ_MPJPE_mm"
    ]

    - paper_average
)


# ============================================================
# 18. PRINT FINAL SUMMARY
# ============================================================

print()

print("=" * 100)

print(
    "FINAL EXTENDED EPOCH SWEEP SUMMARY"
)

print("=" * 100)


print(

    summary.to_string(

        index=False,

        float_format=lambda x:
            f"{x:.3f}"
    )
)


print()

print(
    "Paper Row 1 Avg:",
    f"{paper_average:.3f} mm"
)


print()

print(
    "Full results saved to:"
)

print(
    OUTPUT_CSV
)


print()

print("=" * 100)

print(
    "IMPORTANT"
)

print("=" * 100)

print(
    "Epoch 60 is the official 60-epoch Row 1 reproduction."
)

print(
    "Epoch 65-80 are diagnostic approximate-continuation checkpoints."
)

print(
    "Do not select a checkpoint only because its average is closest "
    "to the paper."
)

print("=" * 100)