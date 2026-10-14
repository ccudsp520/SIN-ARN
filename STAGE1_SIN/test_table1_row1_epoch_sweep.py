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
    'cuda:0' if torch.cuda.is_available() else 'cpu'
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

WEIGHT_DIR = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/weight_table1_row1"
)

OUTPUT_CSV = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/table1_row1_epoch_sweep.csv"
)

TEST_BATCH_SIZE = 24
FRAME = 300

EPOCHS = [55, 56, 57, 58, 59, 60]



# ============================================================
# 2. DATASET
# ============================================================

val_dataset_config = {
    "data_path":
        DATA_ROOT + "xsub/val_data_joint.npy",

    "label_path":
        DATA_ROOT + "xsub/val_label.pkl",

    "use_mmap": True,
    "debug": False,
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

print("Validation samples:", len(test_dataset))
print("Device:", device)


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
# 4. CREATE MODEL
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
# 5. OCCLUSION DEFINITIONS
# ============================================================

parts = {
    1: np.array([5, 6, 7, 8, 22, 23]) - 1,
    2: np.array([9, 10, 11, 12, 24, 25]) - 1,
    3: np.array([22, 23, 24, 25]) - 1,

    4: np.array(
        [13, 14, 15, 16, 17, 18, 19, 20]
    ) - 1,

    5: np.array([1, 2, 3, 4, 21]) - 1,

    6: np.array(
        [5, 6, 7, 8, 13, 14, 15, 16, 22, 23]
    ) - 1,

    7: np.array(
        [9, 10, 11, 12, 17, 18, 19, 20, 24, 25]
    ) - 1,
}

joint_list = np.arange(25)


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
    ("MR0.6", 0.6),
]


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
    "MR0.6": 29.8,
}


# ============================================================
# 6. FIXED RANDOM SEEDS FOR EACH RANDOM-OCCLUSION CONDITION
#
# Every checkpoint gets exactly the SAME random masks.
# ============================================================

condition_seed = {
    "MR0.2": 200,
    "MR0.3": 300,
    "MR0.4": 400,
    "MR0.5": 500,
    "MR0.6": 600,
}


# ============================================================
# 7. APPLY OCCLUSION
# ============================================================

def apply_occlusion(data, occ):

    N, C, T, V, M = data.shape

    mask_np = np.ones(
        (N, C, T, V, M),
        dtype=np.float32
    )

    # Fixed-part P1-P7
    if isinstance(occ, int):

        missing_joints = parts[occ]

        mask_np[
            :, :, :, missing_joints, :
        ] = 0.0

    # Random-joint MR0.2-MR0.6
    else:

        miss_rate = float(occ)

        joint_occlusion_num = int(
            len(joint_list) * miss_rate
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

    masked_data = data * x_mask

    return masked_data, x_mask


# ============================================================
# 8. STANDARD XYZ MPJPE
# ============================================================

def xyz_mpjpe(predicted, target):

    return torch.mean(
        torch.norm(
            predicted - target,
            dim=1
        )
    )


# ============================================================
# 9. SIN FORWARD ONLY
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
# 10. EVALUATION
# ============================================================

all_results = []


for epoch_num in EPOCHS:

    checkpoint = os.path.join(
        WEIGHT_DIR,
        f"row1_epoch{epoch_num}.pth"
    )

    print()
    print("#" * 80)
    print("EPOCH:", epoch_num)
    print("Checkpoint:", checkpoint)
    print("#" * 80)

    if not os.path.exists(checkpoint):
        print("CHECKPOINT NOT FOUND")
        continue

    state_dict = torch.load(
        checkpoint,
        map_location=device
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    epoch_results = []


    for condition_name, occ in conditions:

        print()
        print(
            f"Epoch {epoch_num} | "
            f"Testing {condition_name}"
        )

        # ----------------------------------------------------
        # Critical:
        # Reset random seed so all checkpoints get exactly
        # the same random masks.
        # ----------------------------------------------------

        if condition_name in condition_seed:

            seed = condition_seed[
                condition_name
            ]

            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        total_error = 0.0
        total_samples = 0


        with torch.no_grad():

            for batch in tqdm(
                test_dataloader,
                dynamic_ncols=True
            ):

                data = batch["data"].to(device)

                data = data[
                    :, :, :FRAME, :, :
                ]

                batch_size = data.size(0)


                # --------------------------------------------
                # Apply the selected Table I occlusion
                # --------------------------------------------

                masked_data, x_mask = apply_occlusion(
                    data,
                    occ
                )


                # --------------------------------------------
                # SIN prediction
                # --------------------------------------------

                output = sin_forward(
                    masked_data
                )


                # --------------------------------------------
                # Selective replacement
                # --------------------------------------------

                repaired = (
                    output * (1.0 - x_mask)
                    +
                    data * x_mask
                )


                # --------------------------------------------
                # XYZ MPJPE
                # --------------------------------------------

                error = xyz_mpjpe(
                    repaired,
                    data
                )

                # Weight by batch size.
                # Important because final batch may be smaller.
                total_error += (
                    error.item()
                    * batch_size
                )

                total_samples += batch_size


        mpjpe_mm = (
            total_error
            / total_samples
            * 1000.0
        )

        target = paper_target[
            condition_name
        ]

        diff = mpjpe_mm - target


        row = {
            "Epoch": epoch_num,
            "Condition": condition_name,
            "XYZ_MPJPE_mm": mpjpe_mm,
            "Paper_mm": target,
            "Difference_mm": diff,
        }

        all_results.append(row)
        epoch_results.append(row)


        print(
            f"XYZ MPJPE : {mpjpe_mm:.3f} mm"
        )

        print(
            f"Paper     : {target:.3f} mm"
        )

        print(
            f"Difference: {diff:+.3f} mm"
        )


    # ========================================================
    # EPOCH SUMMARY
    # ========================================================

    epoch_values = [
        x["XYZ_MPJPE_mm"]
        for x in epoch_results
    ]

    target_values = [
        x["Paper_mm"]
        for x in epoch_results
    ]

    avg_mpjpe = np.mean(
        epoch_values
    )

    paper_avg = np.mean(
        target_values
    )

    print()
    print("-" * 70)

    print(
        f"Epoch {epoch_num} Avg XYZ MPJPE : "
        f"{avg_mpjpe:.3f} mm"
    )

    print(
        f"Paper Avg                     : "
        f"{paper_avg:.3f} mm"
    )

    print(
        f"Difference                    : "
        f"{avg_mpjpe - paper_avg:+.3f} mm"
    )

    print("-" * 70)


# ============================================================
# 11. SAVE FULL RESULTS
# ============================================================

df = pd.DataFrame(
    all_results
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 12. SUMMARY BY EPOCH
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
            lambda x: np.mean(np.abs(x))
        )
    )
    .reset_index()
)


summary[
    "Difference_from_Paper_Avg_mm"
] = (
    summary["Avg_XYZ_MPJPE_mm"]
    -
    np.mean(
        list(paper_target.values())
    )
)


print()
print("=" * 90)
print("FINAL EPOCH SWEEP SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print()
print(
    "Paper Row 1 Avg:",
    f"{np.mean(list(paper_target.values())):.3f} mm"
)

print()
print(
    "Full results saved to:",
    OUTPUT_CSV
)