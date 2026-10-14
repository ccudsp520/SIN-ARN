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
# 1. BASIC SETTINGS
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

CHECKPOINT = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/weight_table1_row1/row1_epoch60.pth"
)

OUTPUT_CSV = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/table1_row1_results.csv"
)

TEST_BATCH_SIZE = 24
FRAME = 300


# ============================================================
# 2. NTU RGB+D 60 X-SUB VALIDATION SET
# ============================================================

val_dataset_config = {
    "data_path":
        DATA_ROOT + "xsub/val_data_joint.npy",

    "label_path":
        DATA_ROOT + "xsub/val_label.pkl",

    "use_mmap": True,
    "debug": False,
}


test_dataset = NTUDataSet(val_dataset_config)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=4
)


# ============================================================
# 3. GRAPH
#    Same configuration as training
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
# 4. LOAD ROW 1 MODEL
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

print("Loading checkpoint:")
print(CHECKPOINT)

state_dict = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(device)
model.eval()

print("Checkpoint loaded successfully.")
print("Validation samples:", len(test_dataset))
print("Device:", device)


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


# Paper Table I Row 1
target_table1 = {
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
# 6. OCCLUSION FUNCTION
# ============================================================

def apply_occlusion(data, occ):

    N, C, T, V, M = data.shape

    mask_np = np.ones(
        (N, C, T, V, M),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Fixed-part occlusion P1-P7
    # --------------------------------------------------------

    if occ in parts:

        missing_joints = parts[occ]

        mask_np[
            :, :, :, missing_joints, :
        ] = 0.0

    # --------------------------------------------------------
    # Random-joint occlusion MR0.2-MR0.6
    # --------------------------------------------------------

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
# 7. MPJPE
# ============================================================

def legacy_mpjpe(predicted, target):

    # Exact behavior of the historical source code:
    #
    # tensor shape:
    # [N, C, T, V, M]
    #
    # dim=4 means norm over person dimension.
    #
    # Although this is not conventional XYZ MPJPE,
    # we keep it for reproduction of Table I.

    return torch.mean(
        torch.norm(
            predicted - target,
            dim=4
        )
    )


def xyz_mpjpe(predicted, target):

    # Conventional 3D MPJPE:
    # Euclidean norm over XYZ coordinate dimension.

    return torch.mean(
        torch.norm(
            predicted - target,
            dim=1
        )
    )


# ============================================================
# 8. SIN FORWARD WITHOUT TRAINING MASK GENERATION
# ============================================================

def sin_forward(masked_data):

    # IMPORTANT:
    # Do not call model(masked_data, epoch),
    # because Model_teacher.forward() creates another
    # random training occlusion.
    #
    # We use exactly the trained SIN components:
    #
    # masked skeleton
    # -> MSA-AAGCN 3-layer
    # -> 64-channel features
    # -> Decoder
    # -> predicted skeleton

    gcn_feature = model.full_body(
        masked_data
    )

    output = model.Decoder(
        gcn_feature
    )

    return output


# ============================================================
# 9. TEST ALL TABLE I CONDITIONS
# ============================================================

results = []


with torch.no_grad():

    for condition_name, occ in conditions:

        print()
        print("=" * 70)
        print("Testing:", condition_name)
        print("=" * 70)

        total_legacy_repaired = 0.0
        total_legacy_raw = 0.0
        total_xyz_repaired = 0.0
        num_batches = 0

        for batch in tqdm(
            test_dataloader,
            dynamic_ncols=True
        ):

            data = batch["data"].to(device)

            data = data[
                :, :, :FRAME, :, :
            ]

            masked_data, x_mask = apply_occlusion(
                data,
                occ
            )

            # -----------------------------------------------
            # SIN prediction
            # -----------------------------------------------

            output = sin_forward(
                masked_data
            )

            # -----------------------------------------------
            # Selective replacement
            #
            # missing:
            # use SIN prediction
            #
            # observed:
            # preserve original skeleton
            # -----------------------------------------------

            repaired = (
                output * (1.0 - x_mask)
                +
                data * x_mask
            )

            # -----------------------------------------------
            # Historical metric
            # -----------------------------------------------

            legacy_raw = legacy_mpjpe(
                output,
                data
            )

            legacy_repaired = legacy_mpjpe(
                repaired,
                data
            )

            # -----------------------------------------------
            # Standard XYZ MPJPE
            # -----------------------------------------------

            xyz_repaired = xyz_mpjpe(
                repaired,
                data
            )

            total_legacy_raw += (
                legacy_raw.item()
            )

            total_legacy_repaired += (
                legacy_repaired.item()
            )

            total_xyz_repaired += (
                xyz_repaired.item()
            )

            num_batches += 1


        # NTU coordinates are in meters.
        # Convert to millimeters.
        legacy_raw_mm = (
            total_legacy_raw
            / num_batches
            * 1000.0
        )

        legacy_repaired_mm = (
            total_legacy_repaired
            / num_batches
            * 1000.0
        )

        xyz_repaired_mm = (
            total_xyz_repaired
            / num_batches
            * 1000.0
        )

        target_mm = target_table1[
            condition_name
        ]

        difference_mm = (
            legacy_repaired_mm
            - target_mm
        )

        results.append({
            "Condition":
                condition_name,

            "Target_Table1_mm":
                target_mm,

            "Legacy_Raw_mm":
                legacy_raw_mm,

            "Legacy_Repaired_mm":
                legacy_repaired_mm,

            "XYZ_Repaired_mm":
                xyz_repaired_mm,

            "Difference_mm":
                difference_mm,
        })

        print(
            "Legacy raw MPJPE      :",
            f"{legacy_raw_mm:.3f} mm"
        )

        print(
            "Legacy repaired MPJPE :",
            f"{legacy_repaired_mm:.3f} mm"
        )

        print(
            "XYZ repaired MPJPE    :",
            f"{xyz_repaired_mm:.3f} mm"
        )

        print(
            "Paper target           :",
            f"{target_mm:.3f} mm"
        )

        print(
            "Difference              :",
            f"{difference_mm:+.3f} mm"
        )


# ============================================================
# 10. FINAL TABLE
# ============================================================

df = pd.DataFrame(results)

print()
print("=" * 90)
print("TABLE I ROW 1 REPRODUCTION")
print("=" * 90)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


reproduced_average = df[
    "Legacy_Repaired_mm"
].mean()

paper_average = df[
    "Target_Table1_mm"
].mean()


print()
print(
    "Reproduced Avg:",
    f"{reproduced_average:.3f} mm"
)

print(
    "Paper Avg     :",
    f"{paper_average:.3f} mm"
)

print(
    "Avg difference:",
    f"{reproduced_average - paper_average:+.3f} mm"
)


# ============================================================
# 11. SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print(
    "Results saved to:",
    OUTPUT_CSV
)