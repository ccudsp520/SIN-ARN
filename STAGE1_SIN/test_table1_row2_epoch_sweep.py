import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.Datasets import NTUDataSet
from src.graph import Graph
from table1_row2_model import Model_teacher

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_ROOT = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "hong_MSG3D/data/"
)

WEIGHT_DIR = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/weight_table1_row2"
)

OUTPUT_CSV = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/table1_row2_epoch_sweep.csv"
)

TEST_BATCH_SIZE = 24
FRAME = 300
EPOCHS = [60, 65, 70, 73, 75, 80]

val_dataset_config = {
    "data_path": DATA_ROOT + "xsub/val_data_joint.npy",
    "label_path": DATA_ROOT + "xsub/val_label.pkl",
    "use_mmap": True,
    "debug": False
}

test_dataset = NTUDataSet(val_dataset_config)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

gcn_kernel_size = [5, 2]
graph = Graph(max_hop=gcn_kernel_size[1])

A = torch.tensor(
    graph.A,
    dtype=torch.float32,
    requires_grad=False
).to(device)

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

parts = {
    1: np.array([5, 6, 7, 8, 22, 23]) - 1,
    2: np.array([9, 10, 11, 12, 24, 25]) - 1,
    3: np.array([22, 23, 24, 25]) - 1,
    4: np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1,
    5: np.array([1, 2, 3, 4, 21]) - 1,
    6: np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23]) - 1,
    7: np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25]) - 1
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
    ("MR0.6", 0.6)
]

paper_target = {
    "P1": 31.8,
    "P2": 33.3,
    "P3": 11.1,
    "P4": 34.8,
    "P5": 10.4,
    "P6": 46.2,
    "P7": 47.4,
    "MR0.2": 24.8,
    "MR0.3": 38.5,
    "MR0.4": 63.0,
    "MR0.5": 82.3,
    "MR0.6": 112.2
}

condition_seed = {
    "MR0.2": 200,
    "MR0.3": 300,
    "MR0.4": 400,
    "MR0.5": 500,
    "MR0.6": 600
}

def apply_occlusion(data, occ):
    N, C, T, V, M = data.shape

    mask_np = np.ones(
        (N, C, T, V, M),
        dtype=np.float32
    )

    if isinstance(occ, int):
        missing_joints = parts[occ]
        mask_np[:, :, :, missing_joints, :] = 0.0
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

def xyz_mpjpe(predicted, target):
    return torch.mean(
        torch.norm(
            predicted - target,
            dim=1
        )
    )

def sin_forward(masked_data):
    gcn_feature = model.full_body(masked_data)
    output = model.Decoder(gcn_feature)
    return output

print()
print("=" * 100)
print("TABLE I ROW 2 EPOCH SWEEP")
print("=" * 100)
print("Method             : SIN (MSA-AAGCN)")
print("Training           : Original + P1-P7")
print("Dataset            : NTU RGB+D 60 X-Sub")
print("Validation samples :", len(test_dataset))
print("Epochs             :", EPOCHS)
print("Device             :", device)
print("=" * 100)

all_results = []

for epoch_num in EPOCHS:
    checkpoint = os.path.join(
        WEIGHT_DIR,
        f"row2_epoch{epoch_num}.pth"
    )

    print()
    print("#" * 100)
    print("EPOCH:", epoch_num)
    print("Checkpoint:", checkpoint)
    print("#" * 100)

    if not os.path.exists(checkpoint):
        print("ERROR: checkpoint not found.")
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
        print(f"Epoch {epoch_num} | Testing {condition_name}")

        if condition_name in condition_seed:
            seed = condition_seed[condition_name]
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
                data = data[:, :, :FRAME, :, :]

                batch_size = data.size(0)

                masked_data, x_mask = apply_occlusion(
                    data,
                    occ
                )

                output = sin_forward(masked_data)

                repaired = (
                    output * (1.0 - x_mask)
                    +
                    data * x_mask
                )

                error = xyz_mpjpe(
                    repaired,
                    data
                )

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

        target = paper_target[condition_name]
        difference = mpjpe_mm - target

        row = {
            "Epoch": epoch_num,
            "Condition": condition_name,
            "XYZ_MPJPE_mm": mpjpe_mm,
            "Paper_mm": target,
            "Difference_mm": difference
        }

        all_results.append(row)
        epoch_results.append(row)

        print(f"XYZ MPJPE : {mpjpe_mm:.3f} mm")
        print(f"Paper     : {target:.3f} mm")
        print(f"Difference: {difference:+.3f} mm")

    fixed_values = [
        item["XYZ_MPJPE_mm"]
        for item in epoch_results
        if item["Condition"].startswith("P")
    ]

    random_values = [
        item["XYZ_MPJPE_mm"]
        for item in epoch_results
        if item["Condition"].startswith("MR")
    ]

    all_values = [
        item["XYZ_MPJPE_mm"]
        for item in epoch_results
    ]

    differences = [
        item["Difference_mm"]
        for item in epoch_results
    ]

    fixed_avg = np.mean(fixed_values)
    random_avg = np.mean(random_values)
    overall_avg = np.mean(all_values)
    mad = np.mean(np.abs(differences))
    paper_avg = np.mean(list(paper_target.values()))

    print()
    print("-" * 100)
    print(
        f"Epoch {epoch_num} Fixed-Part Avg : "
        f"{fixed_avg:.3f} mm"
    )
    print(
        f"Epoch {epoch_num} Random-Joint Avg: "
        f"{random_avg:.3f} mm"
    )
    print(
        f"Epoch {epoch_num} Overall Avg    : "
        f"{overall_avg:.3f} mm"
    )
    print(
        f"Paper Overall Avg                : "
        f"{paper_avg:.3f} mm"
    )
    print(
        f"Delta Overall                    : "
        f"{overall_avg - paper_avg:+.3f} mm"
    )
    print(
        f"Mean Absolute Difference         : "
        f"{mad:.3f} mm"
    )
    print("-" * 100)

df = pd.DataFrame(all_results)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

summary_rows = []

for epoch_num in sorted(df["Epoch"].unique()):
    epoch_df = df[df["Epoch"] == epoch_num]

    fixed_df = epoch_df[
        epoch_df["Condition"].str.startswith("P")
    ]

    random_df = epoch_df[
        epoch_df["Condition"].str.startswith("MR")
    ]

    summary_rows.append({
        "Epoch":
            epoch_num,

        "Fixed_Avg_mm":
            fixed_df["XYZ_MPJPE_mm"].mean(),

        "Random_Avg_mm":
            random_df["XYZ_MPJPE_mm"].mean(),

        "Overall_Avg_mm":
            epoch_df["XYZ_MPJPE_mm"].mean(),

        "Delta_from_Paper_Avg_mm":
            (
                epoch_df["XYZ_MPJPE_mm"].mean()
                -
                np.mean(list(paper_target.values()))
            ),

        "Mean_Absolute_Difference_mm":
            np.mean(
                np.abs(
                    epoch_df["Difference_mm"]
                )
            )
    })

summary = pd.DataFrame(summary_rows)

paper_average = np.mean(
    list(paper_target.values())
)

print()
print("=" * 110)
print("FINAL ROW 2 EPOCH SWEEP SUMMARY")
print("=" * 110)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print()
print(
    "Paper Row 2 Avg:",
    f"{paper_average:.3f} mm"
)

print()
print(
    "Full results saved to:"
)
print(OUTPUT_CSV)
print("=" * 110)
