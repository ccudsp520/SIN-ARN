import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.Datasets import NTUDataSet
from src.graph import Graph
from table1_row1_model import Model_teacher


# ============================================================
# IMPORTANT
# ============================================================
# This is an APPROXIMATE continuation from epoch 60 to epoch 80.
#
# The original training script saved only model.state_dict().
# It did NOT save:
#   - optimizer.state_dict()
#   - LR scheduler state
#   - RNG state
#
# Therefore epoch 61-80 here are NOT a mathematically exact
# continuation of the original epoch 1-60 run.
#
# We preserve:
#   - the epoch-60 model weights
#   - the same SIN architecture
#   - the same dataset
#   - the same batch size
#   - the same SGD hyperparameters
#   - the LR that was active after the second decay: 0.001
#
# Use this run only as a diagnostic extended-training experiment.
# Keep row1_epoch60.pth as the official 60-epoch reproduction.
# ============================================================


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = "/media/user/HMR/skeleton_action/IMPUTATION/MSAG3D"

DATA_ROOT = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "hong_MSG3D/data/"
)

CHECKPOINT_60 = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/weight_table1_row1/row1_epoch60.pth"
)

SAVE_DIR = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/weight_table1_row1_extended"
)

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

START_EPOCH = 60       # checkpoint already represents epoch 60
END_EPOCH = 80         # train epochs 61 ... 80
TRAIN_BATCH_SIZE = 16
FRAME = 300

# Epoch 60 training had already passed both LR milestones [30, 50].
# The active LR at epochs 55-60 was 0.001.
CONTINUE_LR = 0.001

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001


# ============================================================
# 2. REPRODUCIBILITY FOR THIS EXTENDED RUN
# ============================================================
# This makes THIS diagnostic run reproducible.
# It cannot recover the exact RNG state after the original epoch 60.

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DATASET
# ============================================================

train_dataset_config = {
    "data_path":
        DATA_ROOT + "xsub/train_data_joint.npy",

    "label_path":
        DATA_ROOT + "xsub/train_label.pkl",

    "use_mmap": True,
    "debug": False,
    "downsample_rate": 1,
}

train_dataset = NTUDataSet(
    train_dataset_config
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 4. GRAPH
#    Same graph configuration as Row 1 training
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
# 5. BUILD THE SAME ROW 1 SIN MODEL
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


# ============================================================
# 6. LOAD EPOCH 60 WEIGHTS
# ============================================================

if not os.path.exists(CHECKPOINT_60):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_60}"
    )

print("=" * 80)
print("TABLE I ROW 1 EXTENDED TRAINING")
print("=" * 80)
print("Mode               : approximate continuation")
print("Load checkpoint    :", CHECKPOINT_60)
print("Train epochs       : 61 -> 80")
print("Batch size         :", TRAIN_BATCH_SIZE)
print("Learning rate      :", CONTINUE_LR)
print("Device             :", device)
print("Save directory     :", SAVE_DIR)
print("=" * 80)

state_dict = torch.load(
    CHECKPOINT_60,
    map_location=device
)

model.load_state_dict(
    state_dict,
    strict=True
)

print("Epoch-60 checkpoint loaded successfully.")


# ============================================================
# 7. LOSS AND OPTIMIZER
# ============================================================
# Same Stage-1 objective as the Row 1 reproduction:
# MSE only.
#
# Optimizer momentum state is necessarily reset because the original
# training script did not save optimizer.state_dict().

criterion_mse = nn.MSELoss().to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=CONTINUE_LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True
)


# ============================================================
# 8. EXTENDED TRAINING: EPOCH 61-80
# ============================================================

model.train()

for epoch_num in range(START_EPOCH + 1, END_EPOCH + 1):

    total_loss = 0.0

    process = tqdm(
        train_dataloader,
        dynamic_ncols=True
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print()
    print(
        f"Epoch {epoch_num}/{END_EPOCH} | "
        f"lr: {current_lr}"
    )

    for batch in process:

        data = batch["data"].to(device)

        data = data[
            :, :, :FRAME, :, :
        ]

        # Model_teacher creates the same Row 1 training masking:
        # Original + MR 0.2-0.6
        mask_data = data.clone()

        model_output = model(
            mask_data,
            epoch_num - 1
        )

        reconstructed = model_output[0]

        loss_mse = criterion_mse(
            reconstructed,
            data
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_mse.backward()

        optimizer.step()

        total_loss += loss_mse.item()

        process.set_description(
            f"loss: {loss_mse.item():.5f}"
        )


    # Keep the same legacy epoch-loss reporting style used by
    # train_table1_row1.py so logs remain comparable.
    epoch_loss_legacy = (
        total_loss / len(train_dataset)
    )

    # Also report the ordinary mean batch loss, which is easier to interpret.
    mean_batch_loss = (
        total_loss / len(train_dataloader)
    )

    checkpoint_path = os.path.join(
        SAVE_DIR,
        f"row1_epoch{epoch_num}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(
        f"Epoch: {epoch_num}, "
        f"Legacy Loss: {epoch_loss_legacy:.6f}, "
        f"Mean Batch Loss: {mean_batch_loss:.6f}"
    )

    print(
        "Saved:",
        checkpoint_path
    )


print()
print("=" * 80)
print("EXTENDED TRAINING FINISHED")
print("=" * 80)
print(
    f"Created checkpoints row1_epoch61.pth "
    f"through row1_epoch80.pth"
)
print("Directory:", SAVE_DIR)
print()
print(
    "Reminder: these epoch 61-80 checkpoints are diagnostic "
    "approximate-continuation checkpoints, not the official "
    "60-epoch Row 1 reproduction."
)
