"""
train_table1_row2.py

Table I Row 2 training:
    SIN (MSA-AAGCN)
    Training occlusion = Original + P1-P7
    Dataset = NTU RGB+D 60 Cross-Subject
    Epochs = 80, uninterrupted
    Batch size = 16
    Initial learning rate = 0.1
    Optimizer = SGD
    Momentum = 0.9
    Weight decay = 1e-4
    Nesterov = True
    Warm-up = 5 epochs
    LR milestones = [30, 50]
    Loss = full-coordinate MSE, matching the Row 1 reproduction path

The masking policy itself is implemented inside table1_row2_model.py.

Checkpoints:
    /media/user/HMR/skeleton_action/IMPUTATION/MSAG3D/
    weight_table1_row2/row2_epoch1.pth
    ...
    weight_table1_row2/row2_epoch80.pth
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.Datasets import NTUDataSet
from src.graph import Graph
from table1_row2_model import Model_teacher


# ============================================================
# 1. PATHS
# ============================================================

DATA_ROOT = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "hong_MSG3D/data/"
)

SAVE_DIR = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/weight_table1_row2"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.set_printoptions(
    threshold=np.inf,
    precision=3,
    suppress=True,
    linewidth=800
)


# ============================================================
# 3. TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 80
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 0.1
FRAME = 300
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
LR_MILESTONES = [30, 50]
WARM_UP_EPOCH = 5

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. DATASET
# ============================================================

train_dataset_config = {
    "data_path":
        DATA_ROOT + "xsub/train_data_joint.npy",

    "label_path":
        DATA_ROOT + "xsub/train_label.pkl",

    "use_mmap": True,
    "debug": False,
    "downsample_rate": 1
}

train_dataset = NTUDataSet(
    train_dataset_config
)

# Keep DataLoader behavior aligned with the Row 1 reproduction.
# The historical Row 1 script did not explicitly pass num_workers,
# so PyTorch uses num_workers=0.
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 5. GRAPH
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
# 6. ROW 2 MODEL
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
# 7. LOSS
# ============================================================

criterion_mse = nn.MSELoss().to(device)


# ============================================================
# 8. OPTIMIZER
# ============================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True
)


# ============================================================
# 9. WARM-UP SCHEDULER
#
# Same scheduler behavior as the Row 1 reproduction.
# ============================================================

class GradualWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        total_epoch,
        after_scheduler=None
    ):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [
            base_lr
            * (self.last_epoch + 1)
            / self.total_epoch
            for base_lr in self.base_lrs
        ]

    def step(
        self,
        epoch=None,
        metric=None
    ):
        if (
            self.last_epoch
            >= self.total_epoch - 1
        ):
            if metric is None:
                return self.after_scheduler.step(
                    epoch
                )
            else:
                return self.after_scheduler.step(
                    metric,
                    epoch
                )
        else:
            return super(
                GradualWarmupScheduler,
                self
            ).step(epoch)


lr_scheduler_pre = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=LR_MILESTONES,
    gamma=0.1
)

lr_scheduler = GradualWarmupScheduler(
    optimizer,
    total_epoch=WARM_UP_EPOCH,
    after_scheduler=lr_scheduler_pre
)


# ============================================================
# 10. STANDARD XYZ MPJPE FOR TRAINING LOG ONLY
#
# Tensor shape: [N, C, T, V, M]
# C=3 is the XYZ coordinate dimension.
#
# This metric is NOT used as the optimization loss.
# ============================================================

def xyz_mpjpe(
    predicted,
    target
):
    if predicted.shape != target.shape:
        raise RuntimeError(
            "MPJPE shape mismatch: "
            f"{predicted.shape} vs {target.shape}"
        )

    return torch.mean(
        torch.norm(
            predicted - target,
            dim=1
        )
    )


# ============================================================
# 11. START INFORMATION
# ============================================================

print()
print("=" * 90)
print("TABLE I ROW 2 TRAINING")
print("=" * 90)
print("Method             : SIN (MSA-AAGCN)")
print("Training occlusion : Original + P1-P7")
print("Dataset            : NTU RGB+D 60 X-Sub")
print("Training samples   :", len(train_dataset))
print("Epochs             :", NUM_EPOCHS)
print("Batch size         :", TRAIN_BATCH_SIZE)
print("Initial LR         :", LEARNING_RATE)
print("LR milestones      :", LR_MILESTONES)
print("Warm-up epochs     :", WARM_UP_EPOCH)
print("Optimizer          : SGD")
print("Momentum           :", MOMENTUM)
print("Weight decay       :", WEIGHT_DECAY)
print("Loss               : full-coordinate MSE")
print("Device             :", device)

if torch.cuda.is_available():
    print(
        "GPU                :",
        torch.cuda.get_device_name(0)
    )

print("Save directory     :", SAVE_DIR)
print("=" * 90)
print()


# ============================================================
# 12. TRAINING
# ============================================================

model.train()

for epoch in range(NUM_EPOCHS):

    epoch_number = epoch + 1

    total_loss = 0.0
    total_xyz_error = 0.0
    total_samples = 0

    current_lr = optimizer.param_groups[0]["lr"]

    print()
    print("=" * 90)
    print(
        f"Epoch {epoch_number}/{NUM_EPOCHS}"
    )
    print(
        "lr:",
        current_lr
    )
    print("=" * 90)

    process = tqdm(
        train_dataloader,
        dynamic_ncols=True
    )

    for batch in process:

        # ----------------------------------------------------
        # Load one training batch
        # ----------------------------------------------------

        data = batch[
            "data"
        ].to(device)

        data = data[
            :, :, :FRAME, :, :
        ]

        batch_size = data.size(0)

        # ----------------------------------------------------
        # Row 2 masking is created inside Model_teacher:
        #
        # Original, P1, P2, P3, P4, P5, P6, or P7.
        #
        # No MR0.2-MR0.6 random-joint training condition
        # is used in this Row 2 experiment.
        # ----------------------------------------------------

        mask_data = data.clone()

        model_output = model(
            mask_data,
            epoch
        )

        reconstructed = model_output[0]

        # ----------------------------------------------------
        # Stage-1 optimization
        #
        # Same path as our Row 1 reproduction:
        # full-coordinate MSE on the raw decoder output.
        # ----------------------------------------------------

        loss_mse = criterion_mse(
            reconstructed,
            data
        )

        optimizer.zero_grad()

        loss_mse.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Logging only
        # ----------------------------------------------------

        total_loss += loss_mse.item()

        with torch.no_grad():
            batch_xyz_error = xyz_mpjpe(
                reconstructed,
                data
            )

        total_xyz_error += (
            batch_xyz_error.item()
            * batch_size
        )

        total_samples += batch_size

        process.set_description(
            f"loss: {loss_mse.item():.5f}"
        )

    # ========================================================
    # END OF EPOCH
    # ========================================================

    # Same legacy-style loss normalization as Row 1.
    legacy_epoch_loss = (
        total_loss
        / len(train_dataset)
    )

    # Conventional mean of batch loss values.
    mean_batch_loss = (
        total_loss
        / len(train_dataloader)
    )

    # Raw-decoder training XYZ MPJPE.
    train_xyz_mpjpe_mm = (
        total_xyz_error
        / total_samples
        * 1000.0
    )

    # Same timing as Row 1: scheduler steps after each epoch.
    lr_scheduler.step()

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        SAVE_DIR,
        f"row2_epoch{epoch_number}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print()
    print(
        f"Epoch: {epoch_number}"
    )
    print(
        "Legacy epoch loss:",
        f"{legacy_epoch_loss:.6f}"
    )
    print(
        "Mean batch loss:",
        f"{mean_batch_loss:.6f}"
    )
    print(
        "Train XYZ MPJPE:",
        f"{train_xyz_mpjpe_mm:.3f} mm"
    )
    print(
        "Saved:",
        checkpoint_path
    )


# ============================================================
# 13. FINISHED
# ============================================================

print()
print("=" * 90)
print("ROW 2 TRAINING FINISHED")
print("=" * 90)
print(
    "Created checkpoints:"
)
print(
    "row2_epoch1.pth -> row2_epoch80.pth"
)
print()
print(
    "Directory:"
)
print(
    SAVE_DIR
)
print()
print(
    "Recommended evaluation checkpoints:"
)
print(
    "60, 65, 70, 73, 75, 80"
)
print("=" * 90)
