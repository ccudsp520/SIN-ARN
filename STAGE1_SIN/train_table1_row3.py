"""
train_table1_row3.py

Table I Row 3 training:
    SIN (MSA-AAGCN)
    Training = Mixed:
        Original + P1-P7 + MR0.2-MR0.6

Dataset:
    NTU RGB+D 60 Cross-Subject

Training:
    80 epochs, uninterrupted
    batch size = 16
    initial LR = 0.1
    SGD, momentum = 0.9
    weight decay = 1e-4
    Nesterov = True
    warm-up = 5 epochs
    milestones = [30, 50]
    loss = full-coordinate MSE

The mixed 13-condition masking policy is implemented in table1_row3_model.py.
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
from table1_row3_model import Model_teacher


# ============================================================
# 1. PATHS
# ============================================================

DATA_ROOT = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "hong_MSG3D/data/"
)

SAVE_DIR = (
    "/media/user/HMR/skeleton_action/IMPUTATION/"
    "MSAG3D/weight_table1_row3"
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


# ============================================================
# 3. SETTINGS
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
# 6. MODEL
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
# 9. WARM-UP + MULTISTEP SCHEDULER
#
# Same behavior as Rows 1 and 2.
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

        super().__init__(
            optimizer
        )

    def get_lr(self):

        return [
            base_lr
            * (self.last_epoch + 1)
            / self.total_epoch

            for base_lr
            in self.base_lrs
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
# ============================================================

def xyz_mpjpe(
    predicted,
    target
):

    if predicted.shape != target.shape:

        raise RuntimeError(
            f"Shape mismatch: "
            f"{predicted.shape} vs {target.shape}"
        )

    return torch.mean(
        torch.norm(
            predicted - target,
            dim=1
        )
    )


# ============================================================
# 11. START INFO
# ============================================================

print()
print("=" * 90)
print("TABLE I ROW 3 TRAINING")
print("=" * 90)
print("Method             : SIN (MSA-AAGCN)")
print("Training occlusion : Mixed")
print("Conditions         : Original + P1-P7 + MR0.2-MR0.6")
print("Sampling policy    : 13-condition permutation")
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
# 12. TRAIN
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

        data = batch[
            "data"
        ].to(device)

        data = data[
            :, :, :FRAME, :, :
        ]

        batch_size = data.size(0)

        # table1_row3_model.py creates the mixed 13-condition
        # training mask internally.
        mask_data = data.clone()

        model_output = model(
            mask_data,
            epoch
        )

        reconstructed = model_output[0]

        loss_mse = criterion_mse(
            reconstructed,
            data
        )

        optimizer.zero_grad()

        loss_mse.backward()

        optimizer.step()

        total_loss += (
            loss_mse.item()
        )

        with torch.no_grad():

            batch_xyz_error = xyz_mpjpe(
                reconstructed,
                data
            )

        total_xyz_error += (
            batch_xyz_error.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        process.set_description(
            f"loss: {loss_mse.item():.5f}"
        )

    # --------------------------------------------------------
    # End epoch
    # --------------------------------------------------------

    legacy_epoch_loss = (
        total_loss
        / len(train_dataset)
    )

    mean_batch_loss = (
        total_loss
        / len(train_dataloader)
    )

    train_xyz_mpjpe_mm = (
        total_xyz_error
        / total_samples
        * 1000.0
    )

    lr_scheduler.step()

    checkpoint_path = os.path.join(
        SAVE_DIR,
        f"row3_epoch{epoch_number}.pth"
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
print("ROW 3 TRAINING FINISHED")
print("=" * 90)
print(
    "Created checkpoints:"
)
print(
    "row3_epoch1.pth -> row3_epoch80.pth"
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
