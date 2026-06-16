import datetime
from pathlib import Path

import torch

from bh2vec.training import train
from bh2vec.vectors_net import VectorsNetNorm

# === Architecture ===
N_HID = 32      # embedding (hand-vector) dimensionality. Was 8 in the paper.
EMB_DIM = 64    # width of the two embedding hidden layers. Was 32.
PRED_DIM = 256  # width of the two prediction hidden layers. Was 128.
# N_HID = 8      # embedding (hand-vector) dimensionality. Was 8 in the paper.
# EMB_DIM = 32    # width of the two embedding hidden layers. Was 32.
# PRED_DIM = 128  # width of the two prediction hidden layers. Was 128.

# === Optimization ===
EPOCHS = 400
BATCH_SIZE = 10000
LR = 5e-5
WEIGHT_DECAY = 0.0
LR_GAMMA = 0.999  # ExponentialLR decay applied each epoch

# === Data ===
TRAIN_FILE = 'data/vectorsnet_train.csv'
VAL_FILE = 'data/vectorsnet_val.csv'

# === Optional warm start ===
# Path to a .pth state_dict to load before training. Must match (N_HID, EMB_DIM, PRED_DIM).
# Useful e.g. for a "final smoothing" pass: point at runs/<id>/best.pth, lower LR, raise LR_GAMMA.
# Set to None for a fresh run.
# CHECKPOINT = None  # e.g. 'runs/20260604_114620/best.pth'
CHECKPOINT = "runs/20260604_114620/best.pth"

# === Run directory ===
run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = Path('runs') / run_id
run_dir.mkdir(parents=True, exist_ok=True)
print(f'Run directory: {run_dir}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = VectorsNetNorm(n_hid=N_HID, emb_dim=EMB_DIM, pred_dim=PRED_DIM).to(device)

if CHECKPOINT is not None:
    net.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f'Loaded checkpoint: {CHECKPOINT}')

train(
    net,
    TRAIN_FILE,
    VAL_FILE,
    run_dir=run_dir,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    lr_gamma=LR_GAMMA,
)
# Per-epoch checkpoints are written by `train` itself:
#   runs/<run_id>/last.pth  - state after the most recent epoch
#   runs/<run_id>/best.pth  - state with the lowest val MSE so far
