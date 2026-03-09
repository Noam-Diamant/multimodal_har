"""Central configuration for the multimodal HAR project.

Every tunable constant and path lives here so that changes propagate
automatically to all modules (data, models, training, notebooks).
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# PROJECT_ROOT resolves to multimodal_har/ regardless of the caller's cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RGB_DIR = DATA_DIR / "RGB"           # 861 AVI videos
INERTIAL_DIR = DATA_DIR / "Inertial" # 861 MAT files (accel + gyro)

# Directory where trained model checkpoints and grid-search results are stored
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# ── Class selection (15 classes: 9 wrist + 6 thigh) ───────────────────────────
# Chosen to span diverse motion types and both sensor placements.
# SUBSET_ACTIONS = [1, 3, 4, 5, 7, 12, 13, 14, 19, 22, 23, 24, 25, 26, 27]
SUBSET_ACTIONS = [1, 3, 4, 7, 13, 19, 22, 23, 24, 27]

# Full mapping of all 27 actions in UTD-MHAD
ACTION_NAMES = {
    1: "Swipe Left", 2: "Swipe Right", 3: "Wave", 4: "Clap",
    5: "Throw", 6: "Arm Cross", 7: "Basketball Shoot",
    8: "Draw X", 9: "Draw Circle CW", 10: "Draw Circle CCW",
    11: "Draw Triangle", 12: "Bowling", 13: "Boxing",
    14: "Baseball Swing", 15: "Tennis Swing", 16: "Arm Curl",
    17: "Tennis Serve", 18: "Push", 19: "Knock",
    20: "Catch", 21: "Pickup & Throw", 22: "Jog", 23: "Walk",
    24: "Sit to Stand", 25: "Stand to Sit",
    26: "Lunge", 27: "Squat",
}

# Maps original 1-based action IDs to contiguous 0-based class indices
# so that labels can be used directly with nn.CrossEntropyLoss.
ACTION_TO_IDX = {a: i for i, a in enumerate(SUBSET_ACTIONS)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}
NUM_CLASSES = len(SUBSET_ACTIONS)
# Human-readable ordered list of class names (index-aligned)
CLASS_NAMES = [ACTION_NAMES[a] for a in SUBSET_ACTIONS]

# Actions 1–21 use a wrist-mounted sensor; 22–27 use a thigh-mounted sensor
SENSOR_PLACEMENT = {i: "wrist" for i in range(1, 22)}
SENSOR_PLACEMENT.update({i: "thigh" for i in range(22, 28)})

# ── Data parameters ────────────────────────────────────────────────────────────
N_FRAMES = 16           # number of uniformly-sampled video frames per clip
IMG_SIZE = 112          # spatial resolution (pixels) for resized video frames
INERTIAL_LEN = 150      # fixed inertial sequence length (3 s × 50 Hz)
INERTIAL_CHANNELS = 6   # 3-axis accelerometer + 3-axis gyroscope
INERTIAL_SAMPLE_RATE = 50  # Hz

# ── Train / val / test split (leave-subjects-out) ─────────────────────────────
# 50 % train, 25 % val, 25 % test.  Subjects are kept disjoint so no person
# appears in more than one split (prevents data leakage).
TRAIN_SUBJECTS = [1, 2, 3, 4]   # 50 %
VAL_SUBJECTS = [5, 6]            # 25 % — used for model selection / grid search
TEST_SUBJECTS = [7, 8]           # 25 % — held-out final evaluation only

# ── ImageNet normalisation (used for pretrained video backbone) ────────────────
# R3D-18 was pretrained on Kinetics-400 with these channel-wise stats.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Default training hyper-parameters ──────────────────────────────────────────
LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 20
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
EPOCHS = 30

# ── Feature dimensions coming out of each backbone ────────────────────────────
INERTIAL_FEAT_DIM = 128   # output of InertialCNN.extract_features()
VIDEO_FEAT_DIM = 512      # R3D-18 produces 512-d after global avg-pool

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
