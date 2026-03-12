"""Video and inertial data loading / preprocessing utilities.

Functions here handle the low-level I/O for each modality:
    * ``sample_video_frames`` — reads an AVI, uniformly samples N frames,
      resizes, normalises with ImageNet stats, and returns a (C, T, H, W) tensor.
    * ``load_inertial`` — reads a MAT file, pads or truncates to a fixed length,
      and returns a (C, T) tensor.
    * ``parse_filename`` — extracts (action, subject, trial) from a filename.
"""

import re
import numpy as np
import cv2
import torch
from scipy.io import loadmat
from pathlib import Path

from src.utils.config import (
    N_FRAMES, IMG_SIZE, INERTIAL_LEN, IMAGENET_MEAN, IMAGENET_STD,
)

# Regex to parse UTD-MHAD filenames like "a1_s2_t3_color.avi"
# Groups: action id, subject id, trial id
FILENAME_RE = re.compile(r"a(\d+)_s(\d+)_t(\d+)_")


def parse_filename(fname: str):
    """Extract (action, subject, trial) ints from a UTD-MHAD filename.

    Returns None if the filename does not match the expected pattern.
    """
    m = FILENAME_RE.match(fname)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def sample_video_frames(
    path: Path,
    n_frames: int = N_FRAMES,
    size: int = IMG_SIZE,
    augment: bool = False,
) -> torch.Tensor:
    """Read an AVI and return a float tensor (C, T, H, W) with uniformly-sampled frames.

    Steps:
        1. Open the video and compute N uniformly-spaced frame indices.
        2. Seek to each index, read the frame, resize to ``size × size``.
        3. Stack into (T, H, W, C), convert to float [0, 1].
        4. Optionally apply horizontal flip augmentation.
        5. Normalise each channel with ImageNet mean/std.
        6. Transpose to (C, T, H, W) which is what R3D-18 expects.
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Guard against corrupted or empty videos
    if total <= 0:
        cap.release()
        return torch.zeros(3, n_frames, size, size)

    # Uniformly-spaced frame indices across the full video duration
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Failed to read — fill with black
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (size, size))
            # OpenCV loads BGR; convert to RGB for consistency with ImageNet stats
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Stack to (T, H, W, C) and scale to [0, 1]
    video = np.stack(frames).astype(np.float32) / 255.0

    # Training-time augmentation: random horizontal flip
    if augment:
        if np.random.rand() < 0.5:
            video = video[:, :, ::-1, :].copy()

    # Normalise per-channel using ImageNet statistics
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    video = (video - mean) / std

    # Rearrange from (T, H, W, C) to (C, T, H, W) — the layout torchvision video models expect
    video = np.transpose(video, (3, 0, 1, 2))
    return torch.from_numpy(video.copy())


def load_inertial(
    path: Path,
    max_len: int = INERTIAL_LEN,
    augment: bool = False,
) -> torch.Tensor:
    """Load a .mat inertial file and return a float tensor (C, T).

    The raw data has shape (time_steps, 6) where the 6 channels are
    [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z].

    Pads with zeros (if shorter than ``max_len``) or truncates (if longer)
    to ensure a fixed temporal dimension for batching.
    """
    mat = loadmat(str(path))

    # The inertial variable is stored under key 'd_iner' or 'd_inertial'
    # depending on the UTD-MHAD version.
    data = mat.get("d_iner", mat.get("d_inertial", None))
    if data is None:
        # Fallback: find the first 2-D numeric array that isn't a MATLAB metadata key
        for k, v in mat.items():
            if not k.startswith("_") and isinstance(v, np.ndarray) and v.ndim == 2:
                data = v
                break
    if data is None:
        # No usable data found — return zeros
        return torch.zeros(6, max_len)

    T, C = data.shape  # (time_steps, 6_channels)
    data = data.astype(np.float32)

    # Training-time augmentation
    if augment:
        # Add small Gaussian noise to simulate sensor imprecision
        data = data + np.random.randn(*data.shape).astype(np.float32) * 0.05
        # Random circular shift along the time axis (up to ±5 samples = ±0.1 s)
        shift = np.random.randint(-5, 6)
        data = np.roll(data, shift, axis=0)

    # Pad or truncate to the target length
    if T >= max_len:
        data = data[:max_len]
    else:
        pad = np.zeros((max_len - T, C), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)

    # Transpose from (T, C) to (C, T) so Conv1d can operate on the time axis
    return torch.from_numpy(data.T.copy())
