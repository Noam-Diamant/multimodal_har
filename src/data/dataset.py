"""PyTorch Dataset and DataLoader factory for UTD-MHAD.

``UTDMHADDataset`` scans the RGB/ and Inertial/ directories, filters by
action and subject, and lazily loads each sample on ``__getitem__``.
``get_dataloaders`` is a convenience wrapper that creates the subject-based
train/val/test split and returns ready-to-use DataLoaders.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import (
    RGB_DIR, INERTIAL_DIR, SUBSET_ACTIONS, ACTION_TO_IDX,
    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS,
    BATCH_SIZE, N_FRAMES, IMG_SIZE, INERTIAL_LEN, SEED,
)
from src.data.preprocessing import parse_filename, sample_video_frames, load_inertial


class UTDMHADDataset(Dataset):
    """Multimodal dataset that returns (video, inertial, label) tuples.

    At construction time it builds an index of available files; actual data
    loading happens lazily in ``__getitem__`` to keep memory usage low.

    Parameters
    ----------
    actions : list of int
        Action IDs to include (e.g. ``[4, 7, 13, 22, 24, 27]``).
    subjects : list of int or None
        Subject IDs to include. ``None`` means all subjects.
    augment : bool
        Whether to apply data augmentation (flips, noise, etc.).
    modality : str
        ``"both"``, ``"video"``, or ``"inertial"`` — which tensors to load.
        Unloaded modalities are returned as zero tensors so that the output
        shape stays the same regardless of the mode (simplifies DataLoader).
    """

    def __init__(
        self,
        actions: List[int] = SUBSET_ACTIONS,
        subjects: Optional[List[int]] = None,
        augment: bool = False,
        modality: str = "both",
    ):
        super().__init__()
        self.augment = augment
        self.modality = modality

        # Build a lookup: (action, subject, trial) → file path
        rgb_index = {}
        for f in sorted(RGB_DIR.glob("*.avi")):
            key = parse_filename(f.name)
            if key:
                rgb_index[key] = f

        inertial_index = {}
        for f in sorted(INERTIAL_DIR.glob("*.mat")):
            key = parse_filename(f.name)
            if key:
                inertial_index[key] = f

        # Merge both indices and filter by action / subject
        self.samples: List[Tuple[Optional[Path], Optional[Path], int, tuple]] = []
        all_keys = sorted(set(rgb_index) | set(inertial_index))
        for key in all_keys:
            action, subject, trial = key
            if action not in actions:
                continue
            if subjects is not None and subject not in subjects:
                continue
            rgb_path = rgb_index.get(key)
            iner_path = inertial_index.get(key)
            label = ACTION_TO_IDX[action]  # 0-based class index
            self.samples.append((rgb_path, iner_path, label, key))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, iner_path, label, key = self.samples[idx]

        # Load video tensor — or return zeros if modality not requested / file missing
        if self.modality in ("both", "video") and rgb_path is not None:
            video = sample_video_frames(rgb_path, augment=self.augment)
        else:
            video = torch.zeros(3, N_FRAMES, IMG_SIZE, IMG_SIZE)

        # Load inertial tensor — same logic
        if self.modality in ("both", "inertial") and iner_path is not None:
            inertial = load_inertial(iner_path, augment=self.augment)
        else:
            inertial = torch.zeros(6, INERTIAL_LEN)

        return video, inertial, label


def get_dataloaders(
    actions: List[int] = SUBSET_ACTIONS,
    batch_size: int = BATCH_SIZE,
    modality: str = "both",
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders with subject-based split.

    Split (leave-subjects-out):
        * **Train** (subjects 1–4, 50 %) — with augmentation
        * **Val**   (subjects 5–6, 25 %) — no augmentation, used for model selection
        * **Test**  (subjects 7–8, 25 %) — no augmentation, held-out final evaluation

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds = UTDMHADDataset(
        actions=actions, subjects=TRAIN_SUBJECTS,
        augment=True, modality=modality,
    )
    val_ds = UTDMHADDataset(
        actions=actions, subjects=VAL_SUBJECTS,
        augment=False, modality=modality,
    )
    test_ds = UTDMHADDataset(
        actions=actions, subjects=TEST_SUBJECTS,
        augment=False, modality=modality,
    )

    # Seeded generator for reproducible shuffling across runs
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
