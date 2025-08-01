"""Utilities for splitting FBG time series into 1 s segments and generating
train/val/test splits."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np


def segment_fbg(
    data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    tests_per_class: Dict[str, Sequence[str]],
    fbg_caps: Sequence[str] | None = None,
    start_frac: float = 0.05,
    end_frac: float = 0.95,
    fs: int = 2048,
    segment_duration: float = 1.0,
) -> Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]:
    """Cut raw FBG signals into 1‑second segments.

    Parameters
    ----------
    data : dict
        Nested dict ``data[test_id][test_number][capteur]['strain']`` storing
        raw strain signals as ``[time, channel]`` arrays.
    tests_per_class : dict
        Mapping ``classe -> list[test_id/test_number]``.
    fbg_caps : list, optional
        List of FBG sensor names. If ``None`` FB1..FB10 will be used.
    start_frac, end_frac : float
        Fraction of the signal to keep before splitting.
    fs : int
        Sampling frequency in Hz.
    segment_duration : float
        Duration of each segment in seconds (default 1s).

    Returns
    -------
    dict
        ``segments_fbg[classe][test_path][capteur] = list[np.ndarray]``.
    """

    if fbg_caps is None:
        fbg_caps = [f"SW_FB{i}" for i in range(1, 11)]

    segment_length = int(fs * segment_duration)
    start_idx = int(start_frac * fs * segment_duration * 80)  # 80 s traces
    end_idx = int(end_frac * fs * segment_duration * 80)
    usable_length = end_idx - start_idx
    n_segments = usable_length // segment_length

    segments_fbg: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {}

    for classe, test_list in tests_per_class.items():
        segments_fbg[classe] = {}
        for test_path in test_list:
            test_id, test_number = test_path.split("/")
            segments_fbg[classe][test_path] = {}
            for capteur in fbg_caps:
                strain = data[test_id][test_number][capteur]["strain"][:]
                signal = np.mean(strain, axis=1)
                signal = signal - np.mean(signal)
                signal_util = signal[start_idx:end_idx]
                segments = [
                    signal_util[i * segment_length : (i + 1) * segment_length]
                    for i in range(n_segments)
                ]
                segments_fbg[classe][test_path][capteur] = segments
    return segments_fbg


def split_segments(
    segments_fbg: Dict[str, Dict[str, Dict[str, Sequence[np.ndarray]]]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int | None = 42,
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Generate a train/val/test split for each segment.

    The returned dictionary mirrors ``segments_fbg`` but contains tags
    ``"train"``, ``"val"`` or ``"test"`` for every segment index.
    """

    if seed is not None:
        random.seed(seed)

    split_segments: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    for classe, tests in segments_fbg.items():
        split_segments[classe] = {}
        for test_path, capteurs in tests.items():
            split_segments[classe][test_path] = {}
            for capteur, segments in capteurs.items():
                n = len(segments)
                indices = list(range(n))
                random.shuffle(indices)
                n_train = int(train_ratio * n)
                n_val = int(val_ratio * n)
                n_test = n - n_train - n_val
                tags = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
                tag_array = [None] * n
                for idx, tag in zip(indices, tags):
                    tag_array[idx] = tag
                split_segments[classe][test_path][capteur] = tag_array

    return split_segments
