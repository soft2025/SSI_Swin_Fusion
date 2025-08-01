"""Generation of SSI vectors and weighting utilities."""
from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import signal
from scipy.linalg import svd


def generate_ssi_vector(
    segment: np.ndarray,
    split: str,
    classe: str,
    test_path: str,
    capteur: str,
    index: int,
    output_base: str,
    fs: int = 2048,
    decim: int = 4,
    lags: int = 40,
    ordre: int = 20,
    top_k: int = 10,
) -> str:
    """Compute the SSI frequencies of a segment and save them."""

    segment_decim = signal.decimate(segment, decim, ftype="fir")
    fs_eff = fs // decim
    N = len(segment_decim)
    hankel_matrix = np.zeros((lags, N - lags))
    for i in range(lags):
        hankel_matrix[i] = segment_decim[i : N - lags + i]
    U, S, Vh = svd(hankel_matrix, full_matrices=False)
    U_r = U[:, :ordre]
    S_r = np.diag(S[:ordre])
    V_r = Vh[:ordre, :]
    X = S_r @ V_r
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    A_hat = X2 @ np.linalg.pinv(X1)
    eigvals = np.linalg.eigvals(A_hat)
    freqs = np.abs(np.angle(eigvals) * fs_eff / (2 * np.pi))
    freqs = np.sort(freqs)[:top_k]

    test_id = test_path.replace("/", "_")
    fname = f"segment_{index:03}.npy"
    out_dir = os.path.join(output_base, split, classe, test_id, capteur)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    np.save(out_path, freqs.astype(np.float32))
    return out_path


def load_weights(table_path: str) -> Dict[tuple, float]:
    """Load the weighting table produced during SSI analysis."""

    df = pd.read_csv(table_path)
    df_long = df.melt(id_vars=["Capteur"], var_name="Ordre_SSI", value_name="Ecart_moyen_Hz")
    df_long["Ordre_SSI"] = df_long["Ordre_SSI"].astype(int)
    max_val = df_long["Ecart_moyen_Hz"].max()
    df_long["Poids"] = df_long["Ecart_moyen_Hz"] / max_val
    return {(row["Capteur"], int(row["Ordre_SSI"])): row["Poids"] for _, row in df_long.iterrows()}


def appliquer_poids(
    freqs: Sequence[float],
    capteur: str,
    weights: Dict[tuple, float],
    orders_all: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply weighting coefficients to an SSI vector."""

    if orders_all is None:
        orders_all = np.arange(2, 41, 2)
    freqs_pond = []
    n = len(freqs)
    for i in range(n):
        ordre = orders_all[i] if i < len(orders_all) else None
        w = weights.get((capteur, int(ordre)), 0.0) if ordre is not None else 0.0
        freqs_pond.append(freqs[i] * w)
    return np.array(freqs_pond)


def apply_weights_to_directory(
    base_dir: str,
    weights: Dict[tuple, float],
    orders_all: Sequence[int] | None = None,
    output_suffix: str = "_weighted",
) -> None:
    """Apply weighting to every SSI vector contained in ``base_dir``."""

    if orders_all is None:
        orders_all = np.arange(2, 41, 2)
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.endswith(".npy") and not file.endswith(f"{output_suffix}.npy"):
                    file_path = os.path.join(root, file)
                    capteur = os.path.basename(os.path.dirname(file_path))
                    try:
                        freqs = np.load(file_path)
                        freqs_w = appliquer_poids(freqs, capteur, weights, orders_all)
                        out_path = file_path.replace(".npy", f"{output_suffix}.npy")
                        np.save(out_path, freqs_w)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
