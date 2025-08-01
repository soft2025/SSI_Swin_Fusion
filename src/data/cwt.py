"""Generation of CWT images with optional FFDNet denoising."""
from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np
import pywt
import torch
from PIL import Image
from skimage.transform import resize


def generate_cwt_image_ffdnet(
    segment: np.ndarray,
    split: str,
    classe: str,
    test_path: str,
    capteur: str,
    index: int,
    output_base: str,
    model: torch.nn.Module,
    device: torch.device,
    img_size: int = 224,
) -> str:
    """Generate and save a denoised CWT image for a 1‑D segment."""

    scales = np.arange(1, img_size + 1)
    cfs, _ = pywt.cwt(segment, scales, "cmor1.5-1.0")
    img = np.abs(cfs)
    img_resized = resize(img, (img_size, img_size), mode="reflect", anti_aliasing=True)
    img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
    img_tensor = torch.from_numpy(img_norm[None, None, ...]).float().to(device)
    sigma = torch.tensor([[25 / 255.]], dtype=img_tensor.dtype, device=device).view(1, 1, 1, 1)
    with torch.no_grad():
        noise_pred = model(img_tensor, sigma)
        denoised = img_tensor - noise_pred
    out_img = denoised.squeeze().cpu().numpy()
    out_img = np.clip(out_img, 0, 1)
    img_uint8 = (out_img * 255).astype(np.uint8)

    test_id = test_path.replace("/", "_")
    fname = f"segment_{index:03}.png"
    out_dir = os.path.join(output_base, split, classe, test_id, capteur)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    Image.fromarray(img_uint8, mode="L").save(out_path)
    return out_path


def generate_cwt_images(
    segments_fbg: Dict[str, Dict[str, Dict[str, Sequence[np.ndarray]]]],
    split_tags: Dict[str, Dict[str, Dict[str, Sequence[str]]]],
    output_base: str,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Generate CWT images for all segments using ``generate_cwt_image_ffdnet``."""

    for classe, tests in segments_fbg.items():
        for test_path, capteurs in tests.items():
            for capteur, segments in capteurs.items():
                tags = split_tags[classe][test_path][capteur]
                for idx, segment in enumerate(segments):
                    tag = tags[idx]
                    if tag not in {"train", "val", "test"}:
                        continue
                    generate_cwt_image_ffdnet(
                        segment=segment,
                        split=tag,
                        classe=classe,
                        test_path=test_path,
                        capteur=capteur,
                        index=idx,
                        output_base=output_base,
                        model=model,
                        device=device,
                    )
