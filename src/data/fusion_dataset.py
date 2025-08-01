"""PyTorch Dataset for the fusion of CWT images and weighted SSI vectors."""
from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Default transform resizing the image to 224x224 and applying ImageNet
# normalization. This matches the expectations of the Swin Transformer
# backbone used in the model.
default_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class FusionDataset(Dataset):
    """Dataset loading pairs of CWT images and weighted SSI vectors.

    The ``csv_path`` file must contain the following columns:

    - ``image_path``: path to the CWT image file
    - ``ssi_path``: path to the weighted SSI vector (``.npy``)
    - ``label``: class label as a string
    - ``split``: data split tag (``"train"``, ``"val"`` or ``"test"``)

    Example
    -------
    >>> from torchvision import transforms
    >>> ds = FusionDataset("metadata.csv", split="train",
    ...                    transform=transforms.ToTensor())
    >>> img, vec, label = ds[0]
    """

    def __init__(self, csv_path: str, split: str = "train", transform=None, label_map=None):
        self.csv_path = csv_path
        self.split = split
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        if transform is None:
            transform = default_transform
        self.transform = transform

        if label_map is None:
            classes = sorted(self.data["label"].unique())
            self.label_map = {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.label_map = label_map

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        ssi_vector = torch.tensor(np.load(row["ssi_path"]), dtype=torch.float32)
        label = torch.tensor(self.label_map[row["label"]], dtype=torch.long)
        return image, ssi_vector, label

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"{self.__class__.__name__}(split='{self.split}', "
            f"num_samples={len(self)})"
        )
