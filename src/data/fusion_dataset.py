import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class FusionDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split="train",
        label_map=None,
        use_ssi=True,
        transform=None,
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.use_ssi = use_ssi
        self.transform = transform

        if label_map is None:
            self.label_map = {
                label: idx
                for idx, label in enumerate(sorted(self.df["label"].unique()))
            }
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = (
                torch.tensor(np.array(img), dtype=torch.float32)
                .permute(2, 0, 1)
                / 255.0
            )

        label = torch.tensor(self.label_map[row["label"]], dtype=torch.long)

        if self.use_ssi:
            ssi = np.load(row["ssi_path"])
            ssi = torch.tensor(ssi, dtype=torch.float32)
            return img, ssi, label
        else:
            return img, label
