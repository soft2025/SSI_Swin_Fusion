
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as T

class FusionDataset(Dataset):
    def __init__(self, csv_path, split='train', transform=None, label_map=None):
        '''
        Dataset personnalisé pour charger les données fusionnées Image + SSI pondéré.

        Args:
            csv_path (str): Chemin du fichier CSV fusionné
            split (str): 'train', 'val' ou 'test'
            transform (torchvision.transforms): Transformations appliquées aux images
            label_map (dict): Mapping optionnel des labels -> entiers
        '''
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        self.transform = transform

        if label_map is None:
            classes = sorted(self.data['label'].unique())
            self.label_map = {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Chargement image
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        # Chargement SSI pondéré
        ssi_vector = np.load(row['ssi_path'])
        ssi_vector = torch.tensor(ssi_vector, dtype=torch.float32)

        # Label encodé
        label = torch.tensor(self.label_map[row['label']], dtype=torch.long)

        return image, ssi_vector, label
