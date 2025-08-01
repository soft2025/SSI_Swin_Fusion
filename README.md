# SSI_Swin_Fusion
Projet de classification FBG avec SSI + Swin Transformer + CBAM

## hawk_data et jeux de données

Les scripts de préparation de données utilisent le paquet `hawk_data` qui fournit
le chargeur du jeu de données FST. Ce paquet n'est pas inclus dans le
répertoire car il est distribué séparément. Pour l'obtenir, clonez le
répertoire correspondant et installez‑le en mode « editable » :

```bash
git clone <URL_to_hawk_data_repo>
pip install -e hawk_data
```

Une fois installé, la classe `hawk_data.FST` sera disponible pour les scripts
et permettra de charger les signaux bruts.

## Génération des segments

`data_preprocessing/segments_generation.py` génère des segments d'une seconde à
partir des signaux FBG et crée un partage train/val/test :

```bash
python data_preprocessing/segments_generation.py \
    --data-dir /chemin/vers/FST \
    --output-dir /chemin/vers/sortie
```

Les fichiers `segments_fbg.pkl` et `split_segments.pkl` sont alors sauvegardés
dans le répertoire spécifié.
