# SSI_Swin_Fusion


This project performs classification of FBG measurements using a fusion of CWT images and SSI vectors with a Swin Transformer backbone.

## Dataset preparation

The raw `FST` dataset (not provided here) must first be segmented and split into train/val/test sets. The preprocessing utilities found in `data_preprocessing/` wrap the functions of `src.data` and expose simple CLIs.

### 1. Generate segments and split
```bash
python data_preprocessing/segments_generation.py \
    --data-dir /path/to/FST \
    --output-dir /path/to/preprocessed
```
This creates `segments_fbg.pkl` and `split_segments.pkl` in the given output directory.

### 2. Generate CWT images
```bash
python data_preprocessing/generate_cwt_images.py \
    --segments-path /path/to/preprocessed/segments_fbg.pkl \
    --split-path /path/to/preprocessed/split_segments.pkl \
    --output-dir /path/to/CWT_images \
    --model-path /path/to/ffdnet.pth \
    --device cuda
```
`--model-path` can be omitted to use the pretrained FFDNet available in `timm`.

### 3. Generate SSI vectors
```bash
python data_preprocessing/generate_ssi_vectors.py \
    --segments-path /path/to/preprocessed/segments_fbg.pkl \
    --split-path /path/to/preprocessed/split_segments.pkl \
    --output-dir /path/to/SSI_vectors \
    --decim 4 --lags 40 --order 20 --top-k 10
```
SSI frequencies are saved for every segment in `SSI_vectors/train|val|test`.

### 4. Create the fusion CSV
```bash
python data_preprocessing/create_fusion_csv.py \
    --cwt-dir /path/to/CWT_images \
    --ssi-dir /path/to/SSI_vectors \
    --weights-csv /path/to/weights.csv \
    --output-csv /path/to/dataset_fusion_weighted.csv
```
If `--weights-csv` is provided, weighting coefficients are applied to all SSI vectors before creating the metadata CSV.

## Training and evaluation

Training is handled by `src/train.py`. It loads the fusion CSV, trains a small model combining a Swin Transformer with an SSI branch and reports validation accuracy after every epoch.

Example command:
```bash
python -m src.train \
    --csv-path /path/to/dataset_fusion_weighted.csv \
    --output-dir runs \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda
```
The trained weights are saved to `runs/fusion_model.pt`. The script prints the validation loss and accuracy so it can be used for basic evaluation.

