# Brain Tumor Detection from Hyperspectral Images (HybridSN-style CNN)

Pixel-wise classification of in-vivo human brain hyperspectral images (HSI) into
**normal tissue (NT)**, **tumor tissue (TT)**, **blood vessel (BV)** and **background (BG)**,
using a patch-based 3D+2D CNN implemented in PyTorch.

## Dataset
HELICoiD in-vivo hyperspectral brain database (400-1000 nm, 826 bands, pushbroom camera).
The data is **not included** in this repo; request it from the authors:
<https://hsibraindatabase.iuma.ulpgc.es/>

Reference: Leon et al., *Hyperspectral imaging benchmark based on machine learning for
intraoperative brain tumour detection*, npj Precision Oncology 7, 119 (2023).
https://doi.org/10.1038/s41698-023-00475-9

Expected layout: one folder per image (e.g. `010-03/`) containing `raw.hdr`, `whiteReference.hdr`,
`darkReference.hdr` and `gtMap.hdr` (plus their data files).

## Pipeline
1. Radiometric calibration with white and dark references
2. Removal of noisy edge bands (826 -> 645), 5-point moving-average smoothing
3. Band decimation to 128 bands, per-pixel min-max normalisation
4. 15x15 spatial-spectral patches centred on each labelled pixel
5. HybridSN-style CNN (3D convs over spectra, then a 2D conv, then a dense head; ~2.56M params)
6. Class-balanced sampling (20,000 samples/epoch), Adam, mixed-precision training, 10 epochs
7. Patient-level train/validation split so no patient appears in both sets;
   best epoch picked by validation Cohen's kappa

## Results (validation set, 14 images)
| Metric | Value |
|---|---|
| Overall accuracy | 90.4% |
| Cohen's kappa | 0.84 |
| Macro F1 | 0.88 |
| Tumor (TT) precision / recall / F1 | 0.87 / 0.74 / 0.80 |

Files in [`results/`](results/): `classification_report.txt`, `metrics.json`, `confusion_matrix.png`,
`training_curves.png`, `preprocessing_example.png` and `predictions/` (ground truth vs prediction
for every validation image) are written when you run the script.
The numbers above come from the original notebook run (`validation_results_from_notebook.txt`).

### Limitations
- Single patient-level split; the same validation set is used to choose the best epoch, so there is
  no fully independent test set.
- Metrics are over labelled pixels only (BG included), so they are not directly comparable to the paper's
  5-fold test results, which exclude BG from macro F1 (70.2% macro F1 there).
- Research prototype only. Not a clinical or diagnostic tool.

## Run
```bash
# 1. Install PyTorch with CUDA (see pytorch.org), then:
pip install -r requirements.txt

# 2. Train, evaluate and save all results
python brain_tumor_hsi_cnn.py --dataset-root "D:/ImageXTumor/Dataset" \
                              --work-dir "D:/ImageXTumor/Process Data" \
                              --results-dir results
```
Preprocessed `.npy` files and `best_model.pt` are written to `--work-dir` and are git-ignored.

## Repo contents
- `brain_tumor_hsi_cnn.py` - full pipeline (preprocessing, training, evaluation, figures)
- `PatientXTumor_original.ipynb` - original exploratory notebook
- `results/` - metrics and figures
