# ForensicsDetective: Assignment2

**Course:** EAS 510 - Basics of AI  
**Assignment:** Project 2  
**Author:** Owen Sutherlin  
**Forked From:** [delveccj/ForensicsDetective](https://github.com/delveccj/ForensicsDetective)

---

## Overview

This project extends the ForensicsDetective framework for PDF provenance detection. PDF files created by different tools (Microsoft Word, Google Docs, Python/ReportLab) contain unique binary signatures that, when rendered as grayscale images, are detectable by machine learning classifiers.

This assignment scales the original approach through controlled image augmentation, introduces two additional classifiers, and conducts a comprehensive robustness analysis to evaluate how performance degrades under real-world document distortions.

---

## Repository Structure
```
Project2/
├── src/                        # Source scripts
│   ├── convert.py
│   ├── train_baseline_classifiers.py
│   ├── generate_python_pdfs.py
│   ├── google_docs_converter.py
│   ├── google_docs_converter_batch.py
│   ├── google_docs_converter_oauth.py
│   ├── pdf_to_binary_image.py
│   └── create_comparison_images.py
├── results/                    # Outputs
│   ├── confusion_matrices/
│   ├── robustness_plots/
│   └── performance_metrics.csv
├── reports/
│   └── final_research_report.pdf
├── SETUP.md
├── AUGMENT.md
├── requirements.txt
└── README.md
```

> **Note on data:** Image data is stored at the repository root rather than in a `data/` folder due to the volume of files generated during augmentation. Original images are in `word_pdfs_png/`, `google_docs_pdfs_png/`, and `python_pdfs_png/`. Augmented variants follow the naming convention `{folder}_{augmentation_type}/` (e.g. `word_pdfs_png_gaussian/`).

---

## Dataset

| Class | Original Images |
|---|---|
| Microsoft Word | 398 |
| Google Docs | 396 |
| Python / ReportLab | 100 |
| **Total** | **894** |

After augmentation the dataset expands to **5,364 images** (6× original size).

---

## Augmentations

Five augmentations were applied independently to each original image:

| Augmentation | Description |
|---|---|
| Gaussian Noise | Additive noise, σ ∈ [5, 20] |
| JPEG Compression | Re-encoded at quality ∈ [20, 80] |
| DPI Downsampling | Scaled to simulate 150 or 72 DPI |
| Random Cropping | 1–3% removed from each border |
| Bit-Depth Reduction | 8-bit grayscale reduced to 4-bit |

See `AUGMENT.md` for full parameter documentation.

---

## Classifiers

Three classifiers were trained exclusively on original (non-augmented) data and evaluated on both original and augmented test sets:

| Classifier | Key Parameters |
|---|---|
| SVC | Linear kernel, C=1.0 |
| Random Forest | 100 estimators, min_samples_split=5 (grid search tuned) |
| KNN | k=3, uniform weighting (grid search tuned) |

---

## Results Summary

| Classifier | Original Accuracy | Best Augmentation | Worst Augmentation |
|---|---|---|---|
| SVC | 1.0000 | Gaussian / JPEG (1.0000) | Crop (0.6096) |
| Random Forest | 0.9832 | JPEG (0.9978) | Crop (0.5638) |
| KNN | 1.0000 | Gaussian / JPEG (0.9989) | Crop (0.9441) |

KNN demonstrated the strongest robustness profile across all augmentation conditions. Statistical significance of pairwise classifier differences was confirmed via McNemar's test (α = 0.05).

---

## Setup & Reproduction

### 1. Clone the repo
```bash
git clone https://github.com/owensutherlin/Assignment2_OwenSutherlin.git
cd Assignment2_OwenSutherlin
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run in Google Colab
Open `Assignment2_w.ipynb` in Google Colab and run all cells in order. The notebook handles dataset augmentation, classifier training, evaluation, and all visualizations end to end.

---

## Dependencies
```
numpy
pandas
matplotlib
scikit-learn
opencv-python
pillow
statsmodels
```

---

## References

- Original ForensicsDetective repository: [delveccj/ForensicsDetective](https://github.com/delveccj/ForensicsDetective)
- scikit-learn documentation: [scikit-learn.org](https://scikit-learn.org)
- statsmodels McNemar's test: [statsmodels.org](https://www.statsmodels.org)
