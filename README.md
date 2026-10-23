# Multimodal CLIP for Primary Progressive Aphasia

Pipeline from a summer 2025 internship at the UCSF NeuroAI Lab (RISE program). It tests whether
pairing **clinical notes** with **T1-weighted MRI** in a shared embedding space separates the three
variants of primary progressive aphasia: nonfluent, logopenic and semantic.

## Method

```
T1 MRI ──BrainIAC preprocessing──▶ BrainIAC foundation model ──▶ 2048-d image embedding ─┐
clinical note ──OpenAI text-embedding-3-large──▶ 3072-d ──linear projection──▶ 2048-d ───┴─▶ CLIP loss
                                                                       (symmetric InfoNCE, τ = 0.07)
```

- Both encoders are **frozen foundation models**. Only the text projection is trained, which keeps the number of
  learned parameters small enough for a small patient cohort.
- Training pulls each patient's note and scan together in the shared space and pushes mismatched
  pairs apart. The learned space is visualized with t-SNE, colored by variant.

## Files

| File | Purpose |
|---|---|
| `clip_pipeline.py` | End to end: CSV of MRI and note paths → preprocessing → embeddings → CLIP training → t-SNE |
| `batch_preprocess_setup.py` | Batch MRI preprocessing and dataset setup (BIDS or flat naming) |
| `batch_brainiac_feature_extraction.py` | BrainIAC features for a folder of scans |
| `single_patient_pipeline.py`, `create_visualizations.py` | One-patient walkthrough and figures |
| `demo/multimodal_ppa_predictor.ipynb` (+ `.py`) | Self-contained demo on synthetic data (51 patients, 3 variants): pairwise CLIP analysis of how well each modality pair separates PPA subtypes (AUC) |
| `create_synthetic_data.py`, `test_unified_pipeline.py` | Synthetic scans and notes for testing without patient data |

## Run on synthetic data

```bash
pip install -r requirements.txt     # also requires a BrainIAC checkout and its weights
export OPENAI_API_KEY=...
python create_synthetic_data.py
python test_unified_pipeline.py
```

**No patient data is included.** All patient MRIs and notes stayed inside UCSF.
