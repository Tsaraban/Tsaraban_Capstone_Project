# Tsaraban_Capstone_Project

# GCN vs RAG for Drug–Drug Interaction Prediction

Honours dissertation project comparing a Graph Convolutional Network (GCN) and a Retrieval-Augmented Generation (RAG) pipeline for predicting drug–drug interactions using the DrugBank database.

---

## Project Overview

This project investigates whether a RAG pipeline can serve as a genuine alternative to a GCN for DDI prediction, offering greater explainability and adaptability alongside competitive predictive performance.

Both systems are built from the same DrugBank data, evaluated on the same held-out test set, and compared using F1-score, Accuracy, and AUROC.

| Model | F1 | Accuracy | AUROC |
|---|---|---|---|
| GCN | 0.796 | 0.788 | 0.869 |
| RAG | 0.667 | 0.500 | 0.617 |

## Data

This project uses the DrugBank full database XML release. Due to licensing restrictions, the raw DrugBank XML file is not included in this repository.

To obtain the data:
1. Register for a free academic account at [drugbank.com](https://www.drugbank.com)
2. Download the full database XML release
3. Find converting code in the directory


## How to Run

Run scripts in order inside Google Colab. Each script saves its output to Google Drive so sessions can be resumed without rerunning expensive steps.

**Step 1 — Parse DrugBank XML**

Extracts drug IDs, names, and all interaction records into a CSV file. Uses lxml iterparse for memory-efficient processing of the ~1.77 GB XML file.

```bash
python 01_parse_drugbank.py
```

Output: `interactions.csv`

**Step 2 — Extract SMILES**

Second pass over the XML to extract molecular structure strings. Merges SMILES onto the interaction table and removes rows where either drug lacks a valid SMILES string.

```bash
python 02_extract_smiles.py
```

Output: `interactions_with_smiles.csv`

**Step 3 — Build dataset**

Deduplicates directed pairs, generates negative samples at a 1:1 ratio, and applies a stratified 80/10/10 train/validation/test split.

```bash
python 03_build_dataset.py
```

Output: `train.csv`, `val.csv`, `test.csv`

**Step 4 — Train GCN**

Converts SMILES strings to molecular graphs using RDKit and PyTorch Geometric. Trains the GCN with early stopping on validation loss. Saves best model weights.

```bash
python 04_gcn_model.py
```

Output: `gcn_best_model.pt`, `training_curves.png`

**Step 5 — Build and run RAG pipeline**

Encodes DrugBank drug documents using BioBERT and builds a FAISS flat L2 index. Saves embeddings and index to Drive. Runs inference on the test set.

```bash
python 05_rag_pipeline.py
```

Output: `drug_embeddings.npy`, `faiss_index.bin`, `rag_predictions.csv`

**Step 6 — Evaluate**

Evaluates both models on the test set. Produces comparison table, confusion matrices, ROC curves, Precision-Recall curves, and bar chart.

```bash
python 06_evaluate.py
```

Output: `results_table.csv`, `confusion_matrices.png`, `roc_curves.png`, `pr_curves.png`, `metric_comparison.png`

---

## Model Architecture

**GCN**
- 3 × GCNConv layers (hidden dims: 64, 128, 128) with ReLU activation
- Global mean pooling → 128-d graph embedding per drug
- Concatenated pair embedding (256-d) → 2 × FC layers with dropout (0.3) → sigmoid output
- Adam optimiser (lr=0.001, weight decay=1e-4), binary cross-entropy loss
- Batch size: 512, max 100 epochs, early stopping patience: 10

**RAG**
- Document corpus: one text document per drug aggregating all DrugBank interaction descriptions
- Encoder: BioBERT (`dmis-lab/biobert-base-cased-v1.2`) — [CLS] token as 768-d embedding
- Index: FAISS flat L2 — exact nearest-neighbour search
- Inference: average two drug embeddings → retrieve top-5 → distance-weighted cosine similarity → sigmoid → binary prediction at threshold 0.5

---

## Key Results

The GCN outperformed the RAG pipeline on every meaningful metric. RAG's poor threshold-based performance was primarily a calibration issue — its similarity scores clustered above 0.5 for nearly every test pair, causing it to predict interaction universally. Its AUROC of 0.617 confirms some real discriminative signal was present despite this.

RAG offered two advantages the GCN could not match:
- **Explainability** — predictions are traceable to specific retrieved pharmacological text
- **Adaptability** — new interactions can be incorporated by re-embedding one document, no retraining required

---

## Limitations

- Negative pairs are randomly sampled from undocumented drug pairs — some may represent real undocumented interactions
- RAG query embeddings are built by averaging two independently computed BioBERT embeddings rather than full pair re-encoding — a computational compromise that may have weakened retrieval signal
- Biologic drugs without valid SMILES strings are excluded — findings apply to small-molecule drugs only
- Binary classification only — interaction type and severity are not predicted

---

## Future Work

- Threshold calibration for RAG using validation-set tuning or Platt scaling
- Full pair-level BioBERT query encoding to isolate the effect of the averaging shortcut
- Hybrid system combining GCN structural predictions with RAG-retrieved evidence for clinician review
- Temporal train/test split to evaluate adaptability advantage under realistic conditions
- Multi-class prediction extending beyond binary interaction detection

---

## References
1. https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions/code
2. https://github.com/sshaghayeghs/DDI-LLM/blob/main/PyG__link_pred_DDI.ipynb
3. https://github.com/dmis-lab/biobert
4. https://github.com/facebookresearch/faiss/blob/main/faiss/python/faiss_example_external_module.swig
More referenced in the actual project.
