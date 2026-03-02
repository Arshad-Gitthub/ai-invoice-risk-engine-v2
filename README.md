# ABC Company — 90% AI Invoice Processing Pipeline

A 9-layer AI system that automatically processes invoices with 90% automation, requiring only 10% human review.

## 9 AI Layers
| Layer | Model | Purpose |
|-------|-------|---------|
| 1 | TF-IDF + GBM | NLP Email Classifier |
| 2 | Learned Similarity | Duplicate Detector |
| 3 | Isolation Forest | Unsupervised Anomaly |
| 4 | DBSCAN | Spatial Anomaly |
| 5 | MLP Neural Network | Supervised Anomaly |
| 6 | Mahalanobis Distance | Vendor Behavior Profiler |
| 7 | GBM Regressor | PO Tolerance Learner |
| 8 | Random Forest | Meta-Learner (Final Decision) |
| 9 | PSI Drift Monitor | Continuous Learning |

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the pipeline
```bash
python abc_90ai_pipeline.py
```

## Project Structure
```
ai-invoice-risk-engine/
├── abc_90ai_pipeline.py        ← Main pipeline (run this)
├── requirements.txt
├── README.md
├── data/
│   ├── __init__.py
│   └── abc_dataset.py          ← Training data generator + live invoices
└── src/
    └── ai_layers/
        ├── __init__.py
        ├── layer1_nlp_parser.py
        ├── layer2_ai_dedup.py
        ├── layer3_ensemble_anomaly.py   ← Contains layers 3, 4, 5
        ├── layer6_vendor_profiler.py
        ├── layer7_po_matcher.py
        ├── layer8_meta_learner.py
        └── layer9_continuous_learning.py
```

## Decisions
- **AUTO_POST** — AI confident, posts automatically (0% human)
- **REVIEW** — Borderline, finance manager reviews (~10% human)
- **HOLD** — High risk, escalated to finance manager
- **REJECTED_DUPLICATE** — Duplicate detected, rejected automatically
