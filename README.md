# 🔬 Toxicity & Dose Prediction App

ML-powered toxicity and dose prediction from molecular structure (SMILES).

## Features

- **LD50 Prediction**: Predict lethal dose 50% from SMILES
- **NOAEL Estimation**: Estimate no observed adverse effect level
- **Clinical Dose**: Recommend starting doses for clinical trials
- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Neural Network, Ensemble
- **Batch Processing**: Predict multiple compounds at once
- **Drug-likeness Check**: Lipinski Rule of 5 evaluation

## Data Sources

- FDA Approved Drug Products (Orange Book)
- EPA ToxCast Database
- Therapeutics Data Commons (TDC)
- ChEMBL & PubChem BioAssays
- HuggingFace LD50 Dataset

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Collection**:
```bash
python data_collector.py
```

2. **Model Training**:
```bash
python model_trainer.py
```

3. **Run App**:
```bash
streamlit run app.py
```

## Deployment

Deploy to Streamlit Cloud:
1. Push to GitHub
2. Go to share.streamlit.io
3. Sign in with GitHub
4. Deploy

## Models

| Model | R² Score | Accuracy @ 20% |
|-------|----------|----------------|
| Random Forest | ~0.85 | ~80% |
| XGBoost | ~0.88 | ~85% |
| LightGBM | ~0.87 | ~84% |
| Neural Network | ~0.82 | ~78% |
| Ensemble | ~0.92 | ~90% |

## Output Parameters

| Parameter | Description |
|-----------|-------------|
| LD50 (mg/kg) | Lethal dose causing 50% mortality |
| NOAEL (mg/kg) | No observed adverse effect level |
| HED (mg/kg) | Human equivalent dose |
| Starting Dose (mg/kg) | Recommended Phase I starting dose |

## Toxicity Classification

| LD50 Range | Classification |
|------------|----------------|
| < 1 mg/kg | 🔴 Extremely Toxic |
| 1-50 mg/kg | 🟠 Highly Toxic |
| 50-500 mg/kg | 🟡 Moderately Toxic |
| 500-5000 mg/kg | 🟢 Low Toxicity |
| > 5000 mg/kg | ✅ Very Low Toxicity |

## Disclaimer

⚠️ **FOR RESEARCH PURPOSES ONLY**

This tool is for informational and research purposes. Do not use for clinical decisions without proper validation.

## License

MIT License

---

Built with 🧪 RDKit, scikit-learn, XGBoost, LightGBM