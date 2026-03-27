"""
TOXICITY PREDICTION MODEL V2 - 119 VALIDATED COMPOUNDS
With Complete Validation Matrix
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, Lipinski, Crippen
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="🔬 Toxicity Prediction v2", page_icon="🔬", layout="wide")

# Load training data
@st.cache_data
def load_training_data():
    df = pd.read_csv("training_data_119.csv")
    return df

training_df = load_training_data()

# Calculate molecular descriptors
def calc_descriptors(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    try:
        desc = {
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Crippen.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Lipinski.NumHDonors(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),
            'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
            'NumAromaticRings': Lipinski.NumAromaticRings(mol),
            'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'RingCount': Descriptors.RingCount(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'BertzCT': Descriptors.BertzCT(mol),
            'Chi0': Descriptors.Chi0(mol),
            'Chi1': Descriptors.Chi1(mol),
            'Kappa1': Descriptors.Kappa1(mol),
            'Kappa2': Descriptors.Kappa2(mol),
            'HallKierAlpha': Descriptors.HallKierAlpha(mol),
            'LabuteASA': Descriptors.LabuteASA(mol),
        }
        morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
        for i in range(256):
            desc[f'FP_{i}'] = int(morgan_fp[i])
        return desc
    except:
        return None

# Train model
@st.cache_resource
def train_toxicity_model():
    features_list = []
    labels = []
    drugs = []
    ld50_values = []
    
    for _, row in training_df.iterrows():
        feat = calc_descriptors(row['SMILES'])
        if feat is not None:
            features_list.append(feat)
            # Toxic = 1 if LD50 < 500 mg/kg
            toxicity_class = 1 if row['LD50'] < 500 else 0
            labels.append(toxicity_class)
            drugs.append(row['Drug'])
            ld50_values.append(row['LD50'])
    
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    ld50_array = np.array(ld50_values)
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test, drugs_train, drugs_test, ld50_train, ld50_test = train_test_split(
        X, y, drugs, ld50_array, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # Train Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=6, random_state=42)
    gb.fit(X_train_s, y_train)
    gb_pred = gb.predict(X_test_s)
    gb_acc = accuracy_score(y_test, gb_pred)
    
    # Try XGBoost
    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, eval_metric='logloss')
        xgb_clf.fit(X_train_s, y_train)
        xgb_pred = xgb_clf.predict(X_test_s)
        xgb_acc = accuracy_score(y_test, xgb_pred)
    except:
        xgb_clf = None
        xgb_acc = 0
        xgb_pred = np.zeros_like(y_test)
    
    # Try LightGBM
    try:
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(n_estimators=300, max_depth=6, random_state=42, verbose=-1)
        lgb_clf.fit(X_train_s, y_train)
        lgb_pred = lgb_clf.predict(X_test_s)
        lgb_acc = accuracy_score(y_test, lgb_pred)
    except:
        lgb_clf = None
        lgb_acc = 0
        lgb_pred = np.zeros_like(y_test)
    
    # Ensemble prediction
    total_acc = rf_acc + gb_acc + xgb_acc + lgb_acc
    if total_acc > 0:
        ensemble_pred = ((rf_pred * rf_acc + gb_pred * gb_acc + 
                       xgb_pred * xgb_acc + lgb_pred * lgb_acc) / total_acc > 0.5).astype(int)
    else:
        ensemble_pred = rf_pred
    
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_s, y_train, cv=cv)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, ensemble_pred)
    
    # Get probabilities for ROC
    rf_prob = rf.predict_proba(X_test_s)[:, 1]
    gb_prob = gb.predict_proba(X_test_s)[:, 1]
    avg_prob = (rf_prob + gb_prob) / 2
    if xgb_clf:
        xgb_prob = xgb_clf.predict_proba(X_test_s)[:, 1]
        avg_prob = (avg_prob + xgb_prob) / 2
    if lgb_clf:
        lgb_prob = lgb_clf.predict_proba(X_test_s)[:, 1]
        avg_prob = (avg_prob + lgb_prob) / 2
    
    return {
        'rf': rf, 'gb': gb, 'xgb': xgb_clf, 'lgb': lgb_clf,
        'scaler': scaler,
        'accuracy': ensemble_acc,
        'rf_acc': rf_acc,
        'gb_acc': gb_acc,
        'xgb_acc': xgb_acc,
        'lgb_acc': lgb_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_cols': list(X.columns),
        'n_samples': len(X),
        'class_dist': {'toxic': int(sum(y)), 'non_toxic': int(len(y)-sum(y))},
        'cm': cm.tolist(),
        'y_test': y_test.tolist(),
        'ensemble_pred': ensemble_pred.tolist(),
        'avg_prob': avg_prob.tolist(),
        'ld50_test': ld50_test.tolist(),
        'drugs_test': drugs_test
    }

# Header
st.markdown("""
<h1 style='text-align: center; color: #1f77b4;'>🔬 Toxicity Prediction Model v2</h1>
<p style='text-align: center; font-size: 1.2rem; color: #666;'>
<b>ML Ensemble • 119 Validated Compounds • Complete Validation Matrix</b>
</p>
""", unsafe_allow_html=True)

# Train model
with st.spinner("Training ensemble model..."):
    model = train_toxicity_model()

st.success(f"✅ Model trained | Accuracy: {model['accuracy']*100:.1f}% | CV: {model['cv_mean']*100:.1f}%")

# Navigation
page = st.sidebar.radio("Navigation", ["🎯 Predict", "📊 Validation Matrix", "📈 Detailed Metrics"])

# ============================================================
# PREDICTION PAGE
# ============================================================
if page == "🎯 Predict":
    st.header("🎯 Toxicity Prediction")
    
    smile_input = st.text_area("Enter SMILES:", placeholder="CC(=O)Oc1ccccc1C(=O)O", height=80)
    
    presets = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Cisplatin": "N[Pt]Cl(N)Cl",
        "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "Nicotine": "CN1CCCC1c2cccnc2",
        "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    }
    
    preset = st.selectbox("Presets:", ["Custom"] + list(presets.keys()))
    if preset != "Custom":
        smile_input = presets[preset]
    
    if smile_input:
        mol = Chem.MolFromSmiles(smile_input.strip())
        
        if mol is None:
            st.error("❌ Invalid SMILES!")
        else:
            desc = calc_descriptors(smile_input.strip())
            if desc:
                X = pd.DataFrame([desc])
                for col in model['feature_cols']:
                    if col not in X.columns:
                        X[col] = 0
                X = X[model['feature_cols']]
                X_s = model['scaler'].transform(X)
                
                probs = []
                probs.append(model['rf'].predict_proba(X_s)[0][1])
                probs.append(model['gb'].predict_proba(X_s)[0][1])
                if model['xgb']:
                    probs.append(model['xgb'].predict_proba(X_s)[0][1])
                if model['lgb']:
                    probs.append(model['lgb'].predict_proba(X_s)[0][1])
                
                avg_prob = np.mean(probs)
                is_toxic = avg_prob > 0.5
                
                if is_toxic:
                    ld50_est = max(1, min(500 * (1 - avg_prob) * 2, 500))
                else:
                    ld50_est = max(500, min(500 + 1500 * avg_prob, 10000))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    toxic_label = "🔴 TOXIC" if is_toxic else "🟢 NON-TOXIC"
                    st.markdown(f"**Classification:**\n### {toxic_label}")
                
                with col2:
                    st.metric("Toxic Probability", f"{avg_prob*100:.1f}%")
                
                with col3:
                    st.metric("Est. LD50", f"{ld50_est:.0f} mg/kg")
                
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 30], 'color': '#27ae60'},
                            {'range': [30, 70], 'color': '#f39c12'},
                            {'range': [70, 100], 'color': '#e74c3c'}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'value': 50}
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# VALIDATION MATRIX PAGE
# ============================================================
elif page == "📊 Validation Matrix":
    st.header("📊 Validation Matrix")
    
    cm = np.array(model['cm'])
    y_test = np.array(model['y_test'])
    ensemble_pred = np.array(model['ensemble_pred'])
    
    # Confusion Matrix Display
    st.markdown("### 🔢 Confusion Matrix")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TP (True Positive)", cm[1][1] if cm.shape == (2, 2) else "N/A")
    with col2:
        st.metric("TN (True Negative)", cm[0][0] if cm.shape == (2, 2) else "N/A")
    with col3:
        st.metric("FP (False Positive)", cm[0][1] if cm.shape == (2, 2) else "N/A")
    with col4:
        st.metric("FN (False Negative)", cm[1][0] if cm.shape == (2, 2) else "N/A")
    
    # Metrics
    st.markdown("### 📊 Performance Metrics")
    
    accuracy = model['accuracy']
    precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.metric("**Accuracy**", f"{accuracy*100:.1f}%")
    with mcol2:
        st.metric("**Precision**", f"{precision*100:.1f}%")
    with mcol3:
        st.metric("**Recall**", f"{recall*100:.1f}%")
    with mcol4:
        st.metric("**F1 Score**", f"{f1*100:.1f}%")
    with mcol5:
        st.metric("**Specificity**", f"{specificity*100:.1f}%")
    
    # Confusion Matrix Table
    st.markdown("### 📋 Confusion Matrix Table")
    
    if cm.shape == (2, 2):
        cm_df = pd.DataFrame(cm, 
                           columns=['Predicted Non-Toxic', 'Predicted Toxic'],
                           index=['Actual Non-Toxic', 'Actual Toxic'])
    else:
        cm_df = pd.DataFrame({'Value': cm.flatten()})
    st.table(cm_df)
    
    # Formula explanations
    st.markdown("""
    ### 📐 Formula Definitions
    
    | Metric | Formula | Description |
    |--------|---------|-------------|
    | **Accuracy** | (TP + TN) / Total | Overall correct predictions |
    | **Precision** | TP / (TP + FP) | Of predicted toxic, how many actually toxic |
    | **Recall** | TP / (TP + FN) | Of actual toxic, how many predicted toxic |
    | **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean |
    | **Specificity** | TN / (TN + FP) | Of actual non-toxic, how many predicted non-toxic |
    """)

# ============================================================
# DETAILED METRICS PAGE
# ============================================================
elif page == "📈 Detailed Metrics":
    st.header("📈 Detailed Model Metrics")
    
    st.markdown(f"""
    ### 🎯 Model Performance Summary
    
    | Metric | Value |
    |-------|-------|
    | **Training Samples** | {model['n_samples']} |
    | **Toxic (LD50 < 500)** | {model['class_dist']['toxic']} |
    | **Non-toxic (LD50 ≥ 500)** | {model['class_dist']['non_toxic']} |
    | **Ensemble Accuracy** | {model['accuracy']*100:.1f}% |
    | **5-Fold Cross-Validation** | {model['cv_mean']*100:.1f}% (±{model['cv_std']*200:.1f}%) |
    """)
    
    st.markdown("""
    ### 📊 Individual Model Performance
    
    | Model | Accuracy |
    |-------|----------|
    | Random Forest | {:.1f}% |
    | Gradient Boosting | {:.1f}% |
    | XGBoost | {:.1f}% |
    | LightGBM | {:.1f}% |
    """.format(
        model['rf_acc']*100,
        model['gb_acc']*100,
        model['xgb_acc']*100,
        model['lgb_acc']*100
    ))
    
    st.markdown("""
    ### 🔬 Training Data Info
    
    All 119 compounds have been **properly validated** from:
    - FDA drug labels
    - DrugBank database
    - Peer-reviewed literature
    
    **Validation Standard:**
    - Toxic: LD50 < 500 mg/kg (oral rat)
    - Non-toxic: LD50 ≥ 500 mg/kg (oral rat)
    """)
    
    # Show some test predictions
    st.markdown("### 🧪 Sample Test Predictions")
    
    test_results = pd.DataFrame({
        'Drug': model['drugs_test'][:20],
        'Actual LD50': model['ld50_test'][:20],
        'Predicted': ['Toxic' if p == 1 else 'Non-Toxic' for p in model['ensemble_pred'][:20]],
        'Actual': ['Toxic' if a == 1 else 'Non-Toxic' for a in model['y_test'][:20]],
        'Correct': ['✅' if p == a else '❌' for p, a in zip(model['ensemble_pred'][:20], model['y_test'][:20])]
    })
    st.dataframe(test_results)

st.sidebar.markdown(f"**Accuracy:** {model['accuracy']*100:.1f}%")
st.sidebar.markdown(f"**Training Data:** {model['n_samples']} compounds")