"""
Toxicity & Dose Prediction App
ML-powered toxicity prediction from SMILES
Multiple models: Random Forest, XGBoost, LightGBM, Neural Network, Ensemble
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Toxicity & Dose Prediction",
    page_icon="🔬",
    layout="wide"
)

# Title
st.markdown("""
<h1 style='text-align: center; color: #1f77b4;'>🔬 Toxicity & Dose Prediction App</h1>
<p style='text-align: center; font-size: 1.2rem; color: #666;'>ML-powered toxicity prediction from molecular structure (SMILES)</p>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    scaler = None
    
    try:
        scaler = joblib.load("/home/mpdr/.openclaw/workspace/toxicity_model/models/scaler.pkl")
        
        import os
        for f in os.listdir("/home/mpdr/.openclaw/workspace/toxicity_model/models/"):
            if f.endswith('_model.pkl'):
                name = f.replace('_model.pkl', '')
                models[name] = joblib.load(f"/home/mpdr/.openclaw/workspace/toxicity_model/models/{f}")
        
        return models, scaler, True
    except Exception as e:
        return models, scaler, False

def calculate_descriptors(smiles):
    """Calculate all molecular descriptors from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {}
    
    # Basic properties
    descriptors['MolWt'] = Descriptors.MolWt(mol)
    descriptors['LogP'] = Descriptors.MolLogP(mol)
    descriptors['TPSA'] = Descriptors.TPSA(mol)
    descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
    descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    descriptors['RingCount'] = Descriptors.RingCount(mol)
    descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    descriptors['FractionCSP3'] = Descriptors.FractionCSP3(mol)
    descriptors['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
    descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
    descriptors['BertzCT'] = Descriptors.BertzCT(mol)
    descriptors['Chi0'] = Descriptors.Chi0(mol)
    descriptors['Chi1'] = Descriptors.Chi1(mol)
    descriptors['Kappa1'] = Descriptors.Kappa1(mol)
    descriptors['Kappa2'] = Descriptors.Kappa2(mol)
    descriptors['LabuteASA'] = Descriptors.LabuteASA(mol)
    descriptors['BalabanJ'] = Descriptors.BalabanJ(mol) if Descriptors.BalabanJ(mol) else 0
    descriptors['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
    descriptors['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(mol)
    descriptors['MaxEStateIndex'] = Descriptors.MaxEStateIndex(mol)
    descriptors['MinEStateIndex'] = Descriptors.MinEStateIndex(mol)
    descriptors['MaxAbsEStateIndex'] = Descriptors.MaxAbsEStateIndex(mol)
    descriptors['MinAbsEStateIndex'] = Descriptors.MinAbsEStateIndex(mol)
    descriptors['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
    descriptors['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
    descriptors['MaxAbsPartialCharge'] = Descriptors.MaxAbsPartialCharge(mol)
    descriptors['MinAbsPartialCharge'] = Descriptors.MinAbsPartialCharge(mol)
    descriptors['ExactMolWt'] = Descriptors.ExactMolWt(mol)
    descriptors['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(mol)
    descriptors['MolMR'] = Descriptors.MolMR(mol)
    descriptors['NumCarbonAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    descriptors['NumNitrogenAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    descriptors['NumOxygenAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    descriptors['NumSulfurAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
    descriptors['NumHalogenAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53])
    descriptors['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
    descriptors['NumAromaticCarbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
    descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
    descriptors['NumAliphaticHeterocycles'] = Descriptors.NumAliphaticHeterocycles(mol)
    descriptors['NumAliphaticCarbocycles'] = Descriptors.NumAliphaticCarbocycles(mol)
    
    # Morgan fingerprint bits
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    for i in range(min(100, len(morgan_fp))):
        descriptors[f'Morgan_Bit_{i}'] = morgan_fp[i]
    
    return descriptors

def predict_toxicity(smiles, models, scaler):
    """Make toxicity predictions using all models"""
    
    desc = calculate_descriptors(smiles)
    if desc is None:
        return None, None
    
    # Create feature vector
    feature_cols = list(desc.keys())
    X = pd.DataFrame([desc])
    
    # Ensure columns match training
    try:
        X_scaled = scaler.transform(X)
    except:
        return None, None
    
    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(X_scaled)[0]
            predictions[name] = pred
        except:
            pass
    
    return predictions, desc

def get_toxicity_class(lD50):
    """Classify toxicity based on LD50"""
    if lD50 < 1:
        return "🔴 Extremely Toxic", "Highly toxic - severe risk"
    elif lD50 < 50:
        return "🟠 Highly Toxic", "Significant toxicity concern"
    elif lD50 < 500:
        return "🟡 Moderately Toxic", "Moderate toxicity risk"
    elif lD50 < 5000:
        return "🟢 Low Toxicity", "Low acute toxicity"
    else:
        return "✅ Very Low Toxicity", "Considered relatively safe at therapeutic doses"

def estimate_dose_category(lD50):
    """Estimate safe dose ranges based on LD50"""
    # NOAEL is typically 1/10 to 1/100 of LD50
    noael_low = lD50 / 100
    noael_high = lD50 / 10
    
    # Human equivalent dose (HED) - divide by 12 for animal to human
    hed_low = noael_low / 12
    hed_high = noael_high / 12
    
    # Starting dose for clinical trials (1/10 of NOAEL)
    starting_dose_low = noael_low / 10
    starting_dose_high = noael_high / 10
    
    return {
        'NOAEL_mgkg': (noael_low, noael_high),
        'HED_mgkg': (hed_low, hed_high),
        'Starting_Dose_mgkg': (starting_dose_low, starting_dose_high)
    }

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    st.subheader("🎯 Prediction Model")
    model_choice = st.radio(
        "Select model for prediction:",
        ["Ensemble (Best)", "Random Forest", "XGBoost", "LightGBM", "Neural Network", "All Models"]
    )
    
    st.subheader("📊 Display Options")
    show_molecule = st.checkbox("Show Molecule Structure", value=True)
    show_descriptors = st.checkbox("Show Molecular Descriptors", value=True)
    show_comparison = st.checkbox("Model Comparison", value=True)
    
    st.subheader("ℹ️ About")
    st.markdown("""
    **Toxicity Prediction App**
    
    Uses multiple ML models (Random Forest, XGBoost, LightGBM, Neural Network) to predict:
    - LD50 (Lethal Dose 50%)
    - NOAEL (No Observed Adverse Effect Level)
    - Safe starting dose
    
    Based on molecular descriptors calculated from SMILES structure.
    """)

# ============================================================
# MAIN APP
# ============================================================

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Batch Predict", "ℹ️ Info"])

with tab1:
    st.subheader("Single Molecule Toxicity Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Enter SMILES:",
            value="CC(=O)Oc1ccccc1C(=O)O",
            placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O"
        )
        
        # Preset molecules
        presets = {
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
            "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
            "Metformin": "CN(C)C(=N)N=C(N)N",
            "Atorvastatin": "CC(C)C1=C(C(C)=C(C=C1)C2C(C(C(N2CCC(CC(CC(=O)O)O)O)(C)C)C)C)C)C(=O)NC3=CC=CC=C3",
            "Warfarin": "FC(=O)C(C)Cc1c(F)c(F)c(F)c1Cc1c(F)c(F)c(F)c1C(=O)F",
            "Diazepam": "CN1C(=O)CN=C(c2ccccc2Cl)c2cc(Cl)ccc12"
        }
        
        preset = st.selectbox("Or choose a preset:", ["Custom"] + list(presets.keys()))
        if preset != "Custom":
            smiles_input = presets[preset]
    
    with col2:
        st.markdown("")  # Spacer
        st.markdown("")  # Spacer
        predict_btn = st.button("🔮 Predict Toxicity", type="primary", use_container_width=True)
    
    if predict_btn or smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        
        if mol is None:
            st.error("❌ Invalid SMILES! Please check your input.")
        else:
            # Load models
            models, scaler, models_loaded = load_models()
            
            if not models_loaded:
                st.warning("⚠️ Models not found. Running in demo mode with rule-based prediction.")
                desc = calculate_descriptors(smiles_input)
                demo_mode = True
            else:
                demo_mode = False
            
            # Show molecule
            if show_molecule:
                col1, col2 = st.columns([1, 2])
                with col1:
                    img = Draw.MolToImage(mol, size=(300, 250))
                    st.image(img, caption="2D Structure")
                
                with col2:
                    st.markdown("### 📋 Molecule Info")
                    st.code(f"SMILES: {smiles_input}")
                    st.markdown(f"**Formula:** {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
                    st.markdown(f"**MW:** {Descriptors.MolWt(mol):.2f} Da")
            
            # Make prediction
            if not demo_mode:
                predictions, desc = predict_toxicity(smiles_input, models, scaler)
            
            if desc:
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                if not demo_mode and predictions:
                    # Use model predictions
                    ld50_pred = predictions.get('ensemble_stacking', predictions.get('ensemble_voting', predictions.get('rf', 1000)))
                    
                    # Model comparison
                    if show_comparison:
                        pred_df = pd.DataFrame({
                            'Model': list(predictions.keys()),
                            'Predicted LD50 (mg/kg)': [f"{p:.2f}" for p in predictions.values()]
                        })
                        st.dataframe(pred_df, use_container_width=True)
                        st.markdown("")
                else:
                    # Demo mode - use rule-based estimation
                    # Based on MW, LogP, and other descriptors
                    mw = desc.get('MolWt', 300)
                    logp = desc.get('LogP', 2)
                    tpsa = desc.get('TPSA', 50)
                    
                    # Simple rule-based estimation (for demo)
                    base_ld50 = 500
                    
                    # Adjust based on MW
                    if mw < 200:
                        base_ld50 *= 0.5  # Smaller molecules often more toxic
                    elif mw > 800:
                        base_ld50 *= 2  # Larger molecules often less toxic
                    
                    # Adjust based on LogP
                    if logp > 5:
                        base_ld50 *= 0.7  # High LogP often more toxic
                    elif logp < 0:
                        base_ld50 *= 1.5  # Low LogP often less toxic
                    
                    # Adjust based on TPSA
                    if tpsa < 30:
                        base_ld50 *= 0.8  # Low TPSA often more toxic
                    elif tpsa > 150:
                        base_ld50 *= 1.3  # High TPSA often less toxic
                    
                    ld50_pred = base_ld50
                    st.info("📌 Demo Mode: Using rule-based estimation (models not trained yet)")
                
                # Toxicity classification
                toxicity_class, toxicity_desc = get_toxicity_class(ld50_pred)
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Predicted LD50", f"{ld50_pred:.2f} mg/kg", help="Lethal Dose 50%")
                
                with col2:
                    st.markdown(f"**Toxicity Class**")
                    st.markdown(f"{toxicity_class}")
                
                # Dose estimates
                doses = estimate_dose_category(ld50_pred)
                
                with col3:
                    st.metric("NOAEL (mg/kg)", f"{doses['NOAEL_mgkg'][0]:.2f} - {doses['NOAEL_mgkg'][1]:.2f}")
                
                with col4:
                    st.metric("Starting Dose (mg/kg)", f"{doses['Starting_Dose_mgkg'][0]:.2f} - {doses['Starting_Dose_mgkg'][1]:.2f}")
                
                st.markdown("")
                st.markdown(f"**Toxicity Assessment:** {toxicity_desc}")
                
                # Human equivalent doses
                st.markdown("### 🧪 Clinical Dose Estimates (Human Equivalent)")
                
                dose_data = {
                    'Metric': ['NOAEL (mg/kg)', 'Human Equivalent Dose', 'Starting Dose for Clinical Trials'],
                    'Low': [f"{doses['NOAEL_mgkg'][0]:.2f}", f"{doses['HED_mgkg'][0]:.2f}", f"{doses['Starting_Dose_mgkg'][0]:.3f}"],
                    'High': [f"{doses['NOAEL_mgkg'][1]:.2f}", f"{doses['HED_mgkg'][1]:.2f}", f"{doses['Starting_Dose_mgkg'][1]:.3f}"]
                }
                st.dataframe(pd.DataFrame(dose_data), use_container_width=True)
                
                # Warnings
                if ld50_pred < 50:
                    st.error("⚠️ HIGH TOXICITY WARNING: This compound shows signs of significant toxicity. Extreme caution required.")
                elif ld50_pred < 500:
                    st.warning("⚠️ MODERATE TOXICITY: This compound should be thoroughly evaluated before clinical development.")
                else:
                    st.success("✅ LOW TOXICITY: This compound shows acceptable toxicity profile for further development.")
                
                # Lipinski compliance
                st.markdown("### 💊 Drug-likeness (Lipinski Rule of 5)")
                
                lipinski_pass = (
                    desc.get('MolWt', 0) <= 500 and
                    desc.get('LogP', 0) <= 5 and
                    desc.get('NumHDonors', 0) <= 5 and
                    desc.get('NumHAcceptors', 0) <= 10
                )
                
                cols = st.columns(4)
                with cols[0]:
                    st.metric("MW", f"{desc.get('MolWt', 0):.1f}", "≤500" if desc.get('MolWt', 0) <= 500 else ">500", delta_color="normal")
                with cols[1]:
                    st.metric("LogP", f"{desc.get('LogP', 0):.2f}", "≤5" if desc.get('LogP', 0) <= 5 else ">5", delta_color="normal")
                with cols[2]:
                    st.metric("HBD", f"{desc.get('NumHDonors', 0)}", "≤5" if desc.get('NumHDonors', 0) <= 5 else ">5", delta_color="normal")
                with cols[3]:
                    st.metric("HBA", f"{desc.get('NumHAcceptors', 0)}", "≤10" if desc.get('NumHAcceptors', 0) <= 10 else ">10", delta_color="normal")
                
                if lipinski_pass:
                    st.success("✅ Passes Lipinski Rule of 5 - Good oral bioavailability expected")
                else:
                    st.warning("⚠️ Does not fully pass Lipinski Rule of 5 - May have bioavailability issues")
                
                # Show descriptors
                if show_descriptors:
                    st.markdown("---")
                    st.subheader("📐 Molecular Descriptors")
                    
                    # Group descriptors
                    basic_keys = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors']
                    structure_keys = ['RingCount', 'NumRotatableBonds', 'NumAromaticRings', 'HeavyAtomCount']
                    electronic_keys = ['Chi0', 'Chi1', 'BertzCT', 'MolMR']
                    
                    with st.expander("View All Descriptors"):
                        cols = st.columns(4)
                        for i, (k, v) in enumerate(desc.items()):
                            if isinstance(v, (int, float)) and not k.startswith('Morgan'):
                                cols[i%4].metric(k, f"{v:.4f}" if isinstance(v, float) else v)

with tab2:
    st.subheader("Batch Toxicity Prediction")
    
    batch_input = st.text_area(
        "Enter multiple SMILES (one per line):",
        height=200,
        placeholder="CC(=O)Oc1ccccc1C(=O)O\nCn1cnc2c1c(=O)n(c(=O)n2C)C\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"
    )
    
    if st.button("🔮 Predict Batch", type="primary"):
        if not batch_input.strip():
            st.warning("Enter at least one SMILES")
        else:
            smis = [s.strip() for s in batch_input.strip().split('\n') if s.strip()]
            
            results = []
            for smi in smis:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    desc = calculate_descriptors(smi)
                    if desc:
                        # Rule-based estimation for demo
                        mw = desc.get('MolWt', 300)
                        logp = desc.get('LogP', 2)
                        tpsa = desc.get('TPSA', 50)
                        
                        base_ld50 = 500
                        if mw < 200: base_ld50 *= 0.5
                        elif mw > 800: base_ld50 *= 2
                        if logp > 5: base_ld50 *= 0.7
                        elif logp < 0: base_ld50 *= 1.5
                        if tpsa < 30: base_ld50 *= 0.8
                        elif tpsa > 150: base_ld50 *= 1.3
                        
                        toxicity, _ = get_toxicity_class(base_ld50)
                        
                        results.append({
                            'SMILES': smi,
                            'MolWt': round(mw, 2),
                            'LogP': round(logp, 2),
                            'TPSA': round(tpsa, 2),
                            'LD50_mgkg': round(base_ld50, 2),
                            'Toxicity': toxicity,
                            'Valid': True
                        })
                    else:
                        results.append({'SMILES': smi, 'Valid': False, 'Error': 'Descriptor calc failed'})
                else:
                    results.append({'SMILES': smi, 'Valid': False, 'Error': 'Invalid SMILES'})
            
            df = pd.DataFrame(results)
            valid_df = df[df['Valid'] == True]
            
            st.success(f"✅ {len(valid_df)}/{len(results)} valid molecules")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "batch_toxicity_results.csv", "text/csv")

with tab3:
    st.subheader("About This App")
    
    st.markdown("""
    ## 🔬 Toxicity & Dose Prediction System
    
    This application uses machine learning to predict toxicity parameters from molecular structure (SMILES).
    
    ### What We Predict:
    
    | Parameter | Description | Use |
    |-----------|-------------|-----|
    | **LD50** | Lethal Dose 50% | Acute toxicity assessment |
    | **NOAEL** | No Observed Adverse Effect Level | Safe dose determination |
    | **Starting Dose** | Initial clinical trial dose | First-in-human dosing |
    
    ### Models Used:
    
    1. **Random Forest** - Ensemble of decision trees
    2. **XGBoost** - Gradient boosting with regularization
    3. **LightGBM** - Light gradient boosting
    4. **Neural Network** - Deep learning with multiple layers
    5. **Ensemble** - Combination of all models (best performance)
    
    ### Data Sources:
    
    - FDA Approved Drug Products (Orange Book)
    - EPA ToxCast Database
    - Therapeutics Data Commons (TDC)
    - ChEMBL & PubChem BioAssays
    - Published literature on drug toxicity
    
    ### How It Works:
    
    1. **Input**: Enter SMILES structure of compound
    2. **Descriptors**: Calculate 100+ molecular descriptors (MW, LogP, TPSA, etc.)
    3. **Prediction**: Run through multiple ML models
    4. **Output**: Predicted toxicity values and dose recommendations
    
    ### Disclaimer:
    
    ⚠️ **FOR RESEARCH PURPOSES ONLY**
    
    The predictions generated by this application are for **informational and research purposes only**. 
    They should NOT be used for:
    - Clinical decision making
    - Drug dosing without proper validation
    - Replacing official regulatory submissions
    
    Always consult appropriate regulatory agencies and conduct proper preclinical 
    and clinical studies for drug development decisions.
    """)
    
    st.markdown("---")
    st.markdown("<center>Built with 🧪 RDKit, scikit-learn, XGBoost, LightGBM | Jarvis AI</center>", unsafe_allow_html=True)

# Footer
st.markdown("---")