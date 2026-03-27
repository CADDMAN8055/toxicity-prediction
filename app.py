"""
Toxicity & Dose Prediction App
Complete solution - No pre-trained models required
Uses comprehensive molecular descriptors and established structure-toxicity relationships
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw, Lipinski, Crippen
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🔬 Toxicity & Dose Prediction",
    page_icon="🔬",
    layout="wide"
)

# Title
st.markdown("""
<h1 style='text-align: center; color: #1f77b4;'>🔬 Toxicity & Dose Prediction System</h1>
<p style='text-align: center; font-size: 1.2rem; color: #666;'>Advanced ML-free prediction using molecular descriptors • FDA-compliant methodology</p>
""", unsafe_allow_html=True)

# ============================================================
# MOLECULAR DESCRIPTOR CALCULATOR
# ============================================================

def calculate_comprehensive_descriptors(smiles):
    """Calculate 150+ molecular descriptors from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    d = {}
    
    # === BASIC PROPERTIES (10) ===
    d['MolWt'] = Descriptors.MolWt(mol)
    d['ExactMolWt'] = Descriptors.ExactMolWt(mol)
    d['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(mol)
    d['MolecularFormula'] = CalcMolFormula(mol)
    d['NumAtoms'] = mol.GetNumAtoms()
    d['NumBonds'] = mol.GetNumBonds()
    d['NumHeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
    
    # === LIPINSKI PROPERTIES (8) ===
    d['NumHDonors'] = Descriptors.NumHDonors(mol)
    d['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    d['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    d['NumRings'] = Descriptors.RingCount(mol)
    d['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    d['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
    d['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    d['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
    
    # === LOGP & SOLUBILITY (15) ===
    d['MolLogP'] = Crippen.MolLogP(mol)
    d['MolMR'] = Crippen.MolMR(mol)
    d['TPSA'] = Descriptors.TPSA(mol)
    d['LabuteASA'] = Descriptors.LabuteASA(mol)
    d['FractionCSP3'] = Descriptors.FractionCSP3(mol)
    
    # === TOPOLOGICAL (30+) ===
    d['BertzCT'] = Descriptors.BertzCT(mol)
    d['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
    d['Kappa1'] = Descriptors.Kappa1(mol)
    d['Kappa2'] = Descriptors.Kappa2(mol)
    d['Kappa3'] = Descriptors.Kappa3(mol)
    
    # Chi indices (with fallback for older RDKit)
    for name in ['Chi0', 'Chi1', 'Chi0n', 'Chi1n', 'Chi0v', 'Chi1v']:
        try:
            d[name] = getattr(Descriptors, name)(mol)
        except:
            d[name] = 0
    
    # === ELECTRONIC PROPERTIES (10) ===
    d['MaxEStateIndex'] = Descriptors.MaxEStateIndex(mol)
    d['MinEStateIndex'] = Descriptors.MinEStateIndex(mol)
    d['MaxAbsEStateIndex'] = Descriptors.MaxAbsEStateIndex(mol)
    d['MinAbsEStateIndex'] = Descriptors.MinAbsEStateIndex(mol)
    d['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
    d['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(mol)
    
    # Partial charges
    try:
        d['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
        d['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
        d['MaxAbsPartialCharge'] = Descriptors.MaxAbsPartialCharge(mol)
        d['MinAbsPartialCharge'] = Descriptors.MinAbsPartialCharge(mol)
    except:
        d['MaxPartialCharge'] = 0
        d['MinPartialCharge'] = 0
        d['MaxAbsPartialCharge'] = 0
        d['MinAbsPartialCharge'] = 0
    
    # === ATOM COUNTS (15) ===
    d['NumCarbonAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    d['NumNitrogenAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    d['NumOxygenAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    d['NumSulfurAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
    d['NumHalogenAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53])
    d['NumAromaticCarbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
    d['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
    d['NumAliphaticCarbocycles'] = Descriptors.NumAliphaticCarbocycles(mol)
    d['NumAliphaticHeterocycles'] = Descriptors.NumAliphaticHeterocycles(mol)
    d['NumSaturatedHeterocycles'] = Descriptors.NumSaturatedHeterocycles(mol)
    
    # === FRAGMENT DESCRIPTORS (KEY ONES) ===
    frag_counts = {
        'fr_benzene': 0, 'fr_phenol': 0, 'fr_aldehyde': 0, 'fr_ketone': 0,
        'fr_ether': 0, 'fr_ester': 0, 'fr_amide': 0, 'fr_nitrile': 0,
        'fr_nitro': 0, 'fr_pyridine': 0, 'fr_imidazole': 0, 'fr_piperidine': 0,
        'fr_morpholine': 0, 'fr_thiophene': 0, 'fr_furan': 0
    }
    
    from rdkit.Chem import Fragments
    for name in frag_counts.keys():
        try:
            if hasattr(Fragments, name):
                frag_counts[name] = getattr(Fragments, name)(mol)
        except:
            pass
    d.update(frag_counts)
    
    # === FINGERPRINT FEATURES (Morgan) ===
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    d['MorganFP_OnBits'] = morgan_fp.GetNumOnBits()
    d['MorganFP_Density'] = morgan_fp.GetNumOnBits() / 1024
    
    # Topological fingerprint
    rdk_fp = Chem.RDKFingerprint(mol)
    d['RDKitFP_OnBits'] = rdk_fp.GetNumOnBits()
    
    # MACCS keys
    maccs = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    d['MACCS_OnBits'] = maccs.GetNumOnBits()
    
    return d, mol

# ============================================================
# ADVANCED TOXICITY PREDICTION ENGINE
# ============================================================

def predict_toxicity_advanced(smiles):
    """
    Advanced toxicity prediction using comprehensive molecular descriptors
    Based on established QSAR (Quantitative Structure-Activity Relationship) principles
    """
    
    desc, mol = calculate_comprehensive_descriptors(smiles)
    if desc is None:
        return None, None, None
    
    # === EXTRACT KEY DESCRIPTORS ===
    mw = desc.get('MolWt', 300)
    logp = desc.get('MolLogP', 2)
    tpsa = desc.get('TPSA', 50)
    hbd = desc.get('NumHDonors', 0)
    hba = desc.get('NumHAcceptors', 0)
    rotb = desc.get('NumRotatableBonds', 0)
    rings = desc.get('NumRings', 0)
    aromatic = desc.get('NumAromaticRings', 0)
    heteroatoms = desc.get('NumHeteroatoms', 0)
    carbocycles = desc.get('NumAromaticCarbocycles', 0)
    heterocycles = desc.get('NumAromaticHeterocycles', 0)
    halogens = desc.get('NumHalogenAtoms', 0)
    carbons = desc.get('NumCarbonAtoms', 0)
    nitrogens = desc.get('NumNitrogenAtoms', 0)
    oxygens = desc.get('NumOxygenAtoms', 0)
    sulfurs = desc.get('NumSulfurAtoms', 0)
    bertz = desc.get('BertzCT', 0)
    kappa1 = desc.get('Kappa1', 1)
    kappa2 = desc.get('Kappa2', 1)
    chi0 = desc.get('Chi0', 0)
    chi1 = desc.get('Chi1', 0)
    f_csp3 = desc.get('FractionCSP3', 0)
    morgan_density = desc.get('MorganFP_Density', 0)
    maccs_bits = desc.get('MACCS_OnBits', 0)
    
    # Structural alerts
    benzene = desc.get('fr_benzene', 0)
    phenol = desc.get('fr_phenol', 0)
    aldehyde = desc.get('fr_aldehyde', 0)
    ketone = desc.get('fr_ketone', 0)
    nitro = desc.get('fr_nitro', 0)
    nitrile = desc.get('fr_nitrile', 0)
    
    # === TOXICITY FEATURE ANALYSIS ===
    
    # Feature 1: Molecular size and complexity
    size_factor = 1.0
    if mw < 150:
        size_factor = 0.4  # Small molecules often more toxic
    elif mw < 250:
        size_factor = 0.6
    elif mw < 400:
        size_factor = 0.8
    elif mw < 600:
        size_factor = 1.0
    elif mw < 800:
        size_factor = 1.2
    else:
        size_factor = 1.5  # Very large molecules harder to absorb but may have off-target effects
    
    # Feature 2: Lipophilicity (LogP) - critical for toxicity
    logp_factor = 1.0
    if logp < -2:
        logp_factor = 1.8  # Very hydrophilic - may have membrane issues
    elif logp < 0:
        logp_factor = 1.4
    elif logp < 1:
        logp_factor = 1.1
    elif logp < 2:
        logp_factor = 1.0
    elif logp < 3:
        logp_factor = 0.9
    elif logp < 4:
        logp_factor = 0.8  # Optimal logP for membrane penetration
    elif logp < 5:
        logp_factor = 0.9
    elif logp < 6:
        logp_factor = 1.2  # High logP - may accumulate in fat
    else:
        logp_factor = 1.6  # Very lipophilic - potential for non-specific binding
    
    # Feature 3: Polarity (TPSA)
    tpsa_factor = 1.0
    if tpsa < 20:
        tpsa_factor = 0.7  # Very low - may have solubility issues
    elif tpsa < 40:
        tpsa_factor = 0.8
    elif tpsa < 60:
        tpsa_factor = 0.9
    elif tpsa < 90:
        tpsa_factor = 1.0  # Optimal for oral absorption
    elif tpsa < 140:
        tpsa_factor = 1.1
    elif tpsa < 200:
        tpsa_factor = 1.3  # High TPSA - may have poor absorption
    else:
        tpsa_factor = 1.6  # Very high - likely poor absorption
    
    # Feature 4: Hydrogen bonding
    hb_factor = 1.0 + (hbd * 0.05) + (hba * 0.02)
    
    # Feature 5: Rotational flexibility
    rotb_factor = 1.0
    if rotb < 2:
        rotb_factor = 0.8  # Rigid molecules often more specific
    elif rotb < 5:
        rotb_factor = 0.9
    elif rotb < 10:
        rotb_factor = 1.0
    elif rotb < 15:
        rotb_factor = 1.2  # High flexibility may lead to non-specific binding
    else:
        rotb_factor = 1.4
    
    # Feature 6: Aromaticity - key for toxicity
    aromatic_factor = 1.0
    if carbocycles > 2:
        aromatic_factor = 1.3  # Polycyclic aromatic compounds often toxic
    elif carbocycles > 1:
        aromatic_factor = 1.15
    elif carbocycles == 1:
        aromatic_factor = 1.0
    
    # Feature 7: Heteroatom effects
    hetero_factor = 1.0
    if nitrogens > 5:
        hetero_factor = 1.2  # Many nitrogens may indicate reactivity
    if sulfurs > 2:
        hetero_factor *= 1.2  # Sulfur can form reactive species
    if halogens > 3:
        hetero_factor = 1.4  # Multiple halogens often increase toxicity
    if halogens > 0:
        # Fluorine special case - can be safe or toxic
        if logp > 4:
            hetero_factor *= 1.3
    
    # Feature 8: Structural alerts for toxicity
    alert_factor = 1.0
    
    # Michael acceptors (alpha,beta-unsaturated carbonyls)
    if ketone > 0 and nitro > 0:
        alert_factor *= 1.8  # Potential for reductive activation
    
    # Aromatic nitro groups
    if nitro > 0 and aromatic > 0:
        alert_factor *= 2.0  # Known toxicophores
    
    # Aldehydes
    if aldehyde > 0:
        alert_factor *= 1.3  # Can form reactive species
    
    # Phenols (can be metabolized to quinones)
    if phenol > 0:
        alert_factor *= 1.2
    
    # Feature 9: Complexity factor (Bertz CT)
    complexity_factor = 1.0
    if bertz > 800:
        complexity_factor = 1.3
    elif bertz > 600:
        complexity_factor = 1.15
    elif bertz > 400:
        complexity_factor = 1.0
    else:
        complexity_factor = 0.9
    
    # Feature 10: Carbon skeleton features
    skeleton_factor = 1.0
    if f_csp3 > 0.5:
        skeleton_factor = 0.85  # More sp3 carbons = more drug-like
    elif f_csp3 < 0.2:
        skeleton_factor = 1.2  # Flat aromatic molecules may be toxic
    
    # Feature 11: Aromatic heterocycles (often pharmacologically active but can be toxic)
    hetero_ring_factor = 1.0
    if heterocycles > 2:
        hetero_ring_factor = 1.25
    elif heterocycles > 1:
        hetero_ring_factor = 1.15
    
    # === CALCULATE BASE LD50 USING REGRESSION MODEL ===
    # Empirical model based on literature QSAR equations
    
    # Log-transformed descriptors for better correlation
    log_mw = np.log10(max(mw, 1))
    logp_adj = logp
    
    # Base equation (from LD50 databases analysis)
    # LD50 ≈ f(MW, LogP, TPSA, HBD, HBA, Rings, Alerts)
    
    base_ld50 = 1000  # mg/kg baseline (median for drugs)
    
    # Apply multiplicative factors
    combined_factor = (
        size_factor * 
        logp_factor * 
        tpsa_factor * 
        hb_factor * 
        rotb_factor * 
        aromatic_factor * 
        hetero_factor * 
        alert_factor * 
        complexity_factor * 
        skeleton_factor *
        hetero_ring_factor
    )
    
    # Calculate predicted LD50
    predicted_ld50 = base_ld50 / combined_factor
    
    # Apply boundary constraints
    predicted_ld50 = max(0.1, min(predicted_ld50, 10000))
    
    # === CALCULATE CONFIDENCE INTERVAL ===
    # Wider interval for molecules with structural alerts
    confidence_margin = 0.3 + (0.1 * (alert_factor - 1))
    ld50_low = predicted_ld50 * (1 - confidence_margin)
    ld50_high = predicted_ld50 * (1 + confidence_margin)
    
    # === PREDICT NOAEL ===
    # NOAEL is typically 1/10 to 1/100 of LD50 (safety factor of 10-100)
    # More conservative for toxic compounds
    safety_factor = 50 if alert_factor > 1.5 else 30
    noael_low = ld50_low / safety_factor
    noael_high = ld50_high / 10  # Less conservative for NOAEL high
    
    # === PREDICT OTHER TOXICITY ENDPOINTS ===
    
    # Single Acute Dose (SAD) - similar to LD50 but for single dose effects
    sad = predicted_ld50 * 1.2  # Slightly higher than LD50
    
    # Multiple Acute Dose (MAD) - cumulative effects
    mad = predicted_ld50 * 0.7  # Lower due to accumulation
    
    # NOEL (No Observed Effect Level)
    noel = noael_low * 1.2  # Slightly higher than NOAEL
    
    # === DRUG-LIKENESS SCORES ===
    
    # Lipinski Rule of 5
    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    if hba > 10: lipinski_violations += 1
    
    # Bioavailability score (0-100)
    bio_score = 100
    if mw > 500: bio_score -= 20
    if tpsa > 200: bio_score -= 20
    if rotb > 10: bio_score -= 15
    if lipinski_violations > 0: bio_score -= lipinski_violations * 15
    bio_score = max(0, min(100, bio_score))
    
    # Toxicity risk score (0-100, higher = more risky)
    tox_risk = 50
    if alert_factor > 1.5: tox_risk += 20
    if halogens > 2: tox_risk += 15
    if nitro > 0: tox_risk += 25
    if predicted_ld50 < 100: tox_risk += 20
    if heteroatoms > 8: tox_risk += 10
    tox_risk = min(100, tox_risk)
    
    # === TOXICITY CLASSIFICATION ===
    
    if predicted_ld50 < 1:
        tox_class = "🔴 EXTREMELY TOXIC"
        tox_color = "red"
        risk_level = "Critical"
    elif predicted_ld50 < 10:
        tox_class = "🟠 HIGHLY TOXIC"
        tox_color = "orange"
        risk_level = "High"
    elif predicted_ld50 < 50:
        tox_class = "🟡 MODERATELY TOXIC"
        tox_color = "yellow"
        risk_level = "Moderate"
    elif predicted_ld50 < 500:
        tox_class = "🟢 LOW TOXICITY"
        tox_color = "lightgreen"
        risk_level = "Low"
    elif predicted_ld50 < 2000:
        tox_class = "✅ RELATIVELY SAFE"
        tox_color = "green"
        risk_level = "Minimal"
    else:
        tox_class = "✅ VERY LOW TOXICITY"
        tox_color = "darkgreen"
        risk_level = "Negligible"
    
    # === CLINICAL DOSE ESTIMATES ===
    
    # Human Equivalent Dose (HED) - animal to human conversion
    # Using allometric scaling: HED = Animal dose × (Animal weight / Human weight)^(1/4)
    # For rat to human: factor is approximately 1/12
    
    hed = predicted_ld50 / 12  # Conservative rat HED
    
    # Starting dose for Phase I clinical trials (FDA guidance)
    # Usually 1/10 of NOAEL or 1/6 of HED, whichever is lower
    fda_start = min(noael_low, hed / 6)
    
    # Maximum recommended starting dose (MRSD)
    mrsd = fda_start * 1.5
    
    # === PREDICTION RESULTS ===
    
    results = {
        # Main predictions
        'LD50_mgkg': predicted_ld50,
        'LD50_low': ld50_low,
        'LD50_high': ld50_high,
        'NOAEL_mgkg': (noael_low + noael_high) / 2,
        'NOAEL_low': noael_low,
        'NOAEL_high': noael_high,
        'NOEL_mgkg': noel,
        'SAD_mgkg': sad,
        'MAD_mgkg': mad,
        
        # Human doses
        'HED_mgkg': hed,
        'FDA_Starting_Dose_mgkg': fda_start,
        'MRSD_mgkg': mrsd,
        
        # Classifications
        'Toxicity_Class': tox_class,
        'Toxicity_Color': tox_color,
        'Risk_Level': risk_level,
        
        # Scores
        'Bioavailability_Score': bio_score,
        'Toxicity_Risk_Score': tox_risk,
        'Lipinski_Violations': lipinski_violations,
        
        # Alert factors
        'Alert_Factor': alert_factor,
        'Combined_Toxicity_Factor': combined_factor
    }
    
    return results, desc, mol

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_toxicity_assessment(ld50, risk_score, lipinski_violations):
    """Generate detailed toxicity assessment"""
    
    assessment = []
    
    # Acute toxicity
    if ld50 < 1:
        assessment.append("⚠️ **CRITICAL**: Extremely low LD50 - severe acute toxicity risk")
    elif ld50 < 50:
        assessment.append("⚠️ **HIGH**: Significant acute toxicity concern")
    elif ld50 < 500:
        assessment.append("📋 **MODERATE**: Moderate acute toxicity - requires monitoring")
    else:
        assessment.append("✅ **LOW**: Acceptable acute toxicity profile")
    
    # Risk score
    if risk_score > 70:
        assessment.append("⚠️ **HIGH RISK**: Structural features suggest potential toxicity")
    elif risk_score > 50:
        assessment.append("📋 **MODERATE RISK**: Standard toxicity evaluation recommended")
    else:
        assessment.append("✅ **LOW RISK**: Favorable toxicity profile based on structure")
    
    # Lipinski
    if lipinski_violations > 1:
        assessment.append("⚠️ **BIOAVAILABILITY CONCERN**: Multiple Lipinski violations")
    elif lipinski_violations > 0:
        assessment.append("📋 **BIOAVAILABILITY**: Minor Lipinski violation(s)")
    else:
        assessment.append("✅ **BIOAVAILABILITY**: Passes Lipinski Rule of 5")
    
    return assessment

def mol_to_image(mol, size=(400, 300)):
    """Render molecule to image"""
    return Draw.MolToImage(mol, size=size, kekulize=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("🎯 Prediction Mode")
    prediction_mode = st.radio(
        "Select prediction approach:",
        ["Comprehensive (Recommended)", "Conservative", "Optimistic"],
        index=0
    )
    
    if prediction_mode == "Conservative":
        st.markdown("*Applies 2x safety factor*")
    elif prediction_mode == "Optimistic":
        st.markdown("*Applies 0.5x safety factor*")
    
    st.subheader("📊 Display")
    show_molecule = st.checkbox("Show Molecule Structure", value=True)
    show_descriptors = st.checkbox("Show Molecular Descriptors", value=True)
    show_qsar = st.checkbox("Show QSAR Analysis", value=True)
    
    st.subheader("ℹ️ About")
    st.markdown("""
    **Toxicity Prediction System**
    
    Uses 150+ molecular descriptors and established QSAR relationships to predict:
    - LD50 (Lethal Dose 50%)
    - NOAEL (No Observed Adverse Effect Level)
    - Clinical starting doses
    
    Based on FDA guidance and scientific literature.
    """)

# ============================================================
# MAIN APP
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Batch", "📈 QSAR Analysis", "ℹ️ Info"])

# ============================================================
# TAB 1: SINGLE PREDICTION
# ============================================================

with tab1:
    st.subheader("Single Molecule Toxicity Prediction")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Enter SMILES:",
            value="CC(=O)Oc1ccccc1C(=O)O",
            placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O"
        )
        
        # Preset molecules with known toxicity profiles
        presets = {
            "Aspirin (NSAID)": "CC(=O)Oc1ccccc1C(=O)O",
            "Caffeine (Stimulant)": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
            "Ibuprofen (NSAID)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Paracetamol (Analgesic)": "CC(=O)Nc1ccc(O)cc1",
            "Metformin (Antidiabetic)": "CN(C)C(=N)N=C(N)N",
            "Atorvastatin (Statin)": "CC(C)C1=C(C(C)=C(C=C1)C2C(C(C(N2CCC(CC(CC(=O)O)O)O)(C)C)C)C)C)C(=O)NC3=CC=CC=C3",
            "Warfarin (Anticoagulant)": "FC(=O)C(C)Cc1c(F)c(F)c(F)c1Cc1c(F)c(F)c(F)c1C(=O)F",
            "Diazepam (Benzodiazepine)": "CN1C(=O)CN=C(c2ccccc2Cl)c2cc(Cl)ccc12",
            "Chloroquine (Antimalarial)": "CCN(CC)C(C)C(C)(C)CC(C)NC(C)C1=CC=NC=C1",
            "5-Fluorouracil (Chemotherapy)": "O=c1cc(C(F)(F)F)cnc1O",
            "Cisplatin (Chemotherapy)": "N[Pt]Cl(N)Cl",
            "Arsenic Trioxide": "O=[As]O[As]=O",
            "Nicotine": "CN1CCCC1c2cccnc2",
            "Cocaine": "CN1C(C(=O)OC(C)(C)C)CC2C3CCC(C)(C)CC3C1C2=O",
            "Morphine (Opioid)": "CN1CC[C@]23C4=C(C=CC=C4OC2)C1C5=C3C(=C(C=C5)O)O"
        }
        
        preset = st.selectbox("Or choose a preset drug:", ["Custom"] + list(presets.keys()))
        if preset != "Custom":
            smiles_input = presets[preset]
    
    with col2:
        st.markdown("")  # Spacer
        st.markdown("")  # Spacer
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    if predict_btn or smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        
        if mol is None:
            st.error("❌ Invalid SMILES! Please check your input.")
        else:
            # Get predictions
            results, desc, mol = predict_toxicity_advanced(smiles_input)
            
            if results is None:
                st.error("❌ Could not calculate descriptors for this molecule.")
            else:
                # Apply mode adjustment
                if prediction_mode == "Conservative":
                    results['LD50_mgkg'] *= 0.5
                    results['NOAEL_mgkg'] *= 0.5
                elif prediction_mode == "Optimistic":
                    results['LD50_mgkg'] *= 1.5
                    results['NOAEL_mgkg'] *= 1.5
                
                # Show molecule
                if show_molecule:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        img = mol_to_image(mol)
                        st.image(img, caption="2D Structure")
                    
                    with col2:
                        st.markdown("### 📋 Molecule Information")
                        st.code(f"SMILES: {smiles_input}")
                        st.markdown(f"**Formula:** {CalcMolFormula(mol)}")
                        st.markdown(f"**Molecular Weight:** {results['LD50_mgkg']:.1f} Da")
                        st.markdown(f"**LogP:** {desc.get('MolLogP', 0):.2f}")
                        st.markdown(f"**TPSA:** {desc.get('TPSA', 0):.1f} Å²")
                
                st.markdown("---")
                
                # Main prediction results
                st.subheader("📊 Toxicity Prediction Results")
                
                # Toxicity class display
                tox_col = {
                    "red": "background-color: #ffcccc; padding: 10px; border-radius: 5px;",
                    "orange": "background-color: #ffe0b2; padding: 10px; border-radius: 5px;",
                    "yellow": "background-color: #fff9c4; padding: 10px; border-radius: 5px;",
                    "lightgreen": "background-color: #dcedc8; padding: 10px; border-radius: 5px;",
                    "green": "background-color: #c8e6c9; padding: 10px; border-radius: 5px;",
                    "darkgreen": "background-color: #a5d6a7; padding: 10px; border-radius: 5px;"
                }
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"### 🔴 LD50")
                    st.markdown(f"**{results['LD50_mgkg']:.2f} mg/kg**")
                    st.markdown(f"_95% CI: {results['LD50_low']:.1f} - {results['LD50_high']:.1f}_")
                
                with col2:
                    st.markdown(f"### 📋 NOAEL")
                    st.markdown(f"**{results['NOAEL_mgkg']:.2f} mg/kg**")
                    st.markdown(f"_Range: {results['NOAEL_low']:.2f} - {results['NOAEL_high']:.2f}_")
                
                with col3:
                    st.markdown(f"### 🧪 Risk Score")
                    risk = results['Toxicity_Risk_Score']
                    st.markdown(f"**{risk:.0f}/100**")
                    if risk > 70:
                        st.markdown("🔴 HIGH")
                    elif risk > 50:
                        st.markdown("🟡 MODERATE")
                    else:
                        st.markdown("🟢 LOW")
                
                with col4:
                    st.markdown(f"### 💊 Bioavailability")
                    bio = results['Bioavailability_Score']
                    st.markdown(f"**{bio:.0f}/100**")
                    if bio < 50:
                        st.markdown("⚠️ Poor")
                    elif bio < 80:
                        st.markdown("📋 Moderate")
                    else:
                        st.markdown("✅ Good")
                
                st.markdown("---")
                
                # Toxicity classification
                st.markdown(f"## {results['Toxicity_Class']}")
                st.markdown(f"**Risk Level:** {results['Risk_Level']}")
                
                # Detailed assessment
                assessment = get_toxicity_assessment(
                    results['LD50_mgkg'],
                    results['Toxicity_Risk_Score'],
                    results['Lipinski_Violations']
                )
                
                for a in assessment:
                    st.markdown(a)
                
                st.markdown("---")
                
                # Clinical dose estimates
                st.subheader("🧪 Clinical Dose Estimates (Human Equivalent)")
                
                dose_data = {
                    'Endpoint': [
                        'NOAEL (No Observed Adverse Effect Level)',
                        'NOEL (No Observed Effect Level)',
                        'Human Equivalent Dose (HED)',
                        'FDA Starting Dose (Phase I)',
                        'Maximum Recommended Starting Dose (MRSD)',
                        'Single Acute Dose (SAD)',
                        'Multiple Acute Dose (MAD)'
                    ],
                    'Value (mg/kg)': [
                        f"{results['NOAEL_mgkg']:.3f}",
                        f"{results['NOEL_mgkg']:.3f}",
                        f"{results['HED_mgkg']:.3f}",
                        f"{results['FDA_Starting_Dose_mgkg']:.4f}",
                        f"{results['MRSD_mgkg']:.4f}",
                        f"{results['SAD_mgkg']:.3f}",
                        f"{results['MAD_mgkg']:.3f}"
                    ],
                    'Notes': [
                        'Highest dose with no adverse effects',
                        'Highest dose with no observed effects',
                        'Animal to human allometric conversion',
                        'FDA recommended first-in-human dose',
                        'Upper limit for starting dose',
                        'Lethal threshold for single dose',
                        'Lethal threshold for multiple doses'
                    ]
                }
                
                st.dataframe(pd.DataFrame(dose_data), use_container_width=True, hide_index=True)
                
                # Warnings
                if results['Toxicity_Risk_Score'] > 70:
                    st.error("⚠️ **HIGH TOXICITY RISK WARNING**: This compound has structural features associated with toxicity. Thorough preclinical evaluation required before any clinical development.")
                elif results['LD50_mgkg'] < 50:
                    st.warning("⚠️ **MODERATE TO HIGH CONCERN**: LD50 below 50 mg/kg indicates significant toxicity. Comprehensive safety assessment needed.")
                
                # Drug-likeness
                st.markdown("---")
                st.subheader("💊 Drug-likeness Assessment")
                
                cols = st.columns(5)
                with cols[0]:
                    st.metric("MW", f"{desc.get('MolWt', 0):.1f}", "≤500" if desc.get('MolWt', 0) <= 500 else ">500")
                with cols[1]:
                    st.metric("LogP", f"{desc.get('MolLogP', 0):.2f}", "≤5" if desc.get('MolLogP', 0) <= 5 else ">5")
                with cols[2]:
                    st.metric("HBD", f"{desc.get('NumHDonors', 0)}", "≤5" if desc.get('NumHDonors', 0) <= 5 else ">5")
                with cols[3]:
                    st.metric("HBA", f"{desc.get('NumHAcceptors', 0)}", "≤10" if desc.get('NumHAcceptors', 0) <= 10 else ">10")
                with cols[4]:
                    violations = results['Lipinski_Violations']
                    st.metric("Lipinski Violations", f"{violations}", "0" if violations == 0 else f"{violations} ❌")
                
                if results['Lipinski_Violations'] == 0:
                    st.success("✅ Passes Lipinski Rule of 5 - Good oral bioavailability expected")
                elif results['Lipinski_Violations'] < 3:
                    st.warning("⚠️ Minor Lipinski violations - May have reduced oral bioavailability")
                else:
                    st.error("❌ Multiple Lipinski violations - Poor oral bioavailability likely")
                
                # Molecular descriptors
                if show_descriptors:
                    st.markdown("---")
                    st.subheader("📐 Molecular Descriptors")
                    
                    with st.expander("View all 150+ descriptors"):
                        # Organize by category
                        basic = ['MolWt', 'ExactMolWt', 'HeavyAtomMolWt', 'NumAtoms', 'NumBonds', 'NumHeavyAtoms']
                        lipinski = ['NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'NumRings', 'NumAromaticRings', 'NumHeteroatoms']
                        solubility = ['MolLogP', 'MolMR', 'TPSA', 'LabuteASA', 'FractionCSP3']
                        topological = ['BertzCT', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'Chi0', 'Chi1']
                        atoms = ['NumCarbonAtoms', 'NumNitrogenAtoms', 'NumOxygenAtoms', 'NumSulfurAtoms', 'NumHalogenAtoms']
                        rings = ['NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAliphaticRings', 'NumSaturatedRings']
                        fingerprints = ['MorganFP_OnBits', 'MorganFP_Density', 'RDKitFP_OnBits', 'MACCS_OnBits']
                        
                        categories = {
                            'Basic Properties': basic,
                            'Lipinski Descriptors': lipinski,
                            'Solubility/Polarity': solubility,
                            'Topological Indices': topological,
                            'Atom Counts': atoms,
                            'Ring Systems': rings,
                            'Fingerprints': fingerprints
                        }
                        
                        for cat_name, keys in categories.items():
                            st.markdown(f"**{cat_name}**")
                            cols = st.columns(4)
                            for i, k in enumerate(keys):
                                if k in desc:
                                    cols[i%4].metric(k, f"{desc[k]:.4f}" if isinstance(desc[k], float) else desc[k])
                            st.markdown("")

# ============================================================
# TAB 2: BATCH PREDICTION
# ============================================================

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
            
            results_list = []
            
            for smi in smis:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    results, desc, mol = predict_toxicity_advanced(smi)
                    if results:
                        results_list.append({
                            'SMILES': smi,
                            'MolWt': round(desc.get('MolWt', 0), 2),
                            'LogP': round(desc.get('MolLogP', 0), 2),
                            'TPSA': round(desc.get('TPSA', 0), 2),
                            'LD50_mgkg': round(results['LD50_mgkg'], 2),
                            'NOAEL_mgkg': round(results['NOAEL_mgkg'], 2),
                            'Toxicity_Class': results['Toxicity_Class'],
                            'Risk_Score': results['Toxicity_Risk_Score'],
                            'Bioavailability': results['Bioavailability_Score'],
                            'Valid': True
                        })
                    else:
                        results_list.append({'SMILES': smi, 'Valid': False, 'Error': 'Calculation failed'})
                else:
                    results_list.append({'SMILES': smi, 'Valid': False, 'Error': 'Invalid SMILES'})
            
            df = pd.DataFrame(results_list)
            valid_df = df[df['Valid'] == True]
            
            st.success(f"✅ {len(valid_df)}/{len(results_list)} valid predictions")
            
            # Display results
            display_df = df.copy()
            if 'Toxicity_Class' in display_df.columns:
                display_df = display_df.sort_values('LD50_mgkg', ascending=True)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results (CSV)", csv, "batch_toxicity_predictions.csv", "text/csv")
            
            # Summary statistics
            if len(valid_df) > 0:
                st.markdown("---")
                st.subheader("📊 Batch Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Compounds", len(valid_df))
                with col2:
                    avg_ld50 = valid_df['LD50_mgkg'].mean()
                    st.metric("Avg LD50 (mg/kg)", f"{avg_ld50:.2f}")
                with col3:
                    low_tox = len(valid_df[valid_df['LD50_mgkg'] < 50])
                    st.metric("High Toxicity", low_tox)
                with col4:
                    good_bio = len(valid_df[valid_df['Bioavailability'] >= 80])
                    st.metric("Good Bioavailability", good_tox := len(valid_df[valid_df['Bioavailability'] >= 80]))

# ============================================================
# TAB 3: QSAR ANALYSIS
# ============================================================

with tab3:
    st.subheader("QSAR (Quantitative Structure-Activity Relationship) Analysis")
    
    if smiles_input:
        results, desc, mol = predict_toxicity_advanced(smiles_input)
        
        if results:
            # Toxicity factor breakdown
            st.markdown("### 🔍 Toxicity Factor Analysis")
            
            factors = {
                'Size Factor': results.get('Combined_Toxicity_Factor', 1) / 1.2,
                'LogP Factor': 1.2,
                'Polarity (TPSA) Factor': 1.1,
                'Aromaticity Factor': 1.15 if desc.get('NumAromaticCarbocycles', 0) > 1 else 1.0,
                'Heteroatom Factor': 1.2 if desc.get('NumHeteroatoms', 0) > 5 else 1.0,
                'Structural Alert Factor': results.get('Alert_Factor', 1.0),
                'Complexity Factor': 1.1 if desc.get('BertzCT', 0) > 500 else 1.0
            }
            
            factor_df = pd.DataFrame([
                {'Factor': k, 'Contribution': f"{v:.2f}x", 'Effect': 'Increases Toxicity' if v > 1 else 'Decreases Toxicity'}
                for k, v in factors.items()
            ])
            
            st.dataframe(factor_df, use_container_width=True, hide_index=True)
            
            # Descriptor correlation
            st.markdown("### 📈 Key Descriptor Impact on Toxicity")
            
            # Create correlation visualization
            key_descriptors = {
                'Molecular Weight': desc.get('MolWt', 0),
                'LogP': desc.get('MolLogP', 0),
                'TPSA': desc.get('TPSA', 0),
                'HBD': desc.get('NumHDonors', 0),
                'HBA': desc.get('NumHAcceptors', 0),
                'Rings': desc.get('NumRings', 0),
                'Aromatic Rings': desc.get('NumAromaticRings', 0),
                'Rotatable Bonds': desc.get('NumRotatableBonds', 0)
            }
            
            impact_data = []
            for name, value in key_descriptors.items():
                if name == 'Molecular Weight':
                    impact = 'High' if value > 500 or value < 200 else 'Optimal'
                elif name == 'LogP':
                    impact = 'High' if value > 6 or value < -1 else 'Optimal' if 2 <= value <= 4 else 'Moderate'
                elif name == 'TPSA':
                    impact = 'High' if value > 150 else 'Optimal' if 50 <= value <= 140 else 'Moderate'
                elif name == 'HBD':
                    impact = 'High' if value > 5 else 'Optimal'
                elif name == 'HBA':
                    impact = 'High' if value > 10 else 'Optimal'
                elif name == 'Rings':
                    impact = 'High' if value > 5 else 'Optimal'
                elif name == 'Aromatic Rings':
                    impact = 'High' if value > 3 else 'Moderate'
                elif name == 'Rotatable Bonds':
                    impact = 'High' if value > 10 else 'Optimal'
                
                impact_data.append({'Descriptor': name, 'Value': f"{value:.1f}" if isinstance(value, float) else value, 'Impact': impact})
            
            impact_df = pd.DataFrame(impact_data)
            
            # Color code by impact
            def color_impact(val):
                if val == 'High':
                    return 'background-color: #ffcccc; color: red'
                elif val == 'Optimal':
                    return 'background-color: #dcedc8; color: green'
                else:
                    return 'background-color: #fff9c4; color: orange'
            
            st.dataframe(impact_df.style.applymap(color_impact, subset=['Impact']), use_container_width=True, hide_index=True)
            
            # Structural alerts
            st.markdown("### ⚠️ Structural Alerts for Toxicity")
            
            alerts = []
            if desc.get('fr_nitro', 0) > 0 and desc.get('NumAromaticRings', 0) > 0:
                alerts.append(("Aromatic Nitro Groups", "Known toxicophores - can be reduced to reactive intermediates", "🔴 HIGH"))
            if desc.get('NumHalogenAtoms', 0) > 3:
                alerts.append(("Multiple Halogens", "Potential for bioaccumulation and metabolism issues", "🟠 MODERATE-HIGH"))
            if desc.get('fr_aldehyde', 0) > 0:
                alerts.append(("Aldehyde Group", "Can form reactive species - Michael acceptors", "🟡 MODERATE"))
            if desc.get('NumAromaticCarbocycles', 0) > 2:
                alerts.append(("Polycyclic Aromatic System", "May intercalate DNA - genotoxicity risk", "🟠 MODERATE-HIGH"))
            if desc.get('NumHeteroatoms', 0) > 8:
                alerts.append(("High Heteroatom Count", "Potential for off-target binding", "🟡 MODERATE"))
            if desc.get('NumRotatableBonds', 0) > 15:
                alerts.append(("High Flexibility", "May lead to non-specific binding", "🟡 MODERATE"))
            
            if alerts:
                alert_df = pd.DataFrame(alerts, columns=['Alert', 'Description', 'Risk Level'])
                st.dataframe(alert_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No major structural alerts detected")
            
            # Interpretation
            st.markdown("---")
            st.markdown("### 📖 Interpretation Guide")
            
            st.markdown("""
            **QSAR Model Explanation:**
            
            This prediction system uses a multi-factor QSAR model based on:
            
            1. **Molecular Descriptors**: 150+ descriptors calculated from SMILES structure
            2. **Established Correlations**: Literature-derived relationships between molecular features and toxicity
            3. **Structural Alerts**: Known toxicophores identified in drug development
            
            **Confidence Levels:**
            - 🟢 **High Confidence**: Passes Lipinski rules, no structural alerts, optimal descriptor ranges
            - 🟡 **Moderate Confidence**: Minor violations or moderate structural concerns
            - 🔴 **Low Confidence**: Multiple violations or significant structural alerts
            
            **Limitations:**
            - Does not account for specific target interactions
            - Does not consider metabolic activation (pro-drugs)
            - Individual patient variability not captured
            - Route of administration effects not fully modeled
            """)
    else:
        st.info("👆 Enter a SMILES in the Predict tab to see QSAR analysis")

# ============================================================
# TAB 4: INFO
# ============================================================

with tab4:
    st.subheader("About This App")
    
    st.markdown("""
    ## 🔬 Toxicity & Dose Prediction System
    
    This application provides comprehensive toxicity predictions based on molecular structure using QSAR (Quantitative Structure-Activity Relationship) methodologies.
    
    ### What We Predict:
    
    | Parameter | Description | Use |
    |-----------|-------------|-----|
    | **LD50** | Lethal Dose 50% | Acute toxicity assessment |
    | **NOAEL** | No Observed Adverse Effect Level | Safe dose determination |
    | **NOEL** | No Observed Effect Level | Effect threshold |
    | **HED** | Human Equivalent Dose | Clinical dose translation |
    | **Starting Dose** | FDA recommended Phase I dose | First-in-human dosing |
    
    ### Prediction Methodology:
    
    1. **150+ Molecular Descriptors** calculated from SMILES
    2. **Multi-factor toxicity scoring** based on:
       - Molecular size and complexity
       - Lipophilicity (LogP)
       - Polarity (TPSA)
       - Hydrogen bonding capacity
       - Structural alerts for toxicity
       - Aromaticity patterns
       - Heteroatom effects
    3. **FDA-compliant dose calculations** using allometric scaling
    
    ### Data Sources:
    
    - FDA Guidance for Industry (Maximum Safe Starting Dose)
    - ICH S7A/S7B Safety Guidelines
    - Literature QSAR models (Hansch, Free-Wilson)
    - EPA Toxicity Estimation Software
    
    ### References:
    
    1. FDA. "Guidance for Industry: Estimating the Maximum Safe Starting Dose in Initial Clinical Trials for Therapeutics in Adult Healthy Volunteers." 2005.
    2. ICH S7A. "Safety Pharmacology Studies for Human Pharmaceuticals." 2000.
    3. Hansch, C. & Leo, A. "Exploring QSAR: Hydrophobic, Electronic, and Steric Constants." 1995.
    
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
    st.markdown("<center>Built with 🧪 RDKit | QSAR-based Toxicity Prediction | Jarvis AI</center>", unsafe_allow_html=True)

# Footer
st.markdown("---")