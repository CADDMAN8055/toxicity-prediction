"""
Toxicity Prediction ML Model
Multiple ML/AI approaches for toxicity prediction:
- Random Forest
- XGBoost
- LightGBM
- Neural Network (MLP)
- Graph Neural Network (if data available)
- Ensemble (combines all models)
Target: >95% accuracy
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.neural_network import MLPRegressor
import joblib

# Try importing advanced libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False
    print("LightGBM not available")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False
    print("RDKit not available")

class ToxicityPredictor:
    """Multi-model toxicity prediction system"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.is_fitted = False
        
    def calculate_descriptors(self, smiles):
        """Calculate molecular descriptors from SMILES"""
        if not RDKIT_AVAILABLE:
            return None
        
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
        
        # More advanced descriptors
        descriptors['MaxEStateIndex'] = Descriptors.MaxEStateIndex(mol)
        descriptors['MinEStateIndex'] = Descriptors.MinEStateIndex(mol)
        descriptors['MaxAbsEStateIndex'] = Descriptors.MaxAbsEStateIndex(mol)
        descriptors['MinAbsEStateIndex'] = Descriptors.MinAbsEStateIndex(mol)
        
        # Partial charges
        descriptors['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
        descriptors['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
        descriptors['MaxAbsPartialCharge'] = Descriptors.MaxAbsPartialCharge(mol)
        descriptors['MinAbsPartialCharge'] = Descriptors.MinAbsPartialCharge(mol)
        
        # Electronic
        descriptors['ExactMolWt'] = Descriptors.ExactMolWt(mol)
        descriptors['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(mol)
        descriptors['MolMR'] = Descriptors.MolMR(mol)
        
        # Atom counts
        descriptors['NumCarbonAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        descriptors['NumNitrogenAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        descriptors['NumOxygenAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        descriptors['NumSulfurAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
        descriptors['NumHalogenAtoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53])
        
        # Ring descriptors
        descriptors['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
        descriptors['NumAromaticCarbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
        descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
        descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
        descriptors['NumAliphaticHeterocycles'] = Descriptors.NumAliphaticHeterocycles(mol)
        descriptors['NumAliphaticCarbocycles'] = Descriptors.NumAliphaticCarbocycles(mol)
        
        # Fingerprint bits (Morgan)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        for i in range(min(100, len(morgan_fp))):  # Use first 100 bits as features
            descriptors[f'Morgan_Bit_{i}'] = morgan_fp[i]
        
        return descriptors
    
    def prepare_data(self, df):
        """Prepare data for training"""
        
        # If SMILES column exists, calculate descriptors
        if 'SMILES' in df.columns and RDKIT_AVAILABLE:
            print("Calculating molecular descriptors from SMILES...")
            all_descriptors = []
            for smi in df['SMILES']:
                desc = self.calculate_descriptors(smi)
                if desc:
                    all_descriptors.append(desc)
                else:
                    all_descriptors.append({})
            
            desc_df = pd.DataFrame(all_descriptors)
            df = pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
        
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ['Drug_Name', 'SMILES', 'Standardized_SMILES', 'LD50_mgkg', 
                       'NOAEL_mgkg', 'NOEL_mgkg', 'Phase', 'Study_Type', 'Species',
                       'SMILES_Hash', 'InChI', 'Name', 'CID']
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = [col for col in self.feature_columns if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        X = df[self.feature_columns].copy()
        y = df['LD50_mgkg'].copy() if 'LD50_mgkg' in df.columns else None
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Handle infinity
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X, y
    
    def build_models(self):
        """Build all ML models"""
        print("\nBuilding models...")
        
        # Random Forest
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        print("✓ Random Forest")
        
        # Gradient Boosting
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        print("✓ Gradient Boosting")
        
        # XGBoost
        if XGB_AVAILABLE:
            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            print("✓ XGBoost")
        
        # LightGBM
        if LGB_AVAILABLE:
            self.models['lgb'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            print("✓ LightGBM")
        
        # Neural Network
        self.models['nn'] = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        print("✓ Neural Network (MLP)")
        
        # Ensemble - Voting
        estimators = [(name, model) for name, model in self.models.items() if name != 'nn']
        if len(estimators) > 1:
            self.models['ensemble_voting'] = VotingRegressor(estimators)
            print("✓ Ensemble (Voting)")
        
        # Ensemble - Stacking
        if len(estimators) > 2:
            self.models['ensemble_stacking'] = StackingRegressor(
                estimators=estimators[:3],
                final_estimator=GradientBoostingRegressor(n_estimators=100, max_depth=5),
                cv=5
            )
            print("✓ Ensemble (Stacking)")
    
    def train(self, X, y):
        """Train all models"""
        
        # Scale features
        self.scalers['X'] = RobustScaler()
        X_scaled = self.scalers['X'].fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining on {len(X_train)} samples, validating on {len(X_test)} samples")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Accuracy at different thresholds
            threshold_10 = np.mean(np.abs(y_test - y_pred_test) / y_test * 100 <= 10)
            threshold_20 = np.mean(np.abs(y_test - y_pred_test) / y_test * 100 <= 20)
            threshold_50 = np.mean(np.abs(y_test - y_pred_test) / y_test * 100 <= 50)
            
            results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'accuracy_10': threshold_10 * 100,
                'accuracy_20': threshold_20 * 100,
                'accuracy_50': threshold_50 * 100
            }
            
            print(f"  Train RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}")
            print(f"  Accuracy @ 10%: {threshold_10*100:.1f}%, @ 20%: {threshold_20*100:.1f}%, @ 50%: {threshold_50*100:.1f}%")
        
        self.is_fitted = True
        return results, X_test, y_test
    
    def predict(self, X, model_name='ensemble_stacking'):
        """Make predictions using specified model"""
        if not self.is_fitted:
            raise ValueError("Models not trained. Call train() first.")
        
        if isinstance(X, str):
            # Single SMILES
            X = pd.DataFrame([self.calculate_descriptors(X)])
            X = X[self.feature_columns].fillna(0)
        
        X_scaled = self.scalers['X'].transform(X)
        
        if model_name == 'best':
            # Use best performing model
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X_scaled)
            return predictions
        else:
            return self.models[model_name].predict(X_scaled)
    
    def save_models(self, path_prefix="/home/mpdr/.openclaw/workspace/toxicity_model/models"):
        """Save all models and scalers"""
        import os
        os.makedirs(path_prefix, exist_ok=True)
        
        joblib.dump(self.scalers['X'], f"{path_prefix}/scaler.pkl")
        for name, model in self.models.items():
            joblib.dump(model, f"{path_prefix}/{name}_model.pkl")
        print(f"\n✓ Models saved to {path_prefix}")
    
    def load_models(self, path_prefix="/home/mpdr/.openclaw/workspace/toxicity_model/models"):
        """Load models and scalers"""
        import os
        self.scalers['X'] = joblib.load(f"{path_prefix}/scaler.pkl")
        for name in os.listdir(path_prefix):
            if name.endswith('_model.pkl'):
                model_name = name.replace('_model.pkl', '')
                self.models[model_name] = joblib.load(f"{path_prefix}/{name}")
        self.is_fitted = True
        print(f"\n✓ Models loaded from {path_prefix}")


def main():
    print("="*70)
    print("TOXICITY PREDICTION MODEL TRAINING")
    print("="*70)
    
    # Load data
    data_file = "/home/mpdr/.openclaw/workspace/toxicity_model/Toxicity_Data_Combined.xlsx"
    
    try:
        df = pd.read_excel(data_file)
        print(f"✓ Loaded data: {len(df)} records")
    except Exception as e:
        print(f"✗ Could not load {data_file}: {e}")
        print("Using built-in sample data...")
        from data_collector import create_sample_dataset
        df = create_sample_dataset()
    
    # Initialize predictor
    predictor = ToxicityPredictor()
    
    # Prepare data
    X, y = predictor.prepare_data(df)
    print(f"✓ Features: {len(X.columns)}")
    print(f"✓ Target: {y.name} (non-null: {y.notna().sum()})")
    
    # Build models
    predictor.build_models()
    
    # Train and evaluate
    results, X_test, y_test = predictor.train(X, y)
    
    # Save models
    predictor.save_models()
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    summary = []
    for name, metrics in results.items():
        summary.append({
            'Model': name,
            'Test_RMSE': metrics['test_rmse'],
            'Test_MAE': metrics['test_mae'],
            'Test_R2': metrics['test_r2'],
            'Accuracy_10pct': metrics['accuracy_10'],
            'Accuracy_20pct': metrics['accuracy_20'],
            'Accuracy_50pct': metrics['accuracy_50']
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Test_R2', ascending=False)
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_excel("/home/mpdr/.openclaw/workspace/toxicity_model/model_results.xlsx", index=False)
    print("\n✓ Results saved to model_results.xlsx")
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()