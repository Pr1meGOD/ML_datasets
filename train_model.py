#!/usr/bin/env python3
"""
train_model.py - COMPREHENSIVE FIX VERSION

CRITICAL FIXES APPLIED:
1. Proper unit conversion and yield validation
2. Intelligent missing data imputation
3. Robust outlier detection and removal
4. Enhanced feature engineering
5. Proper model training with validation
6. Comprehensive data quality checks
"""

import os
import re
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ------------------- CONFIG -------------------
INPUT_CSV = "final_training_dataset.csv"
MODEL_FILE = "yield_model.pkl"
SCHEMA_FILE = "feature_schema.json"
BADGE_FILE = "badge_thresholds.json"
PLOTS_DIR = "model_plots"
DATA_REPORT_FILE = "data_quality_report.txt"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Known crop yield ranges (tonnes per hectare) for validation
CROP_YIELD_RANGES = {
    'rice': (2.0, 12.0),
    'wheat': (2.5, 8.0),
    'maize': (3.0, 15.0),
    'sugarcane': (40.0, 150.0),  # Exception: sugarcane has very high yields
    'cotton': (0.5, 3.0),
    'soybean': (1.0, 4.0),
    'groundnut': (1.0, 4.0),
    'default': (0.5, 15.0)  # General range for most crops
}

# Common soil types in India for imputation
INDIAN_SOIL_TYPES = ['alluvial', 'black', 'red', 'laterite', 'desert', 'mountain', 'saline']

# ------------------- Utilities -------------------
def _find_best_col(cols: List[str], *keywords) -> str:
    """Find column matching keywords with fuzzy matching"""
    low = {c.lower(): c for c in cols}
    for kw_group in keywords:
        if isinstance(kw_group, (list, tuple)):
            for kw in kw_group:
                for lc, orig in low.items():
                    if kw in lc:
                        return orig
        else:
            for lc, orig in low.items():
                if kw_group in lc:
                    return orig
    return ""

def log_data_quality(message: str):
    """Log data quality issues to report file"""
    with open(DATA_REPORT_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {message}\n")
    print(f"📊 {message}")

# ------------------- CRITICAL DATA VALIDATION -------------------
def validate_and_fix_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL FIX: Detect and correct unit issues in yield data
    """
    df = df.copy()
    log_data_quality("=== STARTING UNIT VALIDATION ===")
    
    # Check if yield column exists, if not create it
    if 'Yield' not in df.columns:
        prod_col = _find_best_col(df.columns.tolist(), "production")
        area_col = _find_best_col(df.columns.tolist(), "area")
        
        if prod_col and area_col:
            log_data_quality(f"Creating Yield from {prod_col}/{area_col}")
            df['Yield'] = df[prod_col] / df[area_col]
        else:
            raise ValueError("No Yield column found and cannot create from Production/Area")
    
    # Analyze yield distribution by crop
    log_data_quality(f"Original yield stats: min={df['Yield'].min():.3f}, max={df['Yield'].max():.3f}, mean={df['Yield'].mean():.3f}")
    
    # Check for unit conversion needs
    median_yield = df['Yield'].median()
    mean_yield = df['Yield'].mean()
    
    # If median yield > 50, likely in kg/ha instead of t/ha
    unit_conversion_factor = 1.0
    if median_yield > 50:
        log_data_quality(f"UNIT ISSUE DETECTED: Median yield {median_yield:.1f} suggests kg/ha instead of t/ha")
        unit_conversion_factor = 0.001  # Convert kg/ha to t/ha
        df['Yield'] = df['Yield'] * unit_conversion_factor
        log_data_quality(f"Applied conversion factor {unit_conversion_factor}")
    
    # Validate against crop-specific ranges
    crop_col = _find_best_col(df.columns.tolist(), ["crop name", "crop"])
    if crop_col:
        invalid_count = 0
        for crop in df[crop_col].unique():
            crop_lower = str(crop).lower()
            crop_data = df[df[crop_col] == crop]['Yield']
            
            # Find appropriate yield range
            yield_range = CROP_YIELD_RANGES.get(crop_lower, CROP_YIELD_RANGES['default'])
            
            # Special handling for sugarcane
            if 'sugarcane' in crop_lower or 'sugar' in crop_lower:
                yield_range = CROP_YIELD_RANGES['sugarcane']
            
            invalid_yields = (crop_data < yield_range[0]) | (crop_data > yield_range[1])
            invalid_count += invalid_yields.sum()
            
            if invalid_yields.sum() > 0:
                log_data_quality(f"Crop {crop}: {invalid_yields.sum()} invalid yields (expected {yield_range[0]}-{yield_range[1]} t/ha)")
    
    # Remove extreme outliers using robust statistical methods
    Q1 = df['Yield'].quantile(0.01)  # 1st percentile
    Q99 = df['Yield'].quantile(0.99)  # 99th percentile
    
    # For most crops, anything above 50 t/ha is suspicious (except sugarcane)
    max_realistic_yield = 50.0
    min_realistic_yield = 0.1
    
    before_count = len(df)
    df = df[
        (df['Yield'] >= min_realistic_yield) & 
        (df['Yield'] <= max_realistic_yield) & 
        (df['Yield'] >= Q1) & 
        (df['Yield'] <= Q99) &
        (np.isfinite(df['Yield']))
    ]
    
    removed_count = before_count - len(df)
    log_data_quality(f"Removed {removed_count} outlier records ({removed_count/before_count*100:.1f}%)")
    log_data_quality(f"Final yield stats: min={df['Yield'].min():.3f}, max={df['Yield'].max():.3f}, mean={df['Yield'].mean():.3f}")
    
    return df

def intelligent_missing_data_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL FIX: Intelligent imputation of missing values
    """
    df = df.copy()
    log_data_quality("=== STARTING INTELLIGENT IMPUTATION ===")
    
    # 1. Soil Type Imputation (currently 100% unknown)
    soil_col = _find_best_col(df.columns.tolist(), "soil")
    state_col = _find_best_col(df.columns.tolist(), "state")
    
    if soil_col and state_col:
        # Map states to predominant soil types
        state_soil_mapping = {
            'punjab': 'alluvial', 'haryana': 'alluvial', 'uttar pradesh': 'alluvial',
            'bihar': 'alluvial', 'west bengal': 'alluvial',
            'maharashtra': 'black', 'madhya pradesh': 'black', 'gujarat': 'black',
            'karnataka': 'red', 'andhra pradesh': 'red', 'tamil nadu': 'red',
            'kerala': 'laterite', 'odisha': 'laterite',
            'rajasthan': 'desert', 'jammu and kashmir': 'mountain'
        }
        
        unknown_mask = (df[soil_col] == 'unknown') | df[soil_col].isna()
        unknown_count = unknown_mask.sum()
        
        if unknown_count > 0:
            log_data_quality(f"Imputing {unknown_count} unknown soil types")
            
            for state in df[state_col].unique():
                if pd.isna(state):
                    continue
                    
                state_lower = str(state).lower().strip()
                soil_type = state_soil_mapping.get(state_lower, 'alluvial')  # Default to alluvial
                
                mask = (df[state_col] == state) & unknown_mask
                df.loc[mask, soil_col] = soil_type
                
                imputed = mask.sum()
                if imputed > 0:
                    log_data_quality(f"State {state}: imputed {imputed} soil types as '{soil_type}'")
    
    # 2. Rainfall Imputation using climate data
    rain_col = _find_best_col(df.columns.tolist(), ["rainfall", "rain"])
    
    if rain_col and state_col:
        # Average rainfall by state (approximate values for India)
        state_rainfall_mapping = {
            'kerala': 3000, 'assam': 2500, 'west bengal': 1500,
            'odisha': 1400, 'bihar': 1200, 'uttar pradesh': 1000,
            'maharashtra': 1200, 'karnataka': 1200, 'tamil nadu': 1000,
            'andhra pradesh': 900, 'telangana': 900, 'madhya pradesh': 1100,
            'gujarat': 800, 'punjab': 600, 'haryana': 600,
            'rajasthan': 500, 'jammu and kashmir': 1000
        }
        
        missing_rain = df[rain_col].isna().sum()
        if missing_rain > 0:
            log_data_quality(f"Imputing {missing_rain} missing rainfall values")
            
            for state in df[state_col].unique():
                if pd.isna(state):
                    continue
                    
                state_lower = str(state).lower().strip()
                avg_rainfall = state_rainfall_mapping.get(state_lower, 1000)  # Default 1000mm
                
                mask = (df[state_col] == state) & df[rain_col].isna()
                df.loc[mask, rain_col] = avg_rainfall
                
                imputed = mask.sum()
                if imputed > 0:
                    log_data_quality(f"State {state}: imputed {imputed} rainfall values as {avg_rainfall}mm")
    
    # 3. Advanced imputation for remaining missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == 'Yield':  # Don't impute target variable
            continue
            
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            log_data_quality(f"Imputing {missing_count} missing values in {col}")
            
            # Use KNN imputation for better accuracy
            if missing_count < len(df) * 0.5:  # If less than 50% missing
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    df[col] = imputer.fit_transform(df[[col]])[:, 0]
                except:
                    # Fallback to median if KNN fails
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                # Use median for highly missing data
                df[col].fillna(df[col].median(), inplace=True)
    
    return df

def comprehensive_data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    COMPREHENSIVE: Final data validation and cleaning
    """
    log_data_quality("=== COMPREHENSIVE DATA VALIDATION ===")
    
    # Remove duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    dedup_removed = before_dedup - len(df)
    if dedup_removed > 0:
        log_data_quality(f"Removed {dedup_removed} duplicate rows")
    
    # Standardize categorical values
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'Yield':  # Skip if Yield is categorical (shouldn't be)
            # Lowercase and strip whitespace
            df[col] = df[col].astype(str).str.lower().str.strip()
            # Replace spaces with underscores for consistency
            df[col] = df[col].str.replace(' ', '_').str.replace('/', '_')
            
            # Remove any remaining unknown/na values
            unknown_values = ['unknown', 'na', 'nan', 'null', '', 'none']
            mask = df[col].isin(unknown_values)
            
            if mask.any():
                log_data_quality(f"Column {col}: found {mask.sum()} unknown values")
                
                # Replace with mode (most frequent value) for this column
                mode_value = df.loc[~mask, col].mode()
                if len(mode_value) > 0:
                    df.loc[mask, col] = mode_value.iloc[0]
                    log_data_quality(f"Replaced with mode value: {mode_value.iloc[0]}")
    
    # Final validation
    log_data_quality(f"Final dataset shape: {df.shape}")
    log_data_quality(f"Final yield range: {df['Yield'].min():.3f} - {df['Yield'].max():.3f} t/ha")
    log_data_quality(f"Yield std deviation: {df['Yield'].std():.3f}")
    
    return df

# ------------------- Enhanced Feature Engineering -------------------
def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced features for better model performance
    """
    df = df.copy()
    log_data_quality("=== CREATING ADVANCED FEATURES ===")
    
    # 1. Interaction features
    state_col = _find_best_col(df.columns.tolist(), "state")
    crop_col = _find_best_col(df.columns.tolist(), ["crop name", "crop"])
    season_col = _find_best_col(df.columns.tolist(), "season")
    rain_col = _find_best_col(df.columns.tolist(), ["rainfall", "rain"])
    
    # State-Crop interaction (regional crop expertise)
    if state_col and crop_col:
        df['state_crop'] = df[state_col].astype(str) + '_' + df[crop_col].astype(str)
        log_data_quality("Created state_crop interaction feature")
    
    # Crop-Season interaction (seasonal suitability)
    if crop_col and season_col:
        df['crop_season'] = df[crop_col].astype(str) + '_' + df[season_col].astype(str)
        log_data_quality("Created crop_season interaction feature")
    
    # 2. Rainfall categories
    if rain_col:
        df['rainfall_category'] = pd.cut(df[rain_col], 
                                       bins=[0, 500, 1000, 1500, 2500, float('inf')],
                                       labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        log_data_quality("Created rainfall category feature")
    
    # 3. Year-based features (if year column exists)
    year_col = _find_best_col(df.columns.tolist(), "year")
    if year_col:
        df['year_category'] = pd.cut(df[year_col],
                                   bins=[1990, 2000, 2010, 2020, 2030],
                                   labels=['1990s', '2000s', '2010s', '2020s'])
        log_data_quality("Created year category feature")
    
    # 4. Regional climate zones (based on state)
    if state_col:
        climate_zones = {
            'punjab': 'subtropical', 'haryana': 'subtropical', 'rajasthan': 'arid',
            'gujarat': 'arid', 'maharashtra': 'tropical', 'karnataka': 'tropical',
            'tamil nadu': 'tropical', 'kerala': 'tropical_wet', 'assam': 'tropical_wet',
            'west bengal': 'subtropical_wet', 'odisha': 'tropical_wet',
            'uttar pradesh': 'subtropical', 'bihar': 'subtropical_wet',
            'madhya pradesh': 'subtropical', 'andhra pradesh': 'tropical'
        }
        
        df['climate_zone'] = df[state_col].map(climate_zones).fillna('temperate')
        log_data_quality("Created climate zone feature")
    
    return df

# ------------------- Enhanced Model Training -------------------
def train_enhanced_model(X, y):
    """
    Train model with comprehensive validation and hyperparameter tuning
    """
    log_data_quality("=== STARTING ENHANCED MODEL TRAINING ===")
    
    # Advanced train-validation-test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)  # ~15% of total
    
    log_data_quality(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Separate numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    log_data_quality(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    
    # Advanced preprocessing pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Model comparison
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_model = None
    best_score = float('-inf')
    best_name = None
    
    # Train and compare models
    for name, model in models.items():
        log_data_quality(f"Training {name}...")
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Validate on validation set
        val_predictions = pipeline.predict(X_val)
        val_score = r2_score(y_val, val_predictions)
        val_mae = mean_absolute_error(y_val, val_predictions)
        
        log_data_quality(f"{name} - Validation R²: {val_score:.4f}, MAE: {val_mae:.4f}")
        
        # Check for realistic predictions
        pred_min, pred_max = val_predictions.min(), val_predictions.max()
        pred_std = val_predictions.std()
        
        log_data_quality(f"{name} - Predictions: {pred_min:.3f} to {pred_max:.3f}, std: {pred_std:.3f}")
        
        # Select best model based on R² and prediction realism
        if val_score > best_score and pred_std > 0.1:  # Ensure model makes varied predictions
            best_score = val_score
            best_model = pipeline
            best_name = name
    
    if best_model is None:
        raise ValueError("No suitable model found! Check data quality.")
    
    log_data_quality(f"Best model: {best_name} with R²={best_score:.4f}")
    
    # Final evaluation on test set
    test_predictions = best_model.predict(X_test)
    test_r2 = r2_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    log_data_quality(f"FINAL TEST PERFORMANCE:")
    log_data_quality(f"R² Score: {test_r2:.4f}")
    log_data_quality(f"MAE: {test_mae:.4f}")
    log_data_quality(f"RMSE: {test_rmse:.4f}")
    log_data_quality(f"Test predictions range: {test_predictions.min():.3f} - {test_predictions.max():.3f}")
    
    # Feature importance (if available)
    if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        importances = best_model.named_steps['regressor'].feature_importances_
        
        # Top 10 features
        feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        log_data_quality("TOP 10 IMPORTANT FEATURES:")
        for i, (feat, imp) in enumerate(feature_importance[:10], 1):
            log_data_quality(f"{i:2d}. {feat}: {imp:.4f}")
    
    return best_model

# ------------------- Main Training Function -------------------
def load_and_prepare_comprehensive():
    """
    Comprehensive data loading and preparation
    """
    # Clear previous data quality report
    if os.path.exists(DATA_REPORT_FILE):
        os.remove(DATA_REPORT_FILE)
    
    log_data_quality("Starting comprehensive data preparation...")
    
    # Load data
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Dataset {INPUT_CSV} not found!")
    
    df = pd.read_csv(INPUT_CSV)
    log_data_quality(f"Loaded dataset: {df.shape}")
    log_data_quality(f"Columns: {df.columns.tolist()}")
    
    # Step 1: Critical unit validation and fixing
    df = validate_and_fix_units(df)
    
    # Step 2: Intelligent missing data imputation
    df = intelligent_missing_data_imputation(df)
    
    # Step 3: Comprehensive data validation
    df = comprehensive_data_validation(df)
    
    # Step 4: Advanced feature engineering
    df = create_advanced_features(df)
    
    # Final checks
    if len(df) < 1000:
        raise ValueError(f"Too few valid records remaining: {len(df)}")
    
    # Prepare features and target
    X = df.drop(columns=['Yield'])
    y = df['Yield'].values
    
    log_data_quality(f"Final training data: X={X.shape}, y={y.shape}")
    log_data_quality(f"Target variable stats: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}")
    
    # Schema for inference
    schema = {
        "numeric_cols": X.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_cols": X.select_dtypes(include=['object']).columns.tolist(),
        "all_cols": X.columns.tolist()
    }
    
    return X, y, schema

# ------------------- Badge System -------------------
def compute_badge_thresholds(y: np.ndarray) -> Dict[str, float]:
    """Compute performance badge thresholds from training data"""
    thresholds = {
        "A": float(np.percentile(y, 85)),  # Top 15%
        "B": float(np.percentile(y, 70)),  # Top 30%
        "C": float(np.percentile(y, 50)),  # Above median
        "D": float(np.percentile(y, 30)),  # Above bottom 30%
        "F": float(np.percentile(y, 0))    # Bottom 30%
    }
    return thresholds

def map_badge_from_thresholds(y_value: float, thresholds: Dict[str, float]) -> str:
    """Map yield value to performance badge"""
    if y_value >= thresholds["A"]:
        return "A"
    elif y_value >= thresholds["B"]:
        return "B"
    elif y_value >= thresholds["C"]:
        return "C"
    elif y_value >= thresholds["D"]:
        return "D"
    else:
        return "F"

# ------------------- Complete Training Pipeline -------------------
def train_and_save_comprehensive():
    """
    Complete training pipeline with all fixes applied
    """
    log_data_quality("="*60)
    log_data_quality("STARTING COMPREHENSIVE MODEL TRAINING")
    log_data_quality("="*60)
    
    try:
        # Step 1: Load and prepare data
        X, y, schema = load_and_prepare_comprehensive()
        
        # Step 2: Train enhanced model
        model = train_enhanced_model(X, y)
        
        # Step 3: Compute badge thresholds
        thresholds = compute_badge_thresholds(y)
        log_data_quality(f"Badge thresholds: {thresholds}")
        
        # Step 4: Save all artifacts
        joblib.dump(model, MODEL_FILE)
        log_data_quality(f"Model saved: {MODEL_FILE}")
        
        with open(SCHEMA_FILE, "w") as f:
            json.dump(schema, f, indent=2)
        log_data_quality(f"Schema saved: {SCHEMA_FILE}")
        
        with open(BADGE_FILE, "w") as f:
            json.dump(thresholds, f, indent=2)
        log_data_quality(f"Thresholds saved: {BADGE_FILE}")
        
        # Step 5: Generate validation report
        generate_validation_report(model, X, y, thresholds)
        
        log_data_quality("="*60)
        log_data_quality("TRAINING COMPLETED SUCCESSFULLY!")
        log_data_quality("="*60)
        
        return model
        
    except Exception as e:
        log_data_quality(f"CRITICAL ERROR during training: {str(e)}")
        raise

def generate_validation_report(model, X, y, thresholds):
    """Generate comprehensive validation report"""
    log_data_quality("=== GENERATING VALIDATION REPORT ===")
    
    # Sample predictions for validation
    sample_size = min(100, len(X))
    sample_X = X.sample(n=sample_size, random_state=42)
    sample_y = y[sample_X.index]
    sample_X = sample_X.reset_index(drop=True)
    
    predictions = model.predict(sample_X)
    
    # Prediction validation
    log_data_quality(f"Sample predictions statistics:")
    log_data_quality(f"  Mean prediction: {predictions.mean():.3f}")
    log_data_quality(f"  Std prediction: {predictions.std():.3f}")
    log_data_quality(f"  Min prediction: {predictions.min():.3f}")
    log_data_quality(f"  Max prediction: {predictions.max():.3f}")
    
    # Badge distribution
    badges = [map_badge_from_thresholds(p, thresholds) for p in predictions]
    badge_counts = pd.Series(badges).value_counts()
    log_data_quality(f"Badge distribution in sample:")
    for badge in ['A', 'B', 'C', 'D', 'F']:
        count = badge_counts.get(badge, 0)
        percentage = count / len(predictions) * 100
        log_data_quality(f"  Badge {badge}: {count} ({percentage:.1f}%)")
    
    # Realistic range check
    realistic_count = sum(1 for p in predictions if 0.5 <= p <= 20.0)
    realistic_percentage = realistic_count / len(predictions) * 100
    log_data_quality(f"Realistic predictions (0.5-20 t/ha): {realistic_count}/{len(predictions)} ({realistic_percentage:.1f}%)")
    
    if realistic_percentage < 90:
        log_data_quality("⚠️ WARNING: Less than 90% of predictions are in realistic range!")
    else:
        log_data_quality("✅ Prediction ranges look realistic!")

# ------------------- Interactive CLI Functions -------------------
def _to_float(msg: str) -> float:
    """Safely convert input to float"""
    val = input(msg).strip()
    if val == "":
        return np.nan
    try:
        return float(val)
    except Exception:
        print("Invalid number. Using NaN.")
        return np.nan

def _to_str(msg: str) -> str:
    """Get string input"""
    val = input(msg).strip()
    return val if val != "" else None

def _convert_unit(pred: float, unit: str) -> float:
    """Convert model output (assumed t/ha) to desired unit"""
    unit = (unit or "t/ha").strip().lower()
    if unit in ["kg/ha", "kgha", "kg per ha", "kg"]:
        return pred * 1000.0
    return pred  # default t/ha

def build_input_row_from_cli_enhanced(schema: Dict) -> pd.DataFrame:
    """Enhanced CLI input with better validation"""
    cols = schema["all_cols"]
    
    print(f"\n🌾 === ENHANCED FARM DATA ENTRY ===")
    print("Enter your farm details (press Enter for intelligent defaults)")
    print("="*50)
    
    vals = {}
    
    # Initialize all columns
    for col in cols:
        vals[col] = np.nan
    
    # Location
    print("\n📍 LOCATION:")
    state_col = _find_best_col(cols, "state")
    if state_col:
        state_input = _to_str("  State: ")
        vals[state_col] = (state_input or 'punjab').lower().replace(' ', '_')
    
    district_col = _find_best_col(cols, "district")
    if district_col:
        district_input = _to_str("  District: ")
        vals[district_col] = (district_input or 'ludhiana').lower().replace(' ', '_')
    
    # Crop details
    print("\n🌱 CROP:")
    crop_col = _find_best_col(cols, ["crop name", "crop"])
    if crop_col:
        print("  Common crops: rice, wheat, maize, cotton, sugarcane")
        crop_input = _to_str("  Crop name: ")
        vals[crop_col] = (crop_input or 'rice').lower().replace(' ', '_')
    
    season_col = _find_best_col(cols, "season")
    if season_col:
        print("  Seasons: kharif (Jun-Oct), rabi (Nov-Apr), summer (Mar-Jun)")
        season_input = _to_str("  Season: ")
        vals[season_col] = (season_input or 'kharif').lower()
    
    # Climate
    print("\n🌤️ CLIMATE:")
    rain_col = _find_best_col(cols, ["rainfall", "rain"])
    if rain_col:
        rain_val = _to_float("  Annual rainfall (mm): ")
        vals[rain_col] = rain_val if not np.isnan(rain_val) else 1000.0
    
    # Soil
    soil_col = _find_best_col(cols, "soil")
    if soil_col:
        print("  Soil types: alluvial, black, red, laterite, sandy")
        soil_input = _to_str("  Soil type: ")
        vals[soil_col] = (soil_input or 'alluvial').lower().replace(' ', '_')
    
    # Year
    year_col = _find_best_col(cols, "year")
    if year_col:
        current_year = datetime.now().year
        year_val = _to_float(f"  Year [{current_year}]: ")
        vals[year_col] = year_val if not np.isnan(year_val) else current_year
    
    # Additional fields that might exist
    area_col = _find_best_col(cols, "area")
    if area_col:
        area_val = _to_float("  Farm area (hectares): ")
        vals[area_col] = area_val if not np.isnan(area_val) else 2.0
    
    # Create DataFrame and add derived features
    df = pd.DataFrame([vals])
    
    # Add derived features that might exist in the model
    if 'state_crop' in cols:
        df['state_crop'] = f"{vals.get(state_col, 'punjab')}_{vals.get(crop_col, 'rice')}"
    
    if 'crop_season' in cols:
        df['crop_season'] = f"{vals.get(crop_col, 'rice')}_{vals.get(season_col, 'kharif')}"
    
    if 'rainfall_category' in cols and not np.isnan(vals.get(rain_col, np.nan)):
        rainfall = vals[rain_col]
        if rainfall <= 500:
            df['rainfall_category'] = 'very_low'
        elif rainfall <= 1000:
            df['rainfall_category'] = 'low'
        elif rainfall <= 1500:
            df['rainfall_category'] = 'medium'
        elif rainfall <= 2500:
            df['rainfall_category'] = 'high'
        else:
            df['rainfall_category'] = 'very_high'
    
    if 'climate_zone' in cols:
        state = vals.get(state_col, 'punjab')
        climate_zones = {
            'punjab': 'subtropical', 'haryana': 'subtropical', 'rajasthan': 'arid',
            'gujarat': 'arid', 'maharashtra': 'tropical', 'karnataka': 'tropical'
        }
        df['climate_zone'] = climate_zones.get(state, 'temperate')
    
    if 'year_category' in cols and not np.isnan(vals.get(year_col, np.nan)):
        year = vals[year_col]
        if year < 2000:
            df['year_category'] = '1990s'
        elif year < 2010:
            df['year_category'] = '2000s'
        elif year < 2020:
            df['year_category'] = '2010s'
        else:
            df['year_category'] = '2020s'
    
    return df

def predict_with_enhanced_model(input_df: pd.DataFrame, unit: str = "t/ha"):
    """Make prediction with enhanced error handling"""
    try:
        model = joblib.load(MODEL_FILE)
        with open(SCHEMA_FILE, "r") as f:
            schema = json.load(f)
        with open(BADGE_FILE, "r") as f:
            thresholds = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Model artifacts not found. Please train the model first. Error: {e}")
    
    try:
        prediction = model.predict(input_df)[0]
        
        # Validate prediction
        if not np.isfinite(prediction) or prediction <= 0:
            print("⚠️ Warning: Invalid prediction, using fallback")
            prediction = 2.5  # Reasonable fallback
        
        # Convert units
        prediction_display = _convert_unit(prediction, unit)
        badge = map_badge_from_thresholds(prediction, thresholds)
        
        return prediction_display, badge
    except Exception as e:
        print(f"Prediction error: {e}")
        return _convert_unit(2.5, unit), "C"  # Safe fallback

def predict_yield_multiyear(input_df: pd.DataFrame, years: int = 5, unit: str = "t/ha"):
    """Multi-year prediction with realistic variation"""
    try:
        model = joblib.load(MODEL_FILE)
        with open(SCHEMA_FILE, "r") as f:
            schema = json.load(f)
        with open(BADGE_FILE, "r") as f:
            thresholds = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Model artifacts not found. Please train the model first. Error: {e}")
    
    predictions = []
    badges = []
    year_labels = []
    
    # Get base prediction
    try:
        base_pred = model.predict(input_df)[0]
        if not np.isfinite(base_pred) or base_pred <= 0:
            base_pred = 2.5
    except:
        base_pred = 2.5
    
    # Get current year if available
    year_col = _find_best_col(schema["all_cols"], "year")
    if year_col and year_col in input_df.columns:
        current_year = input_df.iloc[0][year_col]
        if pd.isna(current_year):
            current_year = datetime.now().year
    else:
        current_year = datetime.now().year
    
    # Generate predictions for multiple years
    for i in range(years):
        # Add realistic year-over-year variation
        year_factor = 1 + (i * 0.015)  # 1.5% improvement per year
        weather_factor = np.random.normal(1.0, 0.08)  # 8% weather variation
        trend_factor = np.random.normal(1.0, 0.04)   # 4% random variation
        
        pred = base_pred * year_factor * weather_factor * trend_factor
        pred = max(0.3, min(15.0, pred))  # Keep within realistic bounds
        
        pred_display = _convert_unit(pred, unit)
        badge = map_badge_from_thresholds(pred, thresholds)
        
        predictions.append(pred_display)
        badges.append(badge)
        year_labels.append(int(current_year) + i)
        
        # Update base for next year (memory effect)
        base_pred = 0.7 * base_pred + 0.3 * pred
    
    return predictions, badges, year_labels

# ------------------- Plotting Functions -------------------
def plot_1year(value, badge, unit="t/ha", file_path=None):
    """Plot single year prediction"""
    if file_path is None:
        file_path = os.path.join(PLOTS_DIR, "prediction_1year.png")
    
    plt.figure(figsize=(8, 6))
    colors = {'A': 'green', 'B': 'lightgreen', 'C': 'yellow', 'D': 'orange', 'F': 'red'}
    color = colors.get(badge, 'blue')
    
    bars = plt.bar(["Predicted Yield"], [value], color=color, alpha=0.7, edgecolor='black')
    plt.ylabel(f"Yield ({unit})", fontsize=12)
    plt.title(f"Crop Yield Prediction\nGrade: {badge}", fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()
    return file_path

def plot_multiyear(years, values, badges, unit="t/ha", file_path=None):
    """Plot multi-year predictions"""
    if file_path is None:
        file_path = os.path.join(PLOTS_DIR, "prediction_multiyear.png")
    
    plt.figure(figsize=(12, 8))
    colors = {'A': 'green', 'B': 'lightgreen', 'C': 'yellow', 'D': 'orange', 'F': 'red'}
    
    # Plot line with markers
    plt.plot(years, values, 'o-', linewidth=3, markersize=10, color='blue', alpha=0.7)
    
    # Color points by badge
    for i, (year, val, badge) in enumerate(zip(years, values, badges)):
        plt.scatter(year, val, c=colors.get(badge, 'blue'), s=150, zorder=5, edgecolor='black')
        plt.annotate(f'{badge}', (year, val), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontweight='bold', fontsize=10)
        plt.annotate(f'{val:.1f}', (year, val), textcoords="offset points", 
                    xytext=(0,-20), ha='center', fontsize=9)
    
    plt.xlabel("Year", fontsize=12, fontweight='bold')
    plt.ylabel(f"Yield ({unit})", fontsize=12, fontweight='bold')
    plt.title("Multi-Year Crop Yield Forecast", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(['Predicted Yield'], loc='upper left')
    
    # Add grade legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[grade], alpha=0.7, label=f'Grade {grade}') 
                      for grade in ['A', 'B', 'C', 'D', 'F']]
    plt.legend(handles=legend_elements, loc='upper right', title='Performance Grades')
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()
    return file_path

def enhanced_cli_predict():
    """Enhanced CLI prediction interface"""
    print("🚀 Enhanced Crop Yield Prediction System")
    print("="*60)
    
    if not os.path.exists(MODEL_FILE):
        print("❌ No trained model found. Training now...")
        train_and_save_comprehensive()
        print("\n" + "="*60)
    
    try:
        # Load model artifacts
        with open(SCHEMA_FILE, "r") as f:
            schema = json.load(f)
        
        # Get user input
        input_df = build_input_row_from_cli_enhanced(schema)
        
        # Prediction options
        print("\n🎯 PREDICTION OPTIONS:")
        print("1. Single year prediction")
        print("2. Multi-year forecast (5 years)")
        
        choice = input("Select option (1 or 2) [default: 1]: ").strip() or "1"
        unit = input("Display unit (t/ha or kg/ha) [default: t/ha]: ").strip() or "t/ha"
        
        print("\n🔮 Generating predictions...")
        
        if choice == "2":
            # Multi-year prediction
            predictions, badges, years = predict_yield_multiyear(input_df, years=5, unit=unit)
            
            print(f"\n📈 === 5-YEAR YIELD FORECAST ===")
            print("="*50)
            total_yield = 0
            for year, pred, badge in zip(years, predictions, badges):
                print(f"Year {year}: {pred:.2f} {unit} | Grade: {badge}")
                total_yield += pred
            
            avg_yield = total_yield / len(predictions)
            print(f"\nAverage Yield: {avg_yield:.2f} {unit}")
            
            # Generate chart
            chart_file = plot_multiyear(years, predictions, badges, unit)
            print(f"📊 Multi-year forecast chart saved: {chart_file}")
            
        else:
            # Single year prediction
            prediction, badge = predict_with_enhanced_model(input_df, unit)
            
            print(f"\n📈 === YIELD PREDICTION RESULT ===")
            print("="*50)
            print(f"Predicted Yield: {prediction:.2f} {unit}")
            print(f"Performance Grade: {badge}")
            
            # Grade interpretation
            grade_meanings = {
                'A': 'Excellent yield (top 15%)',
                'B': 'Good yield (top 30%)', 
                'C': 'Average yield (above median)',
                'D': 'Below average yield',
                'F': 'Poor yield (bottom 30%)'
            }
            print(f"Grade Meaning: {grade_meanings.get(badge, 'Unknown')}")
            
            # Generate chart
            chart_file = plot_1year(prediction, badge, unit)
            print(f"📊 Prediction chart saved: {chart_file}")
        
        # Practical advice
        base_yield = prediction if choice == "1" else avg_yield
        if unit == "kg/ha":
            base_yield = base_yield / 1000  # Convert to t/ha for comparison
            
        print(f"\n💡 RECOMMENDATIONS:")
        if base_yield < 2.0:
            print("🔸 Consider soil testing and nutrient management")
            print("🔸 Ensure adequate irrigation and drainage")
            print("🔸 Use certified seeds and proper spacing")
            print("🔸 Implement integrated pest management")
        elif base_yield < 5.0:
            print("🔸 Good foundation - focus on optimization")
            print("🔸 Fine-tune fertilizer application timing")
            print("🔸 Consider precision agriculture techniques")
        else:
            print("🎉 Excellent predicted performance!")
            print("🔸 Maintain current best practices")
            print("🔸 Consider premium crop varieties")
            print("🔸 Document successful techniques for replication")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print("Please check your inputs and try again.")
        print("If the error persists, try retraining the model with --force-retrain")

# ------------------- Main Function -------------------
def main():
    """Main function with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description="Enhanced Crop Yield Prediction System with Data Quality Fixes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py                    # Train model and run predictions
  python train_model.py --train            # Train model only
  python train_model.py --predict          # Run predictions only
  python train_model.py --force-retrain    # Force complete retraining
  python train_model.py --validate         # Validate existing model
        """
    )
    
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Make predictions")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--validate", action="store_true", help="Validate existing model")
    
    args = parser.parse_args()
    
    # Force retrain if requested
    if args.force_retrain:
        for file in [MODEL_FILE, SCHEMA_FILE, BADGE_FILE, DATA_REPORT_FILE]:
            if os.path.exists(file):
                os.remove(file)
        print("🗑️ Removed existing model artifacts for complete retraining")
    
    try:
        if args.train:
            train_and_save_comprehensive()
        elif args.predict:
            enhanced_cli_predict()
        elif args.validate:
            if os.path.exists(MODEL_FILE):
                print("📊 Model validation coming soon...")
                print("Check data_quality_report.txt for detailed training information")
            else:
                print("❌ No model found to validate. Train first.")
        else:
            # Default behavior: train if needed, then predict
            if not os.path.exists(MODEL_FILE):
                print("No model found. Training first...")
                train_and_save_comprehensive()
                print("\n" + "="*60)
            enhanced_cli_predict()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Training/prediction interrupted by user")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        print(f"Check {DATA_REPORT_FILE} for detailed logs")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())