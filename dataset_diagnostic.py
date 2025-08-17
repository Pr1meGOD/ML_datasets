#!/usr/bin/env python3
"""
Quick dataset diagnostic to identify training issues
"""

import pandas as pd
import numpy as np

def diagnose_dataset():
    """Quick diagnosis of your dataset"""
    
    print("🔍 QUICK DATASET DIAGNOSIS")
    print("=" * 40)
    
    # Load your dataset
    try:
        df = pd.read_csv("final_training_dataset.csv")
        print(f"✅ Dataset loaded: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Basic info
    print(f"\n📊 BASIC INFO:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check for required columns
    print(f"\n🔍 COLUMN CHECK:")
    print(f"  All columns: {list(df.columns)}")
    
    # Look for production/area columns
    prod_cols = [col for col in df.columns if 'production' in col.lower()]
    area_cols = [col for col in df.columns if 'area' in col.lower()]
    yield_cols = [col for col in df.columns if 'yield' in col.lower()]
    
    print(f"  Production columns: {prod_cols}")
    print(f"  Area columns: {area_cols}")
    print(f"  Yield columns: {yield_cols}")
    
    # Check if we can create yield column
    if prod_cols and area_cols:
        prod_col = prod_cols[0]
        area_col = area_cols[0]
        print(f"\n📈 YIELD CALCULATION CHECK:")
        print(f"  Using: {prod_col} / {area_col}")
        
        # Calculate yield
        df['Yield'] = df[prod_col] / df[area_col]
        
        # Remove invalid yields
        valid_yields = df[(df['Yield'] > 0) & (df['Yield'].notna()) & (np.isfinite(df['Yield']))]
        
        print(f"  Original rows: {len(df)}")
        print(f"  Valid yield rows: {len(valid_yields)}")
        print(f"  Yield range: {valid_yields['Yield'].min():.3f} - {valid_yields['Yield'].max():.3f}")
        print(f"  Yield mean: {valid_yields['Yield'].mean():.3f}")
        print(f"  Yield std: {valid_yields['Yield'].std():.3f}")
        print(f"  Unique yields: {valid_yields['Yield'].nunique()}")
        
        # Check for training viability
        if len(valid_yields) < 100:
            print(f"❌ CRITICAL: Only {len(valid_yields)} valid rows - insufficient for training!")
        elif len(valid_yields) < 1000:
            print(f"⚠️  WARNING: Only {len(valid_yields)} valid rows - training will be fast but may not be robust")
        else:
            print(f"✅ Sufficient data: {len(valid_yields)} valid rows")
        
        if valid_yields['Yield'].nunique() < 10:
            print(f"❌ CRITICAL: Only {valid_yields['Yield'].nunique()} unique yield values - model won't learn properly!")
        elif valid_yields['Yield'].nunique() < 50:
            print(f"⚠️  WARNING: Only {valid_yields['Yield'].nunique()} unique yield values - limited learning potential")
        
        # Check yield realism
        if valid_yields['Yield'].mean() > 10:
            print(f"⚠️  WARNING: Very high average yield ({valid_yields['Yield'].mean():.1f} t/ha) - check units")
        if valid_yields['Yield'].max() > 20:
            print(f"⚠️  WARNING: Extremely high max yield ({valid_yields['Yield'].max():.1f} t/ha) - possible data issues")
    
    else:
        print(f"❌ CRITICAL: Cannot find production/area columns to calculate yield!")
        if yield_cols:
            yield_col = yield_cols[0]
            print(f"  Found existing yield column: {yield_col}")
            print(f"  Yield range: {df[yield_col].min():.3f} - {df[yield_col].max():.3f}")
            print(f"  Valid yields: {df[yield_col].notna().sum()}")
    
    # Check categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    print(f"\n🏷️  CATEGORICAL COLUMNS:")
    for col in cat_cols:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"    Values: {list(df[col].unique())}")
        elif unique_count > 1000:
            print(f"    ⚠️  Very high cardinality - will slow training")
    
    # Check missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n❓ MISSING VALUES:")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print(f"\n✅ No missing values")
    
    # Final diagnosis
    print(f"\n🎯 DIAGNOSIS SUMMARY:")
    if len(df) < 100:
        print(f"❌ Dataset too small ({len(df)} rows) - this explains 2-3 second training")
    elif len(df) < 1000:
        print(f"⚠️  Small dataset ({len(df)} rows) - explains fast training")
    else:
        print(f"✅ Dataset size adequate ({len(df)} rows)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if len(df) < 1000:
        print(f"  • Get more data - at least 1000 rows recommended")
        print(f"  • Current size will lead to overfitting and poor generalization")
    
    if prod_cols and area_cols:
        df['Yield'] = df[prod_cols[0]] / df[area_cols[0]]
        valid_yields = df[(df['Yield'] > 0) & (df['Yield'].notna()) & (np.isfinite(df['Yield']))]
        if valid_yields['Yield'].nunique() < 50:
            print(f"  • Yield values too uniform - need more diverse conditions")
    
    print(f"  • Use the enhanced training script for better model development")

if __name__ == "__main__":
    diagnose_dataset()