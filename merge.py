#!/usr/bin/env python3
"""
Memory-efficient merge script: handles large datasets by processing in chunks
and optimizing memory usage throughout the merge process.
"""

import pandas as pd
import numpy as np
import unicodedata
import re
from pathlib import Path
import gc  # For garbage collection

# ---------- Configuration ----------
CHUNK_SIZE = 50000  # Process datasets in chunks
PATH_BILL = Path("billashot_extended_final_dataset.csv")
PATH_R13 = Path("dataset13.csv")
PATH_R14 = Path("dataset14.csv")
OUT_PATH = Path("billashot_final_merged.csv")

# ---------- Memory-efficient helpers ----------
def normalize_text(s):
    """Lightweight text normalization"""
    if pd.isnull(s):
        return ""
    s = str(s).strip().lower()
    # Remove special characters and extra spaces
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def optimize_dtypes(df):
    """Optimize DataFrame memory usage by downcasting numeric types"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            # Check if we can downcast
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                # Integer downcasting
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    
            elif str(col_type)[:5] == 'float':
                # Float downcasting
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            # Convert object columns to category if they have few unique values
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
    
    return df

def safe_read_csv(path, **kwargs):
    """Memory-efficient CSV reading with error handling"""
    try:
        # Try reading with different separators
        separators = [',', ';', '\t']
        for sep in separators:
            try:
                df = pd.read_csv(path, sep=sep, low_memory=False, **kwargs)
                if len(df.columns) > 1:  # Successfully parsed
                    print(f"✅ Read {path} with separator '{sep}'")
                    return df
            except Exception:
                continue
        
        # Fallback to default
        df = pd.read_csv(path, low_memory=False, **kwargs)
        return df
    except Exception as e:
        print(f"❌ Error reading {path}: {str(e)}")
        return None

def process_climatology_data(r13_path):
    """Process dataset13 to get rainfall data for state/district"""
    print("🌡️ Processing climatology data...")
    
    r13 = safe_read_csv(r13_path)
    if r13 is None:
        return pd.DataFrame()
    
    print(f"Dataset13 shape: {r13.shape}")
    print(f"Dataset13 columns: {list(r13.columns)[:10]}...")  # Show first 10
    
    # Find state and district columns
    state_col = None
    district_col = None
    
    for col in r13.columns:
        col_lower = col.lower()
        if 'state' in col_lower and state_col is None:
            state_col = col
        if ('district' in col_lower or 'dist' in col_lower) and district_col is None:
            district_col = col
    
    if not state_col or not district_col:
        print("⚠️ Could not find state/district columns in dataset13")
        return pd.DataFrame()
    
    # Create normalized keys
    r13['state_norm'] = r13[state_col].astype(str).apply(normalize_text)
    r13['district_norm'] = r13[district_col].astype(str).apply(normalize_text)
    
    # Find numeric columns (likely rainfall data)
    numeric_cols = []
    for col in r13.columns:
        if col not in [state_col, district_col, 'state_norm', 'district_norm']:
            try:
                r13[col] = pd.to_numeric(r13[col], errors='coerce')
                if not r13[col].isna().all():
                    numeric_cols.append(col)
            except:
                pass
    
    if not numeric_cols:
        print("⚠️ No numeric columns found in dataset13")
        return pd.DataFrame()
    
    # Calculate average rainfall per state/district (for filling missing data)
    r13['rainfall_sum'] = r13[numeric_cols].sum(axis=1)
    
    # Group by state/district - get average rainfall
    rainfall_lookup = r13.groupby(['state_norm', 'district_norm']).agg({
        'rainfall_sum': 'mean'
    }).reset_index()
    
    # Rename to match our target column
    rainfall_lookup = rainfall_lookup.rename(columns={'rainfall_sum': 'Rainfall (mm)'})
    
    rainfall_lookup = optimize_dtypes(rainfall_lookup)
    print(f"✅ Climatology processed: {rainfall_lookup.shape}")
    
    # Clean up memory
    del r13
    gc.collect()
    
    return rainfall_lookup

def process_timeseries_data(r14_path):
    """Process dataset14 to get annual rainfall data for state/district/year"""
    print("📈 Processing timeseries data...")
    
    r14 = safe_read_csv(r14_path)
    if r14 is None:
        return pd.DataFrame()
    
    print(f"Dataset14 shape: {r14.shape}")
    print(f"Dataset14 columns: {list(r14.columns)}")
    
    # Find key columns
    state_col = None
    district_col = None
    year_col = None
    rainfall_col = None
    
    for col in r14.columns:
        col_lower = col.lower()
        if 'state' in col_lower and state_col is None:
            state_col = col
        if ('district' in col_lower or 'dist' in col_lower) and district_col is None:
            district_col = col
        if 'year' in col_lower and year_col is None:
            year_col = col
        if 'rain' in col_lower and rainfall_col is None:
            rainfall_col = col
    
    if not all([state_col, district_col, year_col, rainfall_col]):
        print(f"⚠️ Missing key columns in dataset14:")
        print(f"   State: {state_col}, District: {district_col}")
        print(f"   Year: {year_col}, Rainfall: {rainfall_col}")
        return pd.DataFrame()
    
    # Normalize and convert types
    r14['state_norm'] = r14[state_col].astype(str).apply(normalize_text)
    r14['district_norm'] = r14[district_col].astype(str).apply(normalize_text)
    r14['year_clean'] = pd.to_numeric(r14[year_col], errors='coerce')
    r14['rainfall_clean'] = pd.to_numeric(r14[rainfall_col], errors='coerce')
    
    # Remove invalid data
    r14 = r14.dropna(subset=['year_clean', 'rainfall_clean'])
    
    # Calculate annual rainfall totals per state/district/year
    annual_rainfall = r14.groupby(['state_norm', 'district_norm', 'year_clean']).agg({
        'rainfall_clean': 'sum'  # Sum monthly to get annual total
    }).reset_index()
    
    # Rename to match our target column
    annual_rainfall.columns = ['state_norm', 'district_norm', 'year_clean', 'Rainfall (mm)']
    
    annual_rainfall = optimize_dtypes(annual_rainfall)
    print(f"✅ Timeseries processed: {annual_rainfall.shape}")
    
    # Clean up memory
    del r14
    gc.collect()
    
    return annual_rainfall

def merge_datasets_chunked(bill_path, climatology_rainfall, timeseries_rainfall):
    """Memory-efficient chunked merging - fills Rainfall (mm) column"""
    print("🔗 Starting chunked merge...")
    
    # Get total rows for progress tracking
    total_rows = sum(1 for _ in open(bill_path)) - 1  # Subtract header
    print(f"Total rows to process: {total_rows:,}")
    
    # Process in chunks
    merged_chunks = []
    processed_rows = 0
    
    for chunk_num, bill_chunk in enumerate(pd.read_csv(bill_path, chunksize=CHUNK_SIZE)):
        processed_rows += len(bill_chunk)
        print(f"Processing chunk {chunk_num + 1} ({processed_rows:,}/{total_rows:,} rows)")
        
        # Find state and district columns in billashot
        state_col = None
        district_col = None
        year_col = None
        
        for col in bill_chunk.columns:
            col_lower = col.lower()
            if 'state' in col_lower and state_col is None:
                state_col = col
            if ('district' in col_lower or 'dist' in col_lower) and district_col is None:
                district_col = col
            if 'year' in col_lower and year_col is None:
                year_col = col
        
        if not state_col or not district_col:
            print("⚠️ Could not find state/district columns in billashot chunk")
            merged_chunks.append(bill_chunk)
            continue
        
        # Normalize keys
        bill_chunk['state_norm'] = bill_chunk[state_col].astype(str).apply(normalize_text)
        bill_chunk['district_norm'] = bill_chunk[district_col].astype(str).apply(normalize_text)
        
        # Convert existing Rainfall (mm) to numeric if it exists
        rainfall_col_exists = 'Rainfall (mm)' in bill_chunk.columns
        if rainfall_col_exists:
            bill_chunk['Rainfall (mm)'] = pd.to_numeric(bill_chunk['Rainfall (mm)'], errors='coerce')
        else:
            # Create the column if it doesn't exist
            bill_chunk['Rainfall (mm)'] = np.nan
        
        # Priority 1: Fill from timeseries data (year-specific)
        if not timeseries_rainfall.empty and year_col:
            bill_chunk['year_clean'] = pd.to_numeric(bill_chunk[year_col], errors='coerce')
            
            # Create temporary merge for rainfall data
            temp_merge = bill_chunk.merge(
                timeseries_rainfall,
                on=['state_norm', 'district_norm', 'year_clean'],
                how='left',
                suffixes=('', '_timeseries')
            )
            
            # Fill missing rainfall data with timeseries data
            mask = bill_chunk['Rainfall (mm)'].isna() & temp_merge['Rainfall (mm)_timeseries'].notna()
            bill_chunk.loc[mask, 'Rainfall (mm)'] = temp_merge.loc[mask, 'Rainfall (mm)_timeseries']
            
            filled_from_timeseries = mask.sum()
            if filled_from_timeseries > 0:
                print(f"   ✅ Filled {filled_from_timeseries} rows from timeseries data")
        
        # Priority 2: Fill remaining missing values from climatology data
        if not climatology_rainfall.empty:
            temp_merge = bill_chunk.merge(
                climatology_rainfall,
                on=['state_norm', 'district_norm'],
                how='left',
                suffixes=('', '_climate')
            )
            
            # Fill remaining missing rainfall data with climatology
            mask = bill_chunk['Rainfall (mm)'].isna() & temp_merge['Rainfall (mm)_climate'].notna()
            bill_chunk.loc[mask, 'Rainfall (mm)'] = temp_merge.loc[mask, 'Rainfall (mm)_climate']
            
            filled_from_climate = mask.sum()
            if filled_from_climate > 0:
                print(f"   ✅ Filled {filled_from_climate} rows from climatology data")
        
        # Clean up helper columns
        helper_cols = ['state_norm', 'district_norm', 'year_clean']
        bill_chunk = bill_chunk.drop(columns=[col for col in helper_cols if col in bill_chunk.columns])
        
        # Show rainfall fill statistics for this chunk
        if rainfall_col_exists:
            total_rainfall = len(bill_chunk)
            filled_rainfall = bill_chunk['Rainfall (mm)'].notna().sum()
            print(f"   📊 Chunk {chunk_num + 1}: {filled_rainfall}/{total_rainfall} rows have rainfall data")
        
        # Optimize memory
        bill_chunk = optimize_dtypes(bill_chunk)
        merged_chunks.append(bill_chunk)
        
        # Force garbage collection
        gc.collect()
    
    return merged_chunks

def main():
    """Main execution function"""
    print("🚀 Starting memory-efficient dataset merge...")
    print("🎯 Goal: Fill missing Rainfall (mm) data from datasets 13 & 14")
    
    # Check if files exist
    if not PATH_BILL.exists():
        print(f"❌ Main dataset not found: {PATH_BILL}")
        return
    
    # Process climatology data (dataset13) -> Rainfall lookup by state/district
    climatology_rainfall = pd.DataFrame()
    if PATH_R13.exists():
        climatology_rainfall = process_climatology_data(PATH_R13)
        print(f"📍 Climatology rainfall data: {len(climatology_rainfall)} state/district combinations")
    else:
        print(f"⚠️ Dataset13 not found: {PATH_R13}")
    
    # Process timeseries data (dataset14) -> Annual rainfall by state/district/year
    timeseries_rainfall = pd.DataFrame()
    if PATH_R14.exists():
        timeseries_rainfall = process_timeseries_data(PATH_R14)
        print(f"📍 Timeseries rainfall data: {len(timeseries_rainfall)} state/district/year combinations")
    else:
        print(f"⚠️ Dataset14 not found: {PATH_R14}")
    
    if climatology_rainfall.empty and timeseries_rainfall.empty:
        print("❌ No rainfall data found in either dataset13 or dataset14!")
        return
    
    # Merge datasets in chunks - fill Rainfall (mm) column
    merged_chunks = merge_datasets_chunked(PATH_BILL, climatology_rainfall, timeseries_rainfall)
    
    # Save chunks to final file
    print("💾 Saving final merged dataset...")
    
    first_chunk = True
    total_filled = 0
    total_rows_processed = 0
    
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        for i, chunk in enumerate(merged_chunks):
            chunk.to_csv(f, index=False, header=first_chunk)
            first_chunk = False
            
            # Track rainfall fill statistics
            if 'Rainfall (mm)' in chunk.columns:
                chunk_filled = chunk['Rainfall (mm)'].notna().sum()
                total_filled += chunk_filled
                total_rows_processed += len(chunk)
            
            print(f"Saved chunk {i + 1}/{len(merged_chunks)}")
    
    # Clean up memory
    del merged_chunks
    gc.collect()
    
    # Final statistics
    final_df = pd.read_csv(OUT_PATH, nrows=5)  # Just read first few rows for info
    total_rows = sum(1 for _ in open(OUT_PATH)) - 1
    
    print(f"\n✅ MERGE COMPLETED!")
    print(f"📄 Output file: {OUT_PATH}")
    print(f"📊 Total rows: {total_rows:,}")
    print(f"📈 Total columns: {len(final_df.columns)}")
    
    if 'Rainfall (mm)' in final_df.columns:
        fill_percentage = (total_filled / total_rows_processed) * 100 if total_rows_processed > 0 else 0
        print(f"🌧️ Rainfall data filled: {total_filled:,}/{total_rows_processed:,} rows ({fill_percentage:.1f}%)")
    
    print(f"\n🏛️ Final columns: {list(final_df.columns)}")
    
    print("\n📋 Sample data:")
    print(final_df.head())
    
    print(f"\n🎯 Dataset ready for ML model training!")
    print(f"💡 The existing 'Rainfall (mm)' column has been enhanced with data from datasets 13 & 14")

if __name__ == "__main__":
    main()