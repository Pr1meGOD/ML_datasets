#!/usr/bin/env python3
"""
clean_data.py

Chunked EDA + cleaning for billashot_final_merged.csv -> final_training_dataset.csv
Place this script in the same folder as billashot_final_merged.csv and run:
    python clean_data.py

What it does (summary):
- Two passes (chunked):
  1) Pass 1: compute missingness fractions, sample text examples, build numeric samples to estimate medians/quantiles.
  2) Pass 2: convert numeric columns robustly, create _raw backups, normalize categorical text, apply heuristic unit conversion for Yield,
     flag impossible numeric rows, drop duplicates inside each chunk, and append to final CSV.
- Saves small JSON reports and two PNG plots in "eda_plots".
"""
import os
import re
import json
import math
import random
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------ User filenames (same folder) ------------------------------
INPUT_CSV = "billashot_final_merged.csv"
OUTPUT_CSV = "final_training_dataset.csv"
PLOTS_DIR = "eda_plots"
CHUNKSIZE = 200000  # adjust if you have more/less memory
NUMERIC_SAMPLE_CAP = 200000  # per numeric candidate, used to estimate medians/quantiles
TEXT_EXAMPLE_CAP = 20         # per text column examples to show

# ------------------------------ Helpers ------------------------------
def robust_numeric(series):
    """Robust numeric conversion for messy strings. Returns pd.Series of floats (NaN on failure)."""
    s = series.astype(str).str.strip().replace({'nan': None, 'None': None, '': None})
    s = s.fillna("")
    s = s.str.replace(r'[,\(\)]', '', regex=True)
    s = s.str.replace(r'\b(kg|kgs|kilogram|kilograms|t|ton|tonne|tonnes|tons)\b', '', regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r'/\s*ha\b', '', regex=True, flags=re.IGNORECASE)
    # keep digits, dot, minus, exponent
    s = s.str.replace(r'[^\d\.\-eE]', '', regex=True)
    return pd.to_numeric(s.replace('', np.nan), errors='coerce')

def reservoir_append(lst, new_vals, cap):
    """
    Append values from new_vals to lst until cap reached.
    new_vals assumed iterable of numeric values (not NaN).
    """
    needed = cap - len(lst)
    if needed <= 0:
        return
    # if new_vals long, take random sample of new_vals size needed
    if len(new_vals) > needed:
        # random sample without replacement
        new_vals = random.sample(list(new_vals), needed)
    lst.extend(new_vals[:needed])

def safe_norm_text(x):
    if x is None:
        return None
    if not isinstance(x, str):
        return x
    v = x.strip()
    v = re.sub(r'\s+', ' ', v)
    return v.lower()

# ------------------------------ PASS 1: scan for stats & samples ------------------------------
os.makedirs(PLOTS_DIR, exist_ok=True)
total_rows = 0
nonnull_counts = defaultdict(int)
text_examples = defaultdict(set)
numeric_candidates = []  # determined from column names
numeric_samples = defaultdict(list)  # reservoir of numeric values for quantiles/median
col_names = None

print("PASS 1: scanning file in chunks to gather missingness + numeric samples...")

for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE, low_memory=False):
    if col_names is None:
        col_names = list(chunk.columns)
        # decide numeric candidates by name
        numeric_candidates = [c for c in col_names if any(k in c.lower() for k in ['yield','production','area','rain','temp','temperature','year'])]
        print("Detected columns:", col_names)
        print("Numeric candidates by name:", numeric_candidates)

    total_rows += len(chunk)
    for c in col_names:
        nonnull_counts[c] += int(chunk[c].notna().sum())

    # collect text examples (sample-based)
    for c in chunk.select_dtypes(include=['object']).columns.tolist():
        vals = chunk[c].dropna().astype(str).head(500).tolist()  # sample first 500 of this chunk
        for v in vals:
            if len(text_examples[c]) < TEXT_EXAMPLE_CAP:
                text_examples[c].add(v.strip().lower())

    # build numeric samples by converting candidate columns for this chunk
    for c in numeric_candidates:
        if c in chunk.columns:
            converted = robust_numeric(chunk[c])
            nonnull_vals = converted.dropna().tolist()
            if nonnull_vals:
                # append up to cap
                need = NUMERIC_SAMPLE_CAP - len(numeric_samples[c])
                if need > 0:
                    if len(nonnull_vals) <= need:
                        numeric_samples[c].extend(nonnull_vals)
                    else:
                        # sample a subset to keep memory reasonable
                        numeric_samples[c].extend(random.sample(nonnull_vals, need))

print(f"PASS 1 complete. Rows scanned: {total_rows}")

# compute missing fraction per column
missing_frac = {c: 1.0 - (nonnull_counts[c] / total_rows) for c in col_names}
# compute sample medians / quantiles for numeric candidates
numeric_stats = {}
for c in numeric_candidates:
    sample = numeric_samples[c]
    if len(sample) == 0:
        numeric_stats[c] = {"sample_count": 0}
    else:
        arr = np.array(sample, dtype=float)
        q1 = float(np.nanpercentile(arr, 25))
        q3 = float(np.nanpercentile(arr, 75))
        med = float(np.nanmedian(arr))
        iqr = q3 - q1
        numeric_stats[c] = {"sample_count": len(arr), "median": med, "q1": q1, "q3": q3, "iqr": iqr,
                             "lower": q1 - 1.5 * iqr, "upper": q3 + 1.5 * iqr}

# Decide heuristic for Yield unit conversion (kg/ha -> t/ha)
yield_convert_to_t = False
if 'Yield' in numeric_stats and numeric_stats['Yield'].get('sample_count', 0) > 0:
    med_yield = numeric_stats['Yield']['median']
    print("Estimated median Yield from sample:", med_yield)
    if med_yield > 1000:
        yield_convert_to_t = True
        print("Heuristic: will convert Yield by dividing by 1000 (kg/ha -> t/ha).")

# Save simple JSON reports from pass 1
with open(os.path.join(PLOTS_DIR, 'missing_fraction.json'), 'w') as f:
    json.dump(missing_frac, f, indent=2)
with open(os.path.join(PLOTS_DIR, 'text_examples.json'), 'w') as f:
    json.dump({k: list(v) for k, v in text_examples.items()}, f, indent=2)
with open(os.path.join(PLOTS_DIR, 'numeric_sample_stats.json'), 'w') as f:
    json.dump(numeric_stats, f, indent=2)

# Save missingness bar plot
plt.figure(figsize=(10,4))
items = sorted(missing_frac.items(), key=lambda x: x[1])
cols_plot = [i[0] for i in items]
vals_plot = [i[1] for i in items]
plt.barh(range(len(vals_plot)), vals_plot)
plt.yticks(range(len(cols_plot)), cols_plot)
plt.xlabel("Missing fraction")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "missing_fraction_bar.png"), dpi=150)
plt.close()

print("PASS 1 reports saved to", PLOTS_DIR)

# ------------------------------ PASS 2: chunked cleaning & write output ------------------------------
print("PASS 2: cleaning chunks and writing cleaned CSV:", OUTPUT_CSV)
first_chunk = True
rows_written = 0
rows_flagged_impossible = 0
chunk_idx = 0

for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE, low_memory=False):
    chunk_idx += 1
    # keep original raw strings for numeric candidates
    for c in numeric_candidates:
        if c in chunk.columns:
            chunk[c + "_raw"] = chunk[c]  # backup original string column

    # convert numeric columns robustly
    for c in numeric_candidates:
        if c in chunk.columns:
            chunk[c] = robust_numeric(chunk[c])

    # apply unit conversion for Yield if chosen
    if yield_convert_to_t and 'Yield' in chunk.columns:
        # divide only non-null numeric values
        chunk.loc[chunk['Yield'].notna(), 'Yield'] = chunk.loc[chunk['Yield'].notna(), 'Yield'] / 1000.0

    # normalize text/object columns (strip, collapse spaces, lowercase)
    obj_cols = chunk.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        chunk[c] = chunk[c].where(chunk[c].notna(), None)
        chunk[c] = chunk[c].map(lambda x: safe_norm_text(x) if x is not None else None)

    # flag impossible numeric rows (Yield <= 0 or Area <= 0)
    chunk['impossible_numeric'] = False
    if 'Yield' in chunk.columns:
        chunk.loc[chunk['Yield'].notna() & (chunk['Yield'] <= 0), 'impossible_numeric'] = True
    if 'Area' in chunk.columns:
        chunk.loc[chunk['Area'].notna() & (chunk['Area'] <= 0), 'impossible_numeric'] = True
    rows_flagged_impossible += int(chunk['impossible_numeric'].sum())

    # within-chunk deduplication (drops exact duplicate rows inside this chunk)
    before = len(chunk)
    chunk = chunk.drop_duplicates()
    after = len(chunk)
    dropped_in_chunk = before - after

    # append to output CSV
    if first_chunk:
        chunk.to_csv(OUTPUT_CSV, index=False, mode='w')
        first_chunk = False
    else:
        chunk.to_csv(OUTPUT_CSV, index=False, mode='a', header=False)

    rows_written += len(chunk)
    print(f"Chunk {chunk_idx}: read {before}, deduped {dropped_in_chunk}, wrote {len(chunk)}")

print("PASS 2 complete.")
print(f"Total rows written to {OUTPUT_CSV}: {rows_written}")
print(f"Rows flagged impossible (Yield<=0 or Area<=0): {rows_flagged_impossible}")

# ------------------------------ Outlier report using sample stats ------------------------------
# Use numeric_stats computed in pass1 to estimate IQR outlier counts by second pass (we can re-scan quickly)
outlier_counts = {}
for c, st in numeric_stats.items():
    if st.get("sample_count", 0) == 0:
        continue
    lower = st["lower"]
    upper = st["upper"]
    # count using chunked read
    below = 0
    above = 0
    for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE, low_memory=False):
        if c in chunk.columns:
            vals = robust_numeric(chunk[c]).dropna()
            below += int((vals < lower).sum())
            above += int((vals > upper).sum())
    outlier_counts[c] = {"below": below, "above": above, "lower": lower, "upper": upper}

with open(os.path.join(PLOTS_DIR, 'outlier_counts.json'), 'w') as f:
    json.dump(outlier_counts, f, indent=2)

print("Saved outlier_counts.json in", PLOTS_DIR)

# ------------------------------ Final summary ------------------------------
summary = {
    "input_file": INPUT_CSV,
    "output_file": OUTPUT_CSV,
    "total_rows_scanned_est": total_rows,
    "rows_written": rows_written,
    "rows_flagged_impossible": rows_flagged_impossible,
    "yield_converted_kg_to_t": yield_convert_to_t,
    "numeric_candidates": numeric_candidates,
    "missing_fraction_file": os.path.join(PLOTS_DIR, 'missing_fraction.json'),
    "text_examples_file": os.path.join(PLOTS_DIR, 'text_examples.json'),
    "numeric_sample_stats_file": os.path.join(PLOTS_DIR, 'numeric_sample_stats.json'),
    "outlier_counts_file": os.path.join(PLOTS_DIR, 'outlier_counts.json')
}
with open(os.path.join(PLOTS_DIR, 'cleaning_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("Cleaning summary saved to", os.path.join(PLOTS_DIR, 'cleaning_summary.json'))
print("All done. Review the cleaned CSV and the reports in", PLOTS_DIR)