import os
import pandas as pd
import numpy as np

DATA_DIR = ".../data"
INPUT_FILE = "example.xlsx"
FILE_PATH = os.path.join(DATA_DIR, INPUT_FILE)

wb = pd.ExcelFile(FILE_PATH)
SHEETS = wb.sheet_names
PREFIXES = ["PFX1", "PFX2"]
TARGET_SHEETS = [s for s in SHEETS if any(s.startswith(p) for p in PREFIXES)]

FFILL_COLS = ["col0", "col1", "col2", "col3"]
ALLOWED_CATS = ["catA", "catB"]
DROP_COLS = ["col_avg", "col_drop"]
KEYWORD = "measurement"

def process_sheet(file_path, sheet):
    df = pd.read_excel(file_path, sheet_name=sheet, header=0, skiprows=range(0, 1))
    for col in FFILL_COLS:
        if col in df.columns:
            df[col] = df[col].ffill()
    if "col0" in df.columns:
        df = df[df["col0"].isin(ALLOWED_CATS)]
    if "col2" in df.columns:
        df = df[df["col2"].astype(str).str.contains(KEYWORD, case=False, na=False, regex=False)]
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    id_vars = [col for col in FFILL_COLS if col in df.columns]
    df_melt = df.melt(id_vars=id_vars, var_name='date', value_name=KEYWORD)
    df_melt = df_melt.fillna(np.nan)
    df_processed = df_melt[['date', "col1", KEYWORD]].copy()
    df_processed['sheet_id'] = sheet
    df_processed['time'] = 1
    return df_processed.fillna(np.nan)

dfs = []
for s in TARGET_SHEETS:
    try:
        temp = process_sheet(FILE_PATH, s)
        dfs.append(temp)
    except Exception as err:
        print(f"Error processing sheet {s}: {err}")

final_df = pd.concat(dfs, ignore_index=True)
final_df = final_df.rename(columns={"col1": "colB"})
final_df['time'] = final_df.groupby(['sheet_id', 'date', 'colB']).cumcount() + 1
final_df = final_df[['sheet_id', 'date', 'colB', 'time', KEYWORD]]
final_df = final_df.fillna(np.nan)
final_df.columns = ['A', 'B', 'C', 'D', 'E']

print(final_df.head())

OUTPUT_FILE = os.path.join(DATA_DIR, 'processed_output.xlsx')
final_df.to_excel(OUTPUT_FILE, index=False)
