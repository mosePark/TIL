import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_ucl(fnm, base_dir):
    fp = os.path.join(base_dir, fnm)
    dfx = pd.read_csv(fp, encoding='utf-8-sig')
    dfx['dt_col'] = pd.to_datetime(dfx['ym'] + dfx['date'].astype(str) + '일',
                                   format='%Y년%m월%d일',
                                   errors='coerce')
    cutoff = pd.to_datetime('2025-02-28')
    dfx = dfx[dfx['dt_col'] <= cutoff].dropna(subset=['dt_col']).reset_index(drop=True)
    dfx = dfx.dropna(subset=['TOC(ppm)']).reset_index(drop=True)
    dfx['TOC(ppm)'] = pd.to_numeric(dfx['TOC(ppm)'], errors='coerce')
    
    plt.figure(figsize=(8,6))
    plt.boxplot(dfx['TOC(ppm)'].dropna(), vert=True)
    plt.title(f'Boxplot - {fnm}')
    plt.ylabel('TOC(ppm)')
    plt.show()
    
    inp = input(f"Enter max allowed TOC for [{fnm}]: ")
    if inp.strip():
        try:
            lim = float(inp)
            df_filtered = dfx[dfx['TOC(ppm)'] <= lim].copy()
            print(f"Removed values > {lim}. Count: {len(df_filtered)}")
        except ValueError:
            print("Invalid input. No filtering applied.")
            df_filtered = dfx.copy()
    else:
        df_filtered = dfx.copy()
    
    avg = df_filtered['TOC(ppm)'].mean().round(2)
    std = df_filtered['TOC(ppm)'].std().round(2)
    ucl_val = (avg + 2 * std).round(2)
    
    print(f"[{fnm}] avg: {avg}, std: {std}, UCL: {ucl_val}")
    return ucl_val, df_filtered

if __name__ == "__main__":
    base_dir = ".../data/"
    flist = [.. , .. , ..]
    
    dtype = '일별'
    
    ucl_dict = {}
    for file in flist:
        print("="*60)
        print(f"Processing {file}")
        ucl, _ = calc_ucl(file, base_dir)
        ucl_dict[file] = ucl
    
    print("\nAll UCL results:")
    for key, val in ucl_dict.items():
        print(f"{key}: UCL = {val}")
