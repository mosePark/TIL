import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def manual_ucl_calculation(filename, link):
    file_path = os.path.join(link, filename)
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    df['ymd'] = pd.to_datetime(df['ym'] + df['date'].astype(str) + '일',
                               format='%Y년%m월%d일',
                               errors='coerce')
    cf = pd.to_datetime('2025-02-28')
    df = df[df['ymd'] <= cf].dropna(subset=['ymd']).reset_index(drop=True)
    df = df.dropna(subset=['TOC(ppm)']).reset_index(drop=True)
    df['TOC(ppm)'] = pd.to_numeric(df['TOC(ppm)'], errors='coerce')
    
    plt.figure(figsize=(8,6))
    plt.boxplot(df['TOC(ppm)'].dropna(), vert=True)
    plt.title(f'Boxplot of TOC(ppm) - {filename}')
    plt.ylabel('TOC(ppm)')
    plt.show()

    user_input = input(f"파일 [{filename}]의 TOC(ppm)에서 이상치 제거를 위한 최대 허용값 입력: ")
    
    if user_input.strip():
        try:
            max_allowed = float(user_input)
            df_filtered = df[df['TOC(ppm)'] <= max_allowed].copy()
            print(f"입력하신 값 {max_allowed}보다 큰 값은 제거. 남은 데이터 건수: {len(df_filtered)}")
            
            plt.figure(figsize=(8,6))
            plt.boxplot(df['TOC(ppm)'].dropna(), vert=True)
            plt.axhline(y=max_allowed, color='red', linestyle='--', label=f'경계선 ({max_allowed:.0f})')
            plt.title(f'Boxplot of TOC(ppm) with Threshold - {filename}')
            plt.ylabel('TOC(ppm)')
            plt.legend()
            plt.show()
        except ValueError:
            print("잘못된 입력입니다. 이상치 제거 없이 진행합니다.")
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    mean_toc = df_filtered['TOC(ppm)'].mean().round(2)
    std_toc = df_filtered['TOC(ppm)'].std().round(2)
    ucl_value = (mean_toc + 1.5 * std_toc).round(2)
    
    print(f"[{filename}] 계산 결과: 평균 = {mean_toc}, 표준편차 = {std_toc}, UCL = {ucl_value}")
    return ucl_value, df_filtered

if __name__ == "__main__":
    base_dir = ".../data/"
    flist = [.. , .. , ..]
    
    dtype = '일별'
    
    ucl_dict = {}
    for file in file_list:
        print("="*60)
        print(f"Processing file: {file}")
        ucl_val, filtered_df = manual_ucl_calculation(file, link)
        ucl_results[file] = ucl_val
    
    print("\n전체 파일 UCL 결과:")
    for file, ucl_val in ucl_results.items():
        print(f"{file}: UCL = {ucl_val}")
