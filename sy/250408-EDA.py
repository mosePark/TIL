'''
사용자함수 대규모 리빌딩

- 250404 오전 10시 : 일별, 주별, 월별, 직별 날짜타입 인자 패치 완료
- UCL 이상치 boxplot 보고 판단해 이상치 제거 후 UCL 계산 넣기
- 알람 새로운 유형 추가 - 연속 2회 초과시 1회 카운트
  - 편차 1.5, 2에 대하여 모두 작업

* 위에 내용 완료

< 추가 >
- 박스플랏에 이상치 제거 라인 그리기
- 직별로 groupby 하지말고 그냥 하나의 개별체로 보기

* 위에 내용 완료

'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def process_(filename, link, standard_value, ucl_threshold, ucl2_threshold, datetype):

    # CSV 파일 불러오기 및 날짜 처리
    df = pd.read_csv(os.path.join(link, filename), encoding='utf-8-sig')
    df['ymd'] = pd.to_datetime(df['ym'] + df['date'].astype(str) + '일',
                               format='%Y년%m월%d일',
                               errors='coerce')
    cf = pd.to_datetime('2025-02-28')
    df = df[df['ymd'] <= cf].dropna(subset=['ymd']).reset_index(drop=True)
    df = df.dropna(subset=['TOC(ppm)']).reset_index(drop=True)
    df['TOC(ppm)'] = pd.to_numeric(df['TOC(ppm)'], errors='coerce')
    
    # 기본 통계 계산
    mean_toc = df['TOC(ppm)'].mean().round(2)
    std_toc = df['TOC(ppm)'].std().round(2)
    
    # 인덱스 및 기간 설정
    df = df.set_index('ymd', drop=False)
    if datetype in ['일별', '주별', '월별']:
        freq_map = {'일별': 'D', '주별': 'W', '월별': 'M'}
        df['period'] = df.index.to_period(freq_map[datetype])
        group_key = 'period'
    elif datetype == '직별':
        if '직' not in df.columns:
            raise ValueError("직별 집계를 위해서는 데이터에 '직' 컬럼이 필요합니다.")
        # x축은 날짜(일별)로 사용하며, 직별은 개별 시계열로 처리
        df['period'] = df.index.to_period('D')
        group_key = ['period', '직']
    else:
        raise ValueError("datetype 인자는 '일별', '주별', '월별', '직별' 중 하나여야 합니다.")
    
    # 데이터 집계 및 스무딩
    if isinstance(group_key, list):  # 직별인 경우
        aggregated = df.groupby(group_key)['TOC(ppm)'].mean().reset_index()
        # 피벗: index를 period로, 컬럼은 직별 TOC 시계열 생성
        pivoted = aggregated.pivot(index='period', columns='직', values='TOC(ppm)')
        pivoted_smooth = pivoted.rolling(window=10, min_periods=3).mean()\
                                  .interpolate(method='time', limit_direction='both')
        x_values = pivoted.index.to_timestamp()
    else:  # 일별, 주별, 월별
        aggregated = df.groupby(group_key)['TOC(ppm)'].mean()
        aggregated_smooth = aggregated.rolling(window=10, min_periods=3).mean()\
                                       .interpolate(method='time', limit_direction='both')
        x_values = aggregated.index.to_timestamp()
    
    # 시각화
    plt.figure(figsize=(16,6))
    if isinstance(group_key, list):  # 직별일 경우 각 직의 시계열을 별도로 그림
        for col in pivoted.columns:
            plt.plot(x_values, pivoted[col], linestyle='-', marker='x', markersize=3, label=f"{col} TOC")
            plt.plot(x_values, pivoted_smooth[col], linestyle='-', marker=',', label=f"{col} 스무딩 TOC")
    else:
        plt.plot(x_values, aggregated, linestyle='-', color='lightblue', marker='x', markersize=3, label='TOC')
        plt.plot(x_values, aggregated_smooth, linestyle='-', color='orange', marker='o', markersize=3, label='스무딩 TOC')
    
    # 기준 및 임계치 선 표시
    plt.axhline(standard_value, color='black', linestyle='--', label='기준값')
    plt.axhline(mean_toc, color='blue', linestyle='-', label='평균')
    plt.axhline(ucl_threshold, color='green', linestyle='--', label='UCL 1.5')
    plt.axhline(ucl2_threshold, color='red', linestyle='--', label='UCL 2')
    
    # x축 눈금 설정
    start_date = x_values.min()
    end_date = x_values.max()
    if datetype in ['일별', '월별']:
        ticks = pd.date_range(start=start_date, end=end_date, freq='MS')
    elif datetype == '주별':
        ticks = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    elif datetype == '직별':
        ticks = pd.date_range(start=start_date, end=end_date, freq='MS')
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.strftime('%Y년 %m월') for d in ticks])
    ax.grid(True, which='major', axis='x', linestyle='--', color='gray', linewidth=0.5)
    
    plt.title(filename.split('.')[0])
    plt.ylabel('TOC(ppm)')
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.ylim(0, ucl2_threshold + 100) # plt.ylim(bottom=0)
    plt.tight_layout()
    
    # 결과 저장
    output_folder = os.path.join(link, datetype, 'vis')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, filename.split('.')[0] + '.png'))
    plt.close()
    
    # 알람 계산을 위한 준비
    eps = 0
    df['above_std'] = (df['TOC(ppm)'] >= (standard_value - eps)).astype(int)
    df['above_ucl_1.5'] = (df['TOC(ppm)'] >= (ucl_threshold - eps)).astype(int)
    df['above_ucl_2'] = (df['TOC(ppm)'] >= (ucl2_threshold - eps)).astype(int)

    # 일별, 주별, 월별의 경우 전체 데이터에 대해 rolling 집계 수행
    df['rolling_exceed_std'] = df['above_std'].rolling(window=5, min_periods=5).sum()
    df['rolling_exceed_ucl_1.5'] = df['above_ucl_1.5'].rolling(window=5, min_periods=5).sum()
    df['rolling_exceed_ucl_2'] = df['above_ucl_2'].rolling(window=5, min_periods=5).sum()
    df['rolling_exceed2_std'] = df['above_std'].rolling(window=2).apply(lambda x: 1 if (x == 1).all() else 0, raw=True)
    df['rolling_exceed2_ucl_1.5'] = df['above_ucl_1.5'].rolling(window=2).apply(lambda x: 1 if (x == 1).all() else 0, raw=True)
    df['rolling_exceed2_ucl_2'] = df['above_ucl_2'].rolling(window=2).apply(lambda x: 1 if (x == 1).all() else 0, raw=True)
    df['alarm_std'] = df['rolling_exceed_std'] >= 3
    df['alarm_ucl_1.5'] = df['rolling_exceed_ucl_1.5'] >= 3
    df['alarm_ucl_2'] = df['rolling_exceed_ucl_2'] >= 3
    df['alarm2_std'] = df['rolling_exceed2_std'].map({1.0: True, 0.0: False})
    df['alarm2_ucl_1.5'] = df['rolling_exceed2_ucl_1.5'].map({1.0: True, 0.0: False})
    df['alarm2_ucl_2'] = df['rolling_exceed2_ucl_2'].map({1.0: True, 0.0: False})
    
    ratio_df = df.groupby('period').agg(
        total=('TOC(ppm)', 'count'),
        exceed_std=('TOC(ppm)', lambda x: (x > standard_value).sum()),
        exceed_ucl_1_5=('TOC(ppm)', lambda x: (x > ucl_threshold).sum()),
        exceed_ucl_2=('TOC(ppm)', lambda x: (x > ucl2_threshold).sum()),
        alarm_count_std=('alarm_std', 'sum'),
        alarm_count_ucl_1_5=('alarm_ucl_1.5', 'sum'),
        alarm_count_ucl_2=('alarm_ucl_2', 'sum'),
        alarm2_count_std=('alarm2_std', 'sum'),
        alarm2_count_ucl_1_5=('alarm2_ucl_1.5', 'sum'),
        alarm2_count_ucl_2=('alarm2_ucl_2', 'sum'),
        exceed2_std=('alarm2_std', lambda x: (x == True).sum()),
        exceed2_ucl_1_5=('alarm2_ucl_1.5', lambda x: (x == True).sum()),
        exceed2_ucl_2=('alarm2_ucl_2', lambda x: (x == True).sum())
    ).reset_index()
    ratio_df['exceed_ratio_std'] = (ratio_df['exceed_std'] / ratio_df['total'] * 100).round(2)
    ratio_df['exceed_ratio_ucl_1.5'] = (ratio_df['exceed_ucl_1_5'] / ratio_df['total'] * 100).round(2)
    ratio_df['exceed_ratio_ucl_2'] = (ratio_df['exceed_ucl_2'] / ratio_df['total'] * 100).round(2)
    ratio_df['exceed2_ratio_std'] = (ratio_df['exceed2_std'] / ratio_df['total'] * 100).round(2)
    ratio_df['exceed2_ratio_ucl_1.5'] = (ratio_df['exceed2_ucl_1_5'] / ratio_df['total'] * 100).round(2)
    ratio_df['exceed2_ratio_ucl_2'] = (ratio_df['exceed2_ucl_2'] / ratio_df['total'] * 100).round(2)
    
    odr = [
    "period", "total", 
    "exceed_std", "exceed_ratio_std", "alarm_count_std",
    "exceed2_std", "exceed2_ratio_std", "alarm2_count_std",
    
    "exceed_ucl_1_5", "exceed_ratio_ucl_1.5", "alarm_count_ucl_1_5",
    "exceed2_ucl_1_5", "exceed2_ratio_ucl_1.5", "alarm2_count_ucl_1_5",
    
    "exceed_ucl_2", "exceed_ratio_ucl_2", "alarm_count_ucl_2", 
    "exceed2_ucl_2", "exceed2_ratio_ucl_2", "alarm2_count_ucl_2",
    ]

    ratio_df = ratio_df[odr]

    # CSV 결과 저장
    output_csv = os.path.join(output_folder, filename.split('.')[0] + '_exceed.csv')
    ratio_df.to_csv(output_csv, encoding='utf-8-sig', index=False)
    df_reset = df.reset_index(drop=True)
    alarm_filename = os.path.join(output_folder, filename.split('.')[0] + '_alarm.csv')
    df_reset.to_csv(alarm_filename, encoding='utf-8-sig', index=False)
    
    # 기본 통계 요약 테이블 생성
    min_val = df['TOC(ppm)'].min().round(2)
    q1 = df['TOC(ppm)'].quantile(0.25).round(2)
    median_val = df['TOC(ppm)'].median().round(2)
    q3 = df['TOC(ppm)'].quantile(0.75).round(2)
    max_val = df['TOC(ppm)'].max().round(2)
    mean_minus_standard = mean_toc - standard_value
    stats_dict = {
        '최소값': [min_val],
        'Q1': [q1],
        '평균': [mean_toc],
        '중앙값': [median_val],
        'Q3': [q3],
        '최대값': [max_val],
        '기준값': [standard_value],
        '평균-기준값': [mean_minus_standard],
        'UCL 1.5': [ucl_threshold],
        'UCL 2': [ucl2_threshold]
    }
    stats = pd.DataFrame(stats_dict, index=[filename.split('.')[0]])
    return stats




# 파일 및 기준값 설정
link = ".../data/"
file_list = [파일 이름들 ~ ]

standard_values = {               # 기준값 

}

# 새롭게 제공된 UCL 값들
ucl = {                           # 이상치 제거한 ucl 값 : mu + 1.5 * sigma
   
}

ucl2 = {                           # 이상치 제거한 ucl 값 : mu + 2 * sigma
  
}

# type parapeter
datetype = '직별'  # 예: '일별', '주별', '월별', '직별'

all_stats = []
for file in file_list:
    stats = process_(file, link, standard_values[file], ucl[file], ucl2[file], datetype)
    all_stats.append(stats)

combined_stats = pd.concat(all_stats)
combined_stats.to_csv(os.path.join(link, datetype, 'stats.csv'), encoding='utf-8-sig')
