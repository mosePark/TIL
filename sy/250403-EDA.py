#%%

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
from sklearn.pipeline import Pipeline
import sklearn.model_selection as skm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#%%

link = ".../data/"
file_list = [~~~~~]


for i, file in enumerate(file_list):
    file_path = os.path.join(link, file)
    try:
        df = pd.read_csv(file_path)
        globals()[f"df{i+1}"] = df
        print(f"{file} 파일을 성공적으로 불러왔습니다. (Shape: {df.shape})")
    except Exception as e:
        print(f"{file} 파일을 불러오는 도중 에러 발생: {e}")

#%% 직별

grouped_results = {}

ucl = []

for i in range(1, 9):
    df_name = f"df{i}"
    df_temp = globals()[df_name].copy()
    df_temp['TOC(ppm)'] = pd.to_numeric(df_temp['TOC(ppm)'], errors='coerce')
    
    mean_toc = df_temp['TOC(ppm)'].dropna().mean()
    std_toc = df_temp['TOC(ppm)'].dropna().std()
    upper_spec = mean_toc + 3 * std_toc
    ucl.append(upper_spec)
    
    df_temp = df_temp.dropna(subset=['TOC(ppm)'])
    df_temp['ymd'] = pd.to_datetime((df_temp['ym'] + df_temp['date'].astype(str) + '일').str.replace(' ', ''),
                                    format='%Y년%m월%d일',
                                    errors='coerce')
    grouped = df_temp.groupby(['ymd', '직'])['TOC(ppm)'].mean().reset_index()
    # 그룹 결과에 TOC(ppm) 열만 남기고, 고유한 이름으로 변경
    grouped = grouped[['ymd', '직', 'TOC(ppm)']]
    new_colname = f"{file_list[i-1].replace('.csv', '')}"
    grouped = grouped.rename(columns={'TOC(ppm)': new_colname})
    grouped_results[df_name] = grouped

merged_df = None
for key in sorted(grouped_results.keys()):
    if merged_df is None:
        merged_df = grouped_results[key]
    else:
        merged_df = pd.merge(merged_df, grouped_results[key], on=['ymd', '직'], how='outer')

merged_df
#%% 이상치 제거 : UCL 넘어가는 것

df = merged_df.dropna()

df_filtered = df.copy()
for idx, col in enumerate(df_filtered.columns[2:]):
    df_filtered = df_filtered[df_filtered[col] <= ucl[idx]]
df_filtered

# CSL DRAIN 3000 넘는 값 제거 하기

df_filtered = df_filtered[df_filtered['CSL DRAIN'] <= 3000]

#%% 상관분석

x = df_filtered.drop(columns=['ymd', '직'])

def corr_heatmap(x, y, **kwargs):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)

    ax.set_facecolor(cmap((r + 1) / 2))
    ax.annotate(f"{r:.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes,
                ha='center', va='center', fontsize=12,
                color='white', fontweight='bold')

g = sns.PairGrid(x)
g.map_upper(sns.scatterplot)
g.map_diag(sns.histplot, kde=True)
g.map_lower(corr_heatmap)

cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
norm = Normalize(vmin=-1, vmax=1)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

g.figure.subplots_adjust(right=0.85)

cbar_ax = g.figure.add_axes([0.88, 0.15, 0.02, 0.7])
cbar = g.figure.colorbar(sm, cax=cbar_ax, orientation='vertical')

plt.show()
# %% PCA ##################################################
###########################################################
X = df_filtered.drop(columns=['집수조', 'ymd', '직']).dropna()

X.mean() 
X.var() # 변수간 평균과 분산이 다르다.

scaler = StandardScaler(with_std=True,
                        with_mean=True)

X_scaled = scaler.fit_transform(X)

pca = PCA() # PCs 개수 설정 : n_components=2
pca_result = pca.fit(X_scaled)

pca_result.mean_ # 변수들의 평균 centering

scores = pca.transform(X_scaled)

pca_result.components_ # 주성분 적재값, 각 행의 경우 주성분 적재 벡터

#%% biplot 시각화

i, j = 0, 1
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:,0], scores[:,1])
ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))
for k in range(pca_result.components_.shape[1]):
    ax.arrow(0, 0, pca_result.components_[i,k], pca_result.components_[j,k])
    ax.text(pca_result.components_[i,k],
            pca_result.components_[j,k],
            X.columns[k])

#%%

scale_arrow = s_ = 2
scores[:,1] *= -1
pca_result.components_[1] *= -1 # y축을 뒤집는다
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:,0], scores[:,1])
ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))
for k in range(pca_result.components_.shape[1]):
    ax.arrow(0, 0, s_*pca_result.components_[i,k], s_*pca_result.components_[j,k])
    ax.text(s_*pca_result.components_[i,k],
            s_*pca_result.components_[j,k],
            X.columns[k])

#%%

scores.std(0, ddof=1) # 주성분점수의 표준편차

pca_result.explained_variance_ratio_ # 설명하는 분산 비율 (%)

#%%

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ticks = np.arange(pca_result.n_components_)+1
ax = axes[0]
ax.plot(ticks,
        pca_result.explained_variance_ratio_,
        marker='o')
ax.set_xlabel('주성분');
ax.set_ylabel('설명된 분산 비율')
ax.set_ylim([0,1])
ax.set_xticks(ticks)

ax = axes[1]
ax.plot(ticks,
        pca_result.explained_variance_ratio_.cumsum(),
        marker='o')
ax.set_xlabel('주성분')
ax.set_ylabel('설명된 분산 비율')
ax.set_ylim([0, 1])
ax.set_xticks(ticks)
fig



# %% PCR ##################################################
###########################################################
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


X = df_filtered.drop(columns=['집수조', 'ymd', '직']).dropna()
Y = df_filtered['집수조']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

n_components = 5
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components)),
    ('linreg', skl.LinearRegression())
])

pipe.fit(X_train, Y_train)
Yhat_train = pipe.predict(X_train)
Yhat_test = pipe.predict(X_test)

train_mse = mean_squared_error(Y_train, Yhat_train)
test_mse = mean_squared_error(Y_test, Yhat_test)
train_r2 = r2_score(Y_train, Yhat_train)
test_r2 = r2_score(Y_test, Yhat_test)

n_train = len(Y_train)
n_test = len(Y_test)
p = n_components

train_adj_r2 = adjusted_r2(train_r2, n_train, p)
test_adj_r2 = adjusted_r2(test_r2, n_test, p)

mape_test = np.mean(np.abs((Y_test - Yhat_test) / Y_test)) * 100

# print(f"Train MSE: {train_mse:.3f}")
print(f"Test MSE: {test_mse:.3f}")
# print(f"Train RMSE: {(train_mse**0.5):.3f}")
print(f"Test RMSE: {(test_mse**0.5):.3f}")
# print(f"Train R^2: {train_r2:.3f}")
print(f"Test R^2: {test_r2:.3f}")
# print(f"Train Adjusted R^2: {train_adj_r2:.3f}")
print(f"Test Adjusted R^2: {test_adj_r2:.3f}")
print(f"Test MAPE: {mape_test:.3f}%")

#%%


# 실제 값과 예측 값 산점도
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Yhat_test, alpha=0.7, edgecolor='k', label='예측값')

# x=y (완벽한 예측) 선 그리기
min_val = min(Y_test.min(), Yhat_test.min())
max_val = max(Y_test.max(), Yhat_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='x = y')

plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title('실제 vs 예측')
plt.legend()
plt.show()
# %%

for i in range(7):
    plt.figure(figsize=(10, 8))
    x = X_test.iloc[:, i]
    plt.scatter(x, Y_test, marker="o", label="실제값", alpha=0.7, edgecolor='k')
    plt.scatter(x, Yhat_test, marker="x", label="예측값", alpha=0.7, edgecolor='k')
    plt.xlabel(X_test.columns[i])
    plt.ylabel("집수조")
    plt.title(f"{X_test.columns[i]} vs 집수조")
    plt.legend()
    plt.show()


# %%
