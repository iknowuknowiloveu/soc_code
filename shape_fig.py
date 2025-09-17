# plot_shap_values_first_model.py
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import os
# --- 1. 加载第一个模型的 SHAP 值和数据 ---
model_dir = r'G:\Project-yqr\new\model\weight'

# 加载第一个模型的 SHAP 值
shap_values_path = f"{model_dir}/shap_values_fold_1.joblib"
shap_values_1 = load(shap_values_path)

# 加载数据
plotting_data_path = f'{model_dir}/rf_plotting_data_all_data.joblib'
# 只加载数据（不要预先把所有模型 load 到主进程，避免占用内存 / pickle 问题）
preproc_data = load(os.path.join(model_dir, 'preprocessed_data_all_data.joblib'))
X_all_scaled = np.vstack([preproc_data['X_train_scaled'], preproc_data['X_test_scaled']])
covariate_names = preproc_data['selected_features']

# --- 2. 计算每个变量的平均贡献度（绝对值） ---
mean_abs_shap = np.abs(shap_values_1).mean(axis=0)
feature_importance = pd.DataFrame({
    "feature": covariate_names,
    "importance": mean_abs_shap
}).sort_values("importance", ascending=True)

# --- 3. 绘制整体图 ---
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 6, width_ratios=[2, 2, 2, 2, 2, 2])

# 左边一列：条形图（占 3 行的第 1 列）
ax_bar = fig.add_subplot(gs[:, 0])
ax_bar.barh(feature_importance["feature"], feature_importance["importance"], color="skyblue")
ax_bar.set_xlabel("Mean |SHAP value|")
ax_bar.set_title("Feature Importance (Model 1)", fontsize=12)

# 右边 15 个小图：3 行 × 5 列 KDE 散点密度
for idx, feat in enumerate(covariate_names[:15]):  # 只画前 15 个
    row, col = divmod(idx, 5)
    ax = fig.add_subplot(gs[row, col+1])  # +1 因为第 0 列是 bar chart
    sns.kdeplot(
        x=X_all_scaled[:, idx],
        y=shap_values_1[:, idx],
        fill=True,
        cmap="coolwarm",
        ax=ax,
        thresh=0.05
    )
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel("Feature value")
    ax.set_ylabel("SHAP value")

plt.tight_layout()
plt.show()
