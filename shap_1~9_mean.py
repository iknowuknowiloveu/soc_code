import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import os
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import gaussian_kde

# ================= 日志配置 =================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

# --- 1. 加载数据 ---
def load_data():
    model_dir = r'G:\Project-yqr\new\model\weight'
    logger.info("开始加载数据...")

    preproc_data_path = os.path.join(model_dir, 'preprocessed_data_all_data.joblib')
    preproc_data = load(preproc_data_path)
    X_train_selected = preproc_data['X_train_selected']
    X_test_selected = preproc_data['X_test_selected']
    X_all = np.vstack([X_train_selected, X_test_selected])
    covariate_names = preproc_data['selected_features']
    logger.info(f"数据加载完成，样本数={X_all.shape[0]}，特征数={X_all.shape[1]}")

    return X_all, covariate_names

# --- 2. 并行计算核密度数据 ---
def compute_density(idx_X_cov):
    idx, X_all, shap_values = idx_X_cov
    x = X_all[:, idx]
    y = shap_values[:, idx]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx_sort = z.argsort()
    x_sorted, y_sorted, z_sorted = x[idx_sort], y[idx_sort], z[idx_sort]
    return idx, x_sorted, y_sorted, z_sorted

# --- 3. 绘制单个 SHAP 图函数 ---
def plot_shap_figure(X_all, covariate_names, shap_values, label, std_shap=None):
    # 计算平均 SHAP 贡献度
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": covariate_names,
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=True)

    if std_shap is not None:
        feature_importance["std"] = std_shap[feature_importance.index]

    # 并行计算密度数据
    inputs = [(i, X_all, shap_values) for i in range(len(covariate_names[:15]))]
    results = []
    with ProcessPoolExecutor() as executor:
        for r in tqdm(executor.map(compute_density, inputs), total=len(inputs),
                      desc=f"计算核密度 {label}"):
            results.append(r)
    results.sort(key=lambda x: x[0])

    # --- 绘制总图 ---
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 6, width_ratios=[2, 2, 2, 2, 2, 2], height_ratios=[1, 1, 1])

    # 左侧条形图
    ax_bar = fig.add_subplot(gs[:, 0])
    if std_shap is not None:
        ax_bar.barh(feature_importance["feature"], feature_importance["importance"], xerr=feature_importance["std"],
                    color="skyblue", ecolor="gray", capsize=3)
    else:
        ax_bar.barh(feature_importance["feature"], feature_importance["importance"], color="skyblue")
    ax_bar.set_xlabel("Mean |SHAP value|")
    ax_bar.set_title(f"Feature Importance ({label})", fontsize=12)

    # 右侧密度图
    for i, (idx, x_sorted, y_sorted, z_sorted) in enumerate(results):
        row, col = divmod(i, 5)
        ax = fig.add_subplot(gs[row, col + 1])
        sc = ax.scatter(x_sorted, y_sorted, c=z_sorted, cmap="coolwarm", s=5, alpha=0.6)
        ax.set_title(covariate_names[idx], fontsize=10)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("SHAP value")

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.4)

    # 保存和显示
    output_path = rf'G:\Project-yqr\new\model\weight\shap_{label}.jpg'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"{label} 绘图完成，保存至 {output_path}")
    plt.close(fig)

# --- 4. 主函数 ---
if __name__ == '__main__':
    X_all, covariate_names = load_data()

    # --- 九个 fold 模型 ---
    for i in range(1, 10):
        shap_path = rf'G:\Project-yqr\new\model\weight\shap_values_fold_{i}.joblib'
        logger.info(f"加载 SHAP 模型 fold_{i}: {shap_path}")
        shap_values = load(shap_path)
        plot_shap_figure(X_all, covariate_names, shap_values, f"fold_{i}")

    # --- 平均模型 ---
    shap_mean_path = r'G:\Project-yqr\new\model\weight\shap_values_mean.joblib'
    shap_mean = load(shap_mean_path)

    # 计算九个 fold 的标准差
    shap_all = [load(rf'G:\Project-yqr\new\model\weight\shap_values_fold_{i}.joblib') for i in range(1, 10)]
    shap_all_array = np.stack(shap_all, axis=0)  # shape=(9, n_samples, n_features)
    std_shap = np.std(shap_all_array, axis=0).mean(axis=0)  # 每个特征平均标准差

    plot_shap_figure(X_all, covariate_names, shap_mean, "mean_model", std_shap=std_shap)
