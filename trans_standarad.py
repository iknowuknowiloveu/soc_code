import shap
import numpy as np
from joblib import load, dump
import logging
import os

# ================== 日志配置 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()


# ================== 反标准化 SHAP 值 ==================
def reverse_standardize_shap(shap_values, scaler):
    """
    反标准化 SHAP 值，使其返回原始数据空间的 SHAP 值
    """
    # 获取标准化时的均值和标准差
    means = scaler.mean_
    stds = scaler.scale_

    # 创建一个新的 SHAP 值数组，反标准化每一列（每个特征）
    shap_values_original = shap_values.copy()
    for i in range(shap_values.shape[1]):  # 遍历所有特征
        shap_values_original[:, i] = shap_values[:, i] * stds[i] + means[i]

    return shap_values_original


# ================== 反标准化 SHAP 值并保存 ==================
def reverse_standardize_all_shap(model_dir, num_models, scaler_path):
    """
    反标准化所有模型的 SHAP 值，并计算平均 SHAP 值
    """
    # 加载 scaler
    scaler = load(r'G:\Project-yqr\new\model\weight\scaler_all_data.joblib')
    logger.info("已加载 scaler（用于反标准化）")

    all_shap_values_original = []

    # 逐个处理每个模型的 SHAP 值
    for i in range(num_models):
        shap_file_path = os.path.join(model_dir, f'shap_values_fold_{i + 1}.joblib')
        output_path = os.path.join(model_dir, f'shap_values_fold_{i + 1}_original.joblib')

        # 加载标准化后的 SHAP 值
        shap_values_standardized = load(shap_file_path)
        logger.info(f"已加载模型 {i + 1} 的标准化 SHAP 值，shape={shap_values_standardized.shape}")

        # 反标准化 SHAP 值
        shap_values_original = reverse_standardize_shap(shap_values_standardized, scaler)

        # 将反标准化后的 SHAP 值添加到列表中
        all_shap_values_original.append(shap_values_original)

        # 保存反标准化后的 SHAP 值
        dump(shap_values_original, output_path)
        logger.info(f"模型 {i + 1} 的反标准化 SHAP 值已保存至 {output_path}")

    # 计算所有模型的 SHAP 值的平均
    shap_values_all_original = np.array(all_shap_values_original)
    shap_values_mean_original = np.mean(shap_values_all_original, axis=0)

    # 保存平均 SHAP 值
    mean_path = os.path.join(model_dir, 'shap_values_mean_original.joblib')
    dump(shap_values_mean_original, mean_path)
    logger.info(f"===== 所有模型的平均 SHAP 值已保存至 {mean_path}, shape={shap_values_mean_original.shape} =====")


# ================== 主函数 ==================
if __name__ == "__main__":
    # 配置文件路径
    model_dir = r'G:\Project-yqr\new\model\weight'  # 模型目录
    num_models = 9  # 模型数量
    scaler_file_path = os.path.join(model_dir, 'preprocessed_data_all_data.joblib')  # 保存的 scaler 文件路径

    # 反标准化所有模型的 SHAP 值并计算平均
    reverse_standardize_all_shap(model_dir, num_models, scaler_file_path)
