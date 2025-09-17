import shap
import numpy as np
from joblib import load, dump
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures as concurrent   # 修复 NameError
from tqdm import tqdm
import logging
import time
import os

# ================== 日志配置 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

# ================== 配置：如果你确定要从第 4 个模型开始，请保持 3（0-based） = 第4个
# 如果想自动检测已存在的 shap 文件，设为 None
START_FROM_INDEX = 3  # 0-based，3 表示从第4个开始；设为 None 则自动检测已有 shap 文件

# ================== 1. 加载数据（保持不变） ==================
model_dir = r'G:\Project-yqr\new\model\weight'

# 只加载数据（不要预先把所有模型 load 到主进程，避免占用内存 / pickle 问题）
preproc_data = load(os.path.join(model_dir, 'preprocessed_data_all_data.joblib'))
X_all_scaled = np.vstack([preproc_data['X_train_scaled'], preproc_data['X_test_scaled']])
covariate_names = preproc_data['selected_features']

# 仅构造模型路径列表（子进程内部按路径 load）
num_models = 9
model_paths = [os.path.join(model_dir, f'rf_fold_{i+1}.joblib') for i in range(num_models)]
logger.info("模型路径列表准备完成。")

# ================== 2. 块 SHAP 计算函数 ==================
def compute_shap_chunk(explainer, X_chunk, model_idx, chunk_idx):
    start = time.time()
    shap_chunk = explainer.shap_values(X_chunk)  # SHAP 内部可能使用 numba 等加速
    elapsed = time.time() - start
    logger.info(f"[模型 {model_idx+1}] 块 {chunk_idx+1} 完成, 耗时 {elapsed:.2f}s")
    return shap_chunk

# ================== 3. 单模型 SHAP（子进程中运行） ==================
def compute_shap_model_process(model_idx, model_path, X_data, chunk_size=200):
    """
    在子进程内加载模型（避免在主进程中传模型对象导致 pickle 问题），
    然后按块顺序计算 SHAP 并保存结果。
    """
    start_model = time.time()
    logger.info(f"===== [模型 {model_idx+1}] 开始计算（子进程）: load {os.path.basename(model_path)} =====")

    # 在子进程内 load 模型（避免把模型对象 pickle 到子进程）
    model = load(model_path)
    explainer = shap.TreeExplainer(model)

    n_samples = X_data.shape[0]
    num_chunks = (n_samples + chunk_size - 1) // chunk_size

    shap_values_list = []
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
        X_chunk = X_data[start_idx:end_idx]

        logger.info(f"[模型 {model_idx+1}] 开始计算块 {chunk_idx+1}/{num_chunks}...")
        # 直接计算并记录
        shap_chunk = compute_shap_chunk(explainer, X_chunk, model_idx, chunk_idx)
        shap_values_list.append(shap_chunk)

    # 合并并保存
    shap_values_all = np.vstack(shap_values_list)
    out_path = os.path.join(model_dir, f'shap_values_fold_{model_idx+1}.joblib')
    dump(shap_values_all, out_path)

    logger.info(f"===== [模型 {model_idx+1}] 完成，总耗时 {time.time() - start_model:.2f}s, 保存到 {out_path} =====")
    return shap_values_all

# ================== 4. 并行计算所有模型（按批次，每批最多 max_parallel_models） ==================
def compute_all_models_process(model_paths, X_data, chunk_size=200, max_parallel_models=3):
    num_models = len(model_paths)
    shap_all = [None] * num_models

    # 先处理跳过逻辑：
    if START_FROM_INDEX is None:
        # 自动检测已有的 shap 文件，存在则跳过并加载
        logger.info("START_FROM_INDEX=None -> 自动检测已有 shap 文件并跳过。")
        for idx in range(num_models):
            out_path = os.path.join(model_dir, f'shap_values_fold_{idx+1}.joblib')
            if os.path.exists(out_path):
                logger.info(f"检测到已完成模型 {idx+1} 的文件，加载并跳过计算: {out_path}")
                shap_all[idx] = load(out_path)
    else:
        # 强制从 START_FROM_INDEX 开始（0-based）：把前面的当作已完成并加载它们（如果存在）
        logger.info(f"强制从第 {START_FROM_INDEX+1} 个模型开始（0-based={START_FROM_INDEX}）。")
        for idx in range(0, START_FROM_INDEX):
            out_path = os.path.join(model_dir, f'shap_values_fold_{idx+1}.joblib')
            if os.path.exists(out_path):
                logger.info(f"加载已完成模型 {idx+1}: {out_path}")
                shap_all[idx] = load(out_path)
            else:
                logger.warning(f"预期跳过的模型 {idx+1} 的 SHAP 文件不存在: {out_path} （请确认你已经保存了前 {START_FROM_INDEX} 个结果）")

    # 构建需要计算的索引列表
    remaining_indices = [i for i in range(num_models) if shap_all[i] is None]
    if not remaining_indices:
        logger.info("没有需要计算的模型，全部已存在 SHAP 文件。")
        return shap_all

    # 按批次提交：每批最多 max_parallel_models 个模型并行（每个模型在子进程中串行按块计算）
    for batch_start in range(0, len(remaining_indices), max_parallel_models):
        batch_inds = remaining_indices[batch_start: batch_start + max_parallel_models]
        logger.info(f"===== 开始处理模型批次: {', '.join(str(i+1) for i in batch_inds)} =====")

        # 每个子进程负责一个模型（子进程内按块串行）
        with ProcessPoolExecutor(max_workers=len(batch_inds)) as executor:
            futures = {
                executor.submit(compute_shap_model_process, idx, model_paths[idx], X_data, chunk_size): idx
                for idx in batch_inds
            }

            for f in tqdm(concurrent.as_completed(futures),
                          total=len(futures),
                          desc=f"批次模型 {batch_inds[0]+1}-{batch_inds[-1]+1} 进度",
                          position=0):
                idx = futures[f]
                try:
                    shap_all[idx] = f.result()
                except Exception as e:
                    logger.exception(f"模型 {idx+1} 计算时出错：{e}")

        logger.info(f"===== 批次完成: {', '.join(str(i+1) for i in batch_inds)} =====")

    return shap_all

# ================== 5. 主函数 ==================
if __name__ == "__main__":
    logger.info("===== 开始计算所有模型的 SHAP 值 =====")
    shap_values_all = compute_all_models_process(model_paths, X_all_scaled, chunk_size=200, max_parallel_models=3)

    # 确保每个位置都有结果（否则会在这里抛错）
    for i, v in enumerate(shap_values_all):
        if v is None:
            logger.error(f"模型 {i+1} 的 SHAP 结果缺失（None）。请检查之前的输出文件或计算过程。")
            raise RuntimeError(f"模型 {i+1} 的 SHAP 结果缺失。")

    # 平均 SHAP
    shap_values_mean = np.mean(np.array(shap_values_all), axis=0)
    mean_path = os.path.join(model_dir, 'shap_values_mean.joblib')
    dump(shap_values_mean, mean_path)
    logger.info(f"===== 平均 SHAP 值已保存至 {mean_path}, shape={shap_values_mean.shape} =====")
