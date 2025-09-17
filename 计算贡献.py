import os
import numpy as np
import rasterio
import logging
from datetime import datetime
import gc

# 参数设置
baseline_dir = r'E:/Project-yqr/new/results/weight'  # 基准预测目录
variables = ['su', 'CLC', 'NPP', 'nfer_crop_no3', 'ndep_nhx', 'fd', 'prcptot']  # 冻结变量列表
years = [2015, 2018]  # 需要计算贡献的年份

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
    return data, profile

def save_raster(path, data, profile):
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)

# 遍历所有冻结变量
for freeze_var in variables:
    logger.info(f"开始处理冻结变量: {freeze_var}")
    freeze_dir = f'E:/Project-yqr/new/results/weight/freeze/{freeze_var}'  # 冻结预测目录
    output_dir = f'E:/Project-yqr/new/results/weight/freeze/{freeze_var}/contribution'
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        logger.info(f"开始计算 {year} 年冻结变量 {freeze_var} 的贡献...")
        t_start = datetime.now()

        baseline_path = os.path.join(baseline_dir, f'weight_pred_{year}.tif')
        freeze_path = os.path.join(freeze_dir, f'weight_pred_freeze_{freeze_var}_{year}.tif')

        if not os.path.exists(baseline_path):
            logger.error(f"基准预测文件不存在: {baseline_path}")
            continue
        if not os.path.exists(freeze_path):
            logger.error(f"冻结预测文件不存在: {freeze_path}")
            continue

        try:
            baseline_data, profile = read_raster(baseline_path)
            freeze_data, _ = read_raster(freeze_path)
        except Exception as e:
            logger.error(f"读取栅格失败: {e}")
            continue

        # 计算绝对贡献 = 基准 - 冻结
        abs_contrib = baseline_data - freeze_data

        # 计算相对贡献 = 绝对贡献 / |基准|，基准太小时不计算（设为 nan）
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_contrib = abs_contrib / np.abs(baseline_data)
            rel_contrib[np.abs(baseline_data) <= 1e-6] = np.nan

        # 保存结果
        abs_out_path = os.path.join(output_dir, f'abs_contrib_{freeze_var}_{year}.tif')
        rel_out_path = os.path.join(output_dir, f'rel_contrib_{freeze_var}_{year}.tif')

        try:
            save_raster(abs_out_path, abs_contrib, profile)
            logger.info(f"✅ 绝对贡献保存: {abs_out_path}")
        except Exception as e:
            logger.error(f"保存绝对贡献失败: {e}")

        try:
            save_raster(rel_out_path, rel_contrib, profile)
            logger.info(f"✅ 相对贡献保存: {rel_out_path}")
        except Exception as e:
            logger.error(f"保存相对贡献失败: {e}")

        # 释放内存
        del baseline_data, freeze_data, abs_contrib, rel_contrib
        gc.collect()

        t_end = datetime.now()
        logger.info(f"完成 {year} 年冻结变量 {freeze_var} 贡献计算，用时 {(t_end - t_start).total_seconds():.1f} 秒")

logger.info("所有变量和年份贡献计算完成！")