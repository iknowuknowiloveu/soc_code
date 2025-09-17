import os
import numpy as np
import rasterio
import logging
import gc

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 路径
input_dir = r'E:\Project-yqr\new\results\weight'
output_dir = input_dir  # 保存到同一目录，可改为其他路径
years = [2009, 2015, 2018]

def save_diff_raster(diff_data, extent, output_path, src_profile):
    """保存差值栅格为 TIFF 文件"""
    try:
        transform = rasterio.transform.from_bounds(*extent, diff_data.shape[1], diff_data.shape[0])
        profile = src_profile.copy()
        profile.update({
            'dtype': 'float32',
            'count': 1,
            'transform': transform,
            'nodata': -9999
        })
        with rasterio.open(output_path, 'w', **profile) as dst:
            diff_data_filled = np.where(np.isnan(diff_data), -9999, diff_data)
            dst.write(diff_data_filled, 1)
        logger.info(f"✅ 保存差值栅格: {output_path}")
    except Exception as e:
        logger.error(f"❌ 保存栅格失败: {output_path}, 错误: {str(e)}")
        raise

# 读取数据
data_dict = {}
for year in years:
    path = os.path.join(input_dir, f'weight_pred_{year}.tif')
    if not os.path.exists(path):
        logger.warning(f"❌ 文件不存在: {path}")
        continue
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            data[data <= -9999] = np.nan  # 处理无效值
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            profile = src.profile  # 提取 profile
            data_dict[year] = (data, extent, profile)
            logger.info(f"✅ 加载 {year}, shape={data.shape}, max={np.nanmax(data):.2f}, min={np.nanmin(data):.2f}")
    except Exception as e:
        logger.error(f"❌ 加载 {year} 失败: {str(e)}")
        continue

# 计算并保存差值栅格
diff_pairs = [(2015, 2009, "2015-2009"), (2018, 2015, "2018-2015"), (2018, 2009, "2018-2009")]
for year2, year1, label in diff_pairs:
    if year1 not in data_dict or year2 not in data_dict:
        logger.warning(f"❌ 无法计算 {label} 差值：缺少 {year1} 或 {year2} 数据")
        continue
    data1, extent1, profile1 = data_dict[year1]
    data2, extent2, profile2 = data_dict[year2]
    if data1.shape != data2.shape or extent1 != extent2:
        logger.error(f"❌ {label} 差值失败：栅格 shape 或 extent 不匹配")
        continue
    diff_data = data2 - data1
    output_path = os.path.join(output_dir, f'weight_pred_diff_{label}.tif')
    try:
        save_diff_raster(diff_data, extent1, output_path, profile1)
        logger.info(f"✅ 计算 {label} 差值, shape={diff_data.shape}, max={np.nanmax(diff_data):.2f}, min={np.nanmin(diff_data):.2f}")
    except Exception as e:
        logger.error(f"❌ 处理 {label} 差值失败: {str(e)}")
        continue

gc.collect()
logger.info("✅ 所有差值栅格保存成功")