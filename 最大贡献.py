import os
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_origin

# === 配置 ===
variables = ['su', 'CLC', 'NPP', 'nfer_crop_no3', 'ndep_nhx', 'fd', 'prcptot']
year = 2018
input_root = r'E:/Project-yqr/new/results/weight/freeze'
output_path = fr'E:/Project-yqr/new/results/weight/freeze/max_contributor_{year}.tif'

# === 1. 读取所有贡献栅格 ===
contrib_arrays = []
profile_ref = None

for var in variables:
    path = os.path.join(input_root, var, 'contribution', f'rel_contrib_{var}_{year}.tif')
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data[data <= -9999] = np.nan  # 处理无效值
        contrib_arrays.append(data)
        if profile_ref is None:
            profile_ref = src.profile

# === 2. 堆叠 & 计算最大贡献变量索引（按绝对值）===
stack = np.stack(contrib_arrays, axis=0)  # shape: [n_var, height, width]
abs_stack = np.abs(stack)

# 找出每个像素最大绝对值变量的索引
max_idx = np.nanargmax(abs_stack, axis=0)  # shape: [height, width]

# 将原始位置是 nan 的像素设为 nodata
nan_mask = np.all(np.isnan(stack), axis=0)
max_idx[nan_mask] = -1  # -1 表示无数据

# === 3. 保存为 GeoTIFF ===
profile_ref.update(dtype=rasterio.int16, count=1, nodata=-1)

with rasterio.open(output_path, 'w', **profile_ref) as dst:
    dst.write(max_idx.astype(np.int16), 1)

print(f"✅ 已保存最大贡献变量索引图: {output_path}")
