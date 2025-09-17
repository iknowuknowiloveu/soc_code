import xarray as xr
import pandas as pd
import numpy as np
import os

# -------------------------
# 配置
# -------------------------
nc_folder = r'H:/ERA5-ECI/Annual/'   # NetCDF 文件夹
points_file = r'G:\Project-yqr\组会\DATA\SOC\LUCAS\BD_filled_select.csv' # 点的经纬度和ID
output_dir = r'G:\Project-yqr/ECI_extracted'
os.makedirs(output_dir, exist_ok=True)

years = list(range(2009, 2019))   # 2009–2018

# -------------------------
# 读取点位信息
# -------------------------
points_df = pd.read_csv(points_file)
if "ID" not in points_df.columns:
    points_df = points_df.reset_index().rename(columns={"index": "ID"})
points = points_df[["ID", "Latitude", "Longitude"]].copy()

# -------------------------
# 获取所有 NetCDF 文件
# -------------------------
nc_files = [os.path.join(nc_folder, f) for f in os.listdir(nc_folder) if f.endswith(".nc")]

# -------------------------
# 清理数值函数（修复 + 格式化版）
# -------------------------
def clean_value(val):
    """清理数值：异常缺失值、超大值，并规范输出格式"""
    if val is None:
        return np.nan
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return np.nan
        val = val.item()  # 提取单个元素，避免 DeprecationWarning
    if isinstance(val, float) and np.isnan(val):
        return np.nan
    if val < -1e18:  # 处理 -922337... 之类的缺失值
        return np.nan
    if val > 10000:  # 修正单位
        val = val / 86400000000000
    return round(float(val), 6)  # 统一保留 6 位小数

# -------------------------
# 逐变量存储结果
# -------------------------
results = {}

for nc_file in nc_files:
    ds = xr.open_dataset(nc_file)

    # 找到变量名（排除坐标）
    variables = [v for v in ds.data_vars if v not in ['lat','lon','time','time_bnds']]

    for var in variables:
        print(f"🔹 Processing variable: {var}")
        var_df = points[["ID"]].copy()

        # 循环所有年份
        for year in years:
            ds_year = ds.sel(time=str(year))
            ds_year = ds_year.sortby("lat").sortby("lon")

            # 插值缺失值
            sample = ds_year[var].astype("float32")
            sample = sample.interpolate_na(dim="lat", method="nearest")
            sample = sample.interpolate_na(dim="lon", method="nearest")

            # 提取点值
            values = []
            for _, row in points.iterrows():
                lat, lon = row["Latitude"], row["Longitude"]
                val = sample.sel(lat=lat, lon=lon, method="nearest").values
                val = clean_value(val)
                values.append(val)
            var_df[year] = values

        # 保存该变量的 CSV（保证输出浮点数格式整齐）
        out_file = os.path.join(output_dir, f"{var}_2009_2018.csv")
        var_df.to_csv(out_file, index=False, float_format="%.6f")
        print(f"✅ Saved: {out_file}")
        results[var] = var_df

print("🎉 所有变量提取完成！")
