import xarray as xr
import pandas as pd
import numpy as np
import os

# -------------------------
# 配置
# -------------------------
output_dir = r'G:\Project-yqr\N_extracted'
os.makedirs(output_dir, exist_ok=True)

years = list(range(2009, 2019))   # 2009–2018

# -------------------------
# 读取点位信息（用 2009 的点集作为基准）
# -------------------------
points_file = r'G:\Project-yqr\组会\DATA\SOC\LUCAS\BD_filled_select.csv'
points_df = pd.read_csv(points_file)
if "ID" not in points_df.columns:
    points_df = points_df.reset_index().rename(columns={"index": "ID"})
points = points_df[["ID", "Latitude", "Longitude"]].copy()

# -------------------------
# 加载 NetCDF 数据集（你之前已经提前手动修正了 time）
# -------------------------
ds_files = [
    ds_nfer_crop_no3,
    ds_nfer_crop_nh4,
    ds_ndep_nhx,
    ds_ndep_noy,
    ds_nfer_pas_nh4,
    ds_nfer_pas_no3,
    ds_nmanure_app_crop,
    ds_nmanure_app_pas,
    ds_nmanure_dep_pas,
    ds_nmanure_dep_range
]

# -------------------------
# 提取逻辑
# -------------------------
def clean_value(val):
    """清理数值：异常缺失值"""
    if val is None or np.isnan(val):
        return np.nan
    if val < -1e18:  # 缺失值填充
        return np.nan
    return float(val)

results = {}

for ds in ds_files:
    # 遍历每个 NetCDF 文件里的变量
    variables = [v for v in ds.data_vars if v not in ['lat','lon','time']]
    for var in variables:
        print(f"🔹 Processing variable: {var}")
        var_df = points[["ID"]].copy()

        for year in years:
            # 选取某一年
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

        # 保存该变量的 CSV
        out_file = os.path.join(output_dir, f"{var}_2009_2018.csv")
        var_df.to_csv(out_file, index=False)
        print(f"✅ Saved: {out_file}")
        results[var] = var_df

print("🎉 所有氮肥变量提取完成！")
