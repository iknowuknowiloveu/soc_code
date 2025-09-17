import xarray as xr
import pandas as pd
import numpy as np
import os

# =======================================================
# 配置
# =======================================================
output_dir = r'G:\Project-yqr/N_extracted'
os.makedirs(output_dir, exist_ok=True)

years = list(range(2009, 2019))   # 2009–2018

# 点位文件（基准）
points_file = r'G:\Project-yqr\组会\DATA\SOC\LUCAS\BD_filled_select.csv'
points_df = pd.read_csv(points_file)
if "ID" not in points_df.columns:
    points_df = points_df.reset_index().rename(columns={"index": "ID"})
points = points_df[["ID", "Latitude", "Longitude"]].copy()

# =======================================================
# NetCDF 文件路径 & 起始年份配置
# =======================================================
nc_dir = r'H:\History of anthropogenic Nitrogen inputs\Tian-etal_2022_allfiles'

nc_config = {
    "nfer_crop_no3":   ("nfer_crop_no3.nc",    1925),
    "nfer_crop_nh4":   ("nfer_crop_nh4.nc",    1925),
    "ndep_nhx":        ("ndep_nhx.nc",         1850),
    "ndep_noy":        ("ndep_noy.nc",         1850),
    "nfer_pas_nh4":    ("nfer_pas_nh4.nc",     1961),
    "nfer_pas_no3":    ("nfer_pas_no3.nc",     1961),
    "nmanure_app_crop":("nmanure_app_crop.nc", 1860),
    "nmanure_app_pas": ("nmanure_app_pas.nc",  1860),
    "nmanure_dep_pas": ("nmanure_dep_pas.nc",  1860),
    "nmanure_dep_range":("nmanure_dep_range.nc",1860),
}

# =======================================================
# 工具函数：加载并修正 time
# =======================================================
def load_and_fix_time(path, start_year):
    ds = xr.open_dataset(path, decode_times=False)
    time_vals = ds["time"].values
    time_dates = [pd.Timestamp(f"{int(start_year + year)}-01-01") for year in time_vals]
    ds["time"] = ("time", time_dates)
    return ds

# =======================================================
# 数值清理（修复版）
# =======================================================
def clean_value(val):
    """修复 val 转换，避免 DeprecationWarning"""
    if val is None:
        return np.nan
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return np.nan
        val = val.item()  # 安全提取单个值
    if isinstance(val, float) and np.isnan(val):
        return np.nan
    if val < -1e18:  # 缺失值
        return np.nan
    return float(val)

# =======================================================
# 主循环：加载 → 提取 → 输出
# =======================================================
results = {}

for name, (fname, start_year) in nc_config.items():
    path = os.path.join(nc_dir, fname)
    print(f"📂 Loading {name} from {path}")
    ds = load_and_fix_time(path, start_year)

    variables = [v for v in ds.data_vars if v not in ["lat", "lon", "time"]]
    for var in variables:
        print(f"   🔹 Extracting variable: {var}")
        var_df = points[["ID"]].copy()

        for year in years:
            if year not in ds["time"].dt.year.values:
                print(f"      ⚠️ Year {year} not in dataset {name}, skipping.")
                continue

            ds_year = ds.sel(time=str(year))
            ds_year = ds_year.sortby("lat").sortby("lon")

            sample = ds_year[var].astype("float32")
            sample = sample.interpolate_na(dim="lat", method="nearest")
            sample = sample.interpolate_na(dim="lon", method="nearest")

            values = []
            for _, row in points.iterrows():
                lat, lon = row["Latitude"], row["Longitude"]
                val = sample.sel(lat=lat, lon=lon, method="nearest").values
                val = clean_value(val)
                values.append(val)
            var_df[year] = values

        out_file = os.path.join(output_dir, f"{var}_2009_2018.csv")
        var_df.to_csv(out_file, index=False)
        print(f"   ✅ Saved: {out_file}")
        results[var] = var_df

print("🎉 所有氮肥变量提取完成！")
