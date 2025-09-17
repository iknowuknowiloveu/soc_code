import xarray as xr
import pandas as pd
import numpy as np
import os

# ========================
# 配置
# ========================
spei_nc = r"E:\Project-yqr\ERA5-ECI\spei_MON_3_6_12_scale_era5_land_only_0p09_deg_1981_2019.nc"
spi_nc  = r"E:\Project-yqr\ERA5-ECI\spi_MON_3_6_12_scale_era5_land_only_0p09_deg_1981_2019.nc"

points_file = r"E:\Project-yqr\组会\DATA\SOC\LUCAS\BD_filled_select.csv"  # 点的经纬度和ID
output_dir = r"E:\Project-yqr\ERA5-ECI\csv_output"
os.makedirs(output_dir, exist_ok=True)

years = list(range(2009, 2019))   # 2009–2018

# ========================
# 读取点位
# ========================
points_df = pd.read_csv(points_file)
if "ID" not in points_df.columns:
    points_df = points_df.reset_index().rename(columns={"index": "ID"})
points = points_df[["ID", "Latitude", "Longitude"]].copy()

# ========================
# 通用函数：提取某个变量的年均值 (scale=3)，带插值
# ========================
def extract_annual_mean(nc_path, var_name, points, years, label):
    ds = xr.open_dataset(nc_path)
    da = ds[var_name].sel(scale=3)   # 只取 3-month scale
    
    # 聚合到年均值
    da_year = da.groupby("time.year").mean(dim="time")
    
    results = points[["ID"]].copy()
    
    for year in years:
        if year not in da_year["year"].values:
            print(f"⚠️ {year} 不在 {label} 数据范围内，跳过")
            continue
        
        # 取某一年数据
        sample = da_year.sel(year=year).sortby("lat").sortby("lon")
        
        # ====== 插值缺失值 ======
        sample = sample.where(np.isfinite(sample))  # 屏蔽非有限值
        sample = sample.interpolate_na(dim="lat", method="nearest")
        sample = sample.interpolate_na(dim="lon", method="nearest")
        
        # 提取点
        values = []
        for _, row in points.iterrows():
            val = sample.sel(lat=row["Latitude"], lon=row["Longitude"], method="nearest").values
            if isinstance(val, np.ndarray):
                val = val.item()
            if not np.isfinite(val):
                val = np.nan
            values.append(val)
        results[year] = values
    
    out_file = os.path.join(output_dir, f"{label}_12_annual_mean_2009_2018.csv")
    results.to_csv(out_file, index=False)
    print(f"✅ 保存完成: {out_file}")
    return results

# ========================
# 执行提取
# ========================
spei_df = extract_annual_mean(spei_nc, "spei", points, years, "spei")

print("🎉 SPEI-12 & SPI-12 年均值提取完成！")
