import xarray as xr
import geopandas as gpd
import os

# ==========================
# 配置
# ==========================
data_dir = r"F:\GLEAM4.1土壤湿度数据\SMrz根际土壤湿度"
shp_path = r"E:\Project-yqr\组会\DATA\shp\combine\EU_1984.shp"
output_dir = r"E:\Project-yqr\EU_TIFF_500m"

years_to_extract = [2009, 2015, 2018]

# ==========================
# 读取shp
# ==========================
mask_shp = gpd.read_file(shp_path)

# ==========================
# 处理每个nc文件
# ==========================
for nc_file in os.listdir(data_dir):
    if nc_file.endswith(".nc"):
        nc_path = os.path.join(data_dir, nc_file)
        print(f"正在处理文件: {nc_path}")
        
        # 读取 NetCDF
        ds = xr.open_dataset(nc_path)
        
        # 假设变量名为 'SMrz'（GLEAM 通常就是这样）
        var_name = "SMrz"
        if var_name not in ds:
            raise ValueError(f"变量 {var_name} 不在文件 {nc_file} 中，可用变量有: {list(ds.data_vars)}")
        
        da = ds[var_name]
        
        # 绑定地理坐标 (GLEAM 通常是 lat/lon)
        da = da.rio.write_crs("EPSG:4326")

        # ==========================
        # 按年份提取
        # ==========================
        for year in years_to_extract:
            if str(year) not in nc_file:  # 文件名里有年份可直接筛选
                continue

            print(f"  提取 {year} 年的数据...")

            # 取当年的数据
            da_year = da.sel(time=str(year))
            
            # 求年平均（避免多时段）
            da_year_mean = da_year.mean(dim="time", skipna=True)

            # 裁剪
            da_clip = da_year_mean.rio.clip(mask_shp.geometry, mask_shp.crs, drop=True)

            # 输出路径
            out_tif = os.path.join(output_dir, f"EU_SMrz_{year}.tif")
            da_clip.rio.to_raster(out_tif, compress="LZW")

            print(f"  已保存: {out_tif}")
