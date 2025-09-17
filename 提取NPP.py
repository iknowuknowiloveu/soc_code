import pandas as pd
import os

# ==================================
# 配置
# ==================================
base_file = r"G:\Project-yqr\组会\DATA\SOC\LUCAS\BD_filled_select.csv"
npp_folder = r"G:\Project-yqr\组会\DATA\SOC\soc_npp_point"
output_file = r'G:\Project-yqr/NPP_2009_2018.csv'

years = list(range(2009, 2019))

# ==================================
# 读取基准点文件
# ==================================
points_df = pd.read_csv(base_file)
if "ID" not in points_df.columns:
    points_df = points_df.reset_index().rename(columns={"index": "ID"})

points = points_df[["ID", "Latitude", "Longitude"]].copy()

# ==================================
# 读取所有 NPP CSV 并合并
# ==================================
all_files = [os.path.join(npp_folder, f) for f in os.listdir(npp_folder) if f.endswith(".csv")]

npp_list = []
for f in all_files:
    df = pd.read_csv(f)
    # 只保留需要的列
    df = df[["ID", "Latitude", "Longitude", "Date", "MOD17A3HGF_061_Npp_500m"]].copy()
    # 转换年份
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    npp_list.append(df)

npp_df = pd.concat(npp_list, ignore_index=True)

# ==================================
# 过滤目标年份
# ==================================
npp_df = npp_df[npp_df["Year"].isin(years)]

# ==================================
# Pivot：ID → 年份列
# ==================================
pivot_df = npp_df.pivot_table(
    index=["ID", "Latitude", "Longitude"],
    columns="Year",
    values="MOD17A3HGF_061_Npp_500m",
    aggfunc="mean"   # 如果有重复点，取平均
).reset_index()

# 列排序：ID, Lat, Lon, 年份...
pivot_df = pivot_df[["ID", "Latitude", "Longitude"] + years]

# ==================================
# 和基准点对齐（只保留基准点中的 ID）
# ==================================
final_df = points.merge(pivot_df, on=["ID", "Latitude", "Longitude"], how="left")

# 保存
final_df.to_csv(output_file, index=False)

print(f"✅ 提取完成，输出文件：{output_file}")
