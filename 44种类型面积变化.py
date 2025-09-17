import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --------------------------------
# 定义 CORINE 小类 -> 名称
acc_to_name = {
    111: "Continuous urban fabric", 112: "Discontinuous urban fabric",
    121: "Industrial or commercial units", 122: "Road and rail networks and associated land",
    123: "Port areas", 124: "Airports", 131: "Mineral extraction sites",
    132: "Dump sites", 133: "Construction sites", 141: "Green urban areas",
    142: "Sport and leisure facilities", 211: "Non irrigated arable land",
    212: "Permanently irrigated land", 213: "Rice fields", 221: "Vineyards",
    222: "Fruit trees and berry plantations", 223: "Olive groves",
    231: "Pastures", 241: "Annual crops associated with permanent crops",
    242: "Complex cultivation patterns", 243: "Land principally occupied by agriculture, with significant areas of natural vegetation",
    244: "Agro forestry areas", 311: "Broadleaved forest", 312: "Coniferous forest",
    313: "Mixed forest", 321: "Natural grasslands", 322: "Moors and heathland",
    323: "Sclerophyllous vegetation", 324: "Transitional woodland shrub",
    331: "Beaches, dunes, sands", 332: "Bare rocks", 333: "Sparsely vegetated areas",
    334: "Burnt areas", 335: "Glaciers and perpetual snow", 411: "Inland marshes",
    412: "Peat bogs", 421: "Salt marshes", 422: "Salines", 423: "Intertidal flats",
    511: "Water courses", 512: "Water bodies", 521: "Coastal lagoons",
    522: "Estuaries", 523: "Sea and ocean", 999: "Nodata"
}

# 小类 -> 大类（取首位数字）
acc_to_major = {acc: int(str(acc)[0]) if acc != 999 else np.nan for acc in acc_to_name.keys()}

# 大类编号 -> 英文名
major_to_name = {
    1: "Artificial surfaces",
    2: "Agricultural areas",
    3: "Forest and seminatural areas",
    4: "Wetlands",
    5: "Water bodies"
}

# --------------------------------
# 输入数据
folder = r"G:\Project-yqr\CLC_new"
years = [2009, 2015, 2018]
files = {
    2009: "CLCACC_2009.tif",
    2015: "CLCACC_2015.tif",
    2018: "CLCACC_2018.tif"
}

# 分辨率
cell_area = 0.25  # km² (500m * 500m)

# 存放结果
minor_stats = {}
major_stats = {}

for year in years:
    path = os.path.join(folder, files[year])
    with rasterio.open(path) as src:
        arr = src.read(1)

    arr = np.where(np.isin(arr, [0, 999, 65535]), np.nan, arr)
    flat = arr[~np.isnan(arr)].astype(int)

    # 小类面积
    unique, counts = np.unique(flat, return_counts=True)
    minor_stats[year] = {u: c * cell_area for u, c in zip(unique, counts)}

    # 大类面积
    major_stats[year] = {}
    for u, c in zip(unique, counts):
        major = acc_to_major.get(u, np.nan)
        if not np.isnan(major):
            major_stats[year][major] = major_stats[year].get(major, 0) + c * cell_area

# --------------------------------
# 构建 DataFrame
df_minor = pd.DataFrame(minor_stats).sort_index()  # 行=小类，列=年份
df_major = pd.DataFrame(major_stats).sort_index()  # 行=大类，列=年份

# --------------------------------
# 打印面积前十的小类（各年份）
total_area = df_minor.sum(axis=1)
top10_classes = total_area.nlargest(10).index

print("面积前十的小类：")
for c in top10_classes:
    vals = [f"{df_minor.loc[c, y]:.2f}" for y in years]
    print(f"{c} - {acc_to_name[c]}: " + ", ".join(f"{y}:{v} km²" for y, v in zip(years, vals)))

# --------------------------------
# 绘图
fig, axes = plt.subplots(2, 1, figsize=(22, 12), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})

bar_width = 0.25
bar_spacing = 0.3
positions_minor = np.arange(len(df_minor.index))

# 颜色
colors = cm.Set2(np.linspace(0, 1, len(years)))

# ---------- 小类 ----------
for i, year in enumerate(years):
    axes[0].bar(
        positions_minor + (i - (len(years) - 1) / 2) * bar_spacing,
        df_minor[year].reindex(df_minor.index, fill_value=0).values,
        width=bar_width,
        label=str(year),
        color=colors[i]
    )

# 给前十小类加文字标注
for c in top10_classes:
    idx = df_minor.index.get_loc(c)
    val = df_minor.loc[c, years[-1]]  # 用最新年份标注
    axes[0].text(idx, val, acc_to_name[c], ha="center", va="bottom", rotation=90, fontsize=8)

axes[0].set_ylabel("Area (km²)")
axes[0].set_title("Land use change (44 classes)")
axes[0].legend()
axes[0].set_xticks(positions_minor)
axes[0].set_xticklabels(df_minor.index, rotation=90)

# ---------- 大类 ----------
# 计算大类在小类轴上的跨度
major_positions = {}
for major in df_major.index:
    sub_classes = [c for c in df_minor.index if acc_to_major[c] == major]
    left = positions_minor[df_minor.index.get_loc(sub_classes[0])]
    right = positions_minor[df_minor.index.get_loc(sub_classes[-1])]
    center = (left + right) / 2
    major_positions[major] = (left, right, center)

for i, year in enumerate(years):
    for major, (left, right, center) in major_positions.items():
        total_width = (right - left + 1) * 0.8
        bar_width_major = total_width / len(years)
        axes[1].bar(
            center - total_width/2 + i * bar_width_major + bar_width_major/2,
            df_major.loc[major, year],
            width=bar_width_major,
            color=colors[i],
            label=str(year) if major == df_major.index[0] else ""
        )

axes[1].set_ylabel("Area (km²)")
axes[1].set_title("Land use change (5 major classes)")
axes[1].set_xticks([v[2] for v in major_positions.values()])
axes[1].set_xticklabels([major_to_name[m] for m in df_major.index])

# ---------- 分隔虚线（贯穿上下） ----------
for major, (left, right, _) in major_positions.items():
    axes[0].axvline(x=left - 0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(x=left - 0.5, color="gray", linestyle="--", alpha=0.5)

# 末尾也加一条
axes[0].axvline(x=positions_minor[-1] + 0.5, color="gray", linestyle="--", alpha=0.5)
axes[1].axvline(x=positions_minor[-1] + 0.5, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(r"G:\Project-yqr\CLC_new\landuse_change.png", dpi=300, bbox_inches="tight")  # ✅ 保存本地图片
plt.show()
