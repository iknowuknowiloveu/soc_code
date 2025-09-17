import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import logging
import gc
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 路径
input_dir = r'Project-yqr\EU_TIFF_500m'
years = [2009, 2015, 2018]
map_crs = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52)

# 字体兼容
try:
    arial_path = fm.findfont("Arial")
    plt.rcParams["font.family"] = "Arial"
    logger.info(f"✅ 使用 Arial 字体: {arial_path}")
except:
    plt.rcParams["font.family"] = "DejaVu Sans"
    logger.warning("⚠️ Arial 字体未找到，已自动回退为 DejaVu Sans")
plt.rcParams["font.size"] = 16

# CLC 类别定义（假设 1-44，基于标准 CLC 命名）
clc_classes = {
    'Artificial surfaces': [
        (1, 'Continuous urban fabric'),
        (2, 'Discontinuous urban fabric'),
        (3, 'Industrial or commercial units'),
        (4, 'Road and rail networks'),
        (5, 'Port areas'),
        (6, 'Airports'),
        (7, 'Mineral extraction sites'),
        (8, 'Dump sites'),
        (9, 'Construction sites'),
        (10, 'Green urban areas'),
        (11, 'Sport and leisure facilities')
    ],
    'Agricultural areas': [
        (12, 'Non-irrigated arable land'),
        (13, 'Permanently irrigated land'),
        (14, 'Rice fields'),
        (15, 'Vineyards'),
        (16, 'Fruit trees and berry plantations'),
        (17, 'Olive groves'),
        (18, 'Pastures'),
        (19, 'Annual crops with permanent crops'),
        (20, 'Complex cultivation patterns'),
        (21, 'Agriculture with natural vegetation'),
        (22, 'Agro-forestry areas')
    ],
    'Forest and semi-natural areas': [
        (23, 'Broad-leaved forest'),
        (24, 'Coniferous forest'),
        (25, 'Mixed forest'),
        (26, 'Natural grasslands'),
        (27, 'Moors and heathland'),
        (28, 'Sclerophyllous vegetation'),
        (29, 'Transitional woodland-shrub'),
        (30, 'Beaches, dunes, sands'),
        (31, 'Bare rocks'),
        (32, 'Sparsely vegetated areas'),
        (33, 'Burnt areas'),
        (34, 'Glaciers and perpetual snow')
    ],
    'Wetlands': [
        (35, 'Inland marshes'),
        (36, 'Peat bogs'),
        (37, 'Salt marshes'),
        (38, 'Salines'),
        (39, 'Intertidal flats')
    ],
    'Water bodies': [
        (40, 'Water courses'),
        (41, 'Water bodies'),
        (42, 'Coastal lagoons'),
        (43, 'Estuaries'),
        (44, 'Sea and ocean')
    ]
}

# 颜色分配
colors = [
    # Artificial surfaces (红色系，11 种)
    '#a50026', '#b2182b', '#d73027', '#e34a33', '#f46d43', '#fc8d59',
    '#fdbb84', '#fdd49e', '#fee8c8', '#fff7ec', '#ffffff',
    # Agricultural areas (黄色系，11 种)
    '#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a',
    '#e31a1c', '#bd0026', '#800026', '#b30000', '#d7301f',
    # Forest and semi-natural areas (绿色系，12 种)
    '#006837', '#1a9850', '#238b45', '#41ab5d', '#66bd63', '#78c679',
    '#a6d96a', '#c7e9b4', '#d9f0a3', '#edf8b1', '#f7fcb9', '#ffffe5',
    # Wetlands (紫色系，5 种)
    '#54278f', '#6a51a3', '#807dba', '#9e9ac8', '#bcbddc',
    # Water bodies (蓝色系，5 种)
    '#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6'
]
bounds = list(range(1, 46))  # 1 到 44，边界为 [1, 2, ..., 45]
cmap = mcolors.ListedColormap(colors[:44])  # 确保 44 种颜色
norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(colors[:44]))

# 读取栅格
datasets = []
for year in years:
    path = os.path.join(input_dir, f'EU_CLC_{year}.tif')
    if not os.path.exists(path):
        logger.warning(f"❌ 文件不存在: {path}")
        continue
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            data[data <= -9999] = np.nan  # 处理无效值
            masked = np.ma.masked_invalid(data)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            unique_vals = np.unique(masked[~masked.mask]).astype(int)
            datasets.append((year, masked, extent))
            logger.info(f"✅ 加载 {year}, shape={masked.shape}, unique values={unique_vals}")
    except Exception as e:
        logger.error(f"❌ 加载 {year} 失败: {str(e)}")
        continue

# 创建子图
fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), subplot_kw={'projection': map_crs})

for ax, (year, data, extent) in zip(axes, datasets):
    ax.set_extent([-10, 35, 35, 71], crs=ccrs.PlateCarree())
    im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent,
                   transform=ccrs.PlateCarree(), origin='upper')

    ax.text(0.04, 0.96, str(year), transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='gray', linewidth=0.4, alpha=0.8)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([0, 10, 20, 30])
    gl.ylocator = mticker.FixedLocator([40, 50, 60, 70])
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size': 16, 'rotation': 0}
    gl.ylabel_style = {'size': 16, 'rotation': 0}

# 添加图例
fig.subplots_adjust(bottom=0.3)  # 留出底部空间
legend_ax = fig.add_axes([0.1, 0.05, 0.8, 0.2])
legend_ax.axis('off')
legend_elements = []
legend_labels = []
for category, classes in clc_classes.items():
    legend_elements.append(Patch(color='none', label=category))  # 类别标题
    legend_labels.append(category)
    for code, name in classes:
        legend_elements.append(Patch(facecolor=colors[code-1], edgecolor='none'))
        legend_labels.append(f"{code}: {name}")
plt.legend(legend_elements, legend_labels, loc='center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 0.5), frameon=False, handlelength=1, handleheight=1)

plt.tight_layout()

# 保存为 JPG 文件
output_path = os.path.join(input_dir, 'EU_CLC_comparison.jpg')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
logger.info(f"📸 已保存图像到: {output_path}")


plt.show()
gc.collect()
logger.info("✅ 土地利用分布图生成成功（未保存图片）")