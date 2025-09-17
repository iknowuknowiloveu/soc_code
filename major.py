import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import logging
import gc
from rasterio.transform import from_bounds

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 输入路径
input_dir = r"E:\Project-yqr\EU_TIFF_500m\transitions"
periods = [(2009, 2015), (2015, 2018), (2009, 2018)]
map_crs = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52)

# 自己转自己的 TID
self_transition_tids = [1, 7, 13, 19, 25]

# 读取转换类型
def load_transition_types():
    transition_types = {}
    types_file = os.path.join(input_dir, "major_transition_types.txt")
    try:
        with open(types_file, 'r', encoding='utf-8') as f:
            for line in f:
                tid = int(line.split(":")[0].replace("ID ", "").strip())
                name = line.split(":")[1].strip()
                # 跳过自己转自己
                if tid in self_transition_tids:
                    logger.info(f"跳过自己转自己转换: {name} (tid={tid})")
                    continue
                transition_types[tid] = name
        logger.info(f"📝 已加载并过滤自己转自己转换类型: {types_file}, 共 {len(transition_types)} 种")
    except Exception as e:
        logger.error(f"❌ 加载大类转换类型失败: {str(e)}")
    return transition_types

# 读取栅格并过滤自己转自己
def read_transition_tiff(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodatavals[0] if src.nodatavals else 0
        # 将 nodata 和自己转自己 TID 转为 nan
        data = np.where(np.isin(data, self_transition_tids) | (data == nodata), np.nan, data)
        bounds = src.bounds
    return data, bounds

# 绘图
def plot_major_transition_maps():
    transition_types = load_transition_types()
    if not transition_types:
        logger.error("❌ 没有有效的转换类型")
        return

    # 读取栅格
    datasets = []
    for from_year, to_year in periods:
        path = os.path.join(input_dir, f"major_transition_{from_year}_to_{to_year}.tif")
        data, bounds = read_transition_tiff(path)
        unique_vals = np.unique(data[~np.isnan(data)]).astype(int)
        logger.info(f"加载 {os.path.basename(path)}，有效 TID: {unique_vals}")
        datasets.append((from_year, to_year, data, bounds))

    # 计算实际存在的有效 TID
    all_valid_tids = sorted({int(val) for _, _, data, _ in datasets for val in np.unique(data[~np.isnan(data)])})
    logger.info(f"实际存在的有效 TID: {all_valid_tids}")

    # 建立颜色映射
    colors = plt.get_cmap("tab20").colors
    cmap = mcolors.ListedColormap([colors[(tid - 1) % 20] for tid in all_valid_tids])
    norm = mcolors.BoundaryNorm(
        boundaries=[tid - 0.5 for tid in all_valid_tids] + [max(all_valid_tids) + 0.5], 
        ncolors=len(all_valid_tids)
    )

    # 创建子图
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), subplot_kw={'projection': map_crs})
    if len(datasets) == 1:
        axes = [axes]

    for ax, (from_year, to_year, data, bounds) in zip(axes, datasets):
        rows, cols = data.shape
        x = np.linspace(bounds.left, bounds.right, cols + 1)
        y = np.linspace(bounds.bottom, bounds.top, rows + 1)
        X, Y = np.meshgrid(x, y)

        im = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='flat', transform=ccrs.PlateCarree())

        ax.set_extent([-10, 35, 35, 71], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='gray', linewidth=0.4, alpha=0.8)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator([0, 10, 20, 30])
        gl.ylocator = mticker.FixedLocator([40, 50, 60, 70])
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True

        ax.text(0.04, 0.96, f"{from_year} → {to_year}", transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

        # 打印每个 TID 的像素数量
        for tid in all_valid_tids:
            count = np.sum(data == tid)
            logger.info(f"{from_year}->{to_year} TID {tid}: {count} pixels")

    # 色条
    cbar_ax = fig.add_axes([0.22, -0.05, 0.56, 0.045])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=all_valid_tids)
    cb.ax.set_xticklabels([transition_types[tid] for tid in all_valid_tids], rotation=45, ha='right', fontsize=10)
    cb.set_label("Major Transition Types", fontsize=16, weight='bold')

    plt.subplots_adjust(bottom=0.2)
    plt.show()
    gc.collect()
    logger.info("✅ 完成大类转换空间分布图显示")

if __name__ == "__main__":
    plot_major_transition_maps()
