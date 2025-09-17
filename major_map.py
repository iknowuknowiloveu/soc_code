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

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 输入输出路径
input_dir = r"G:\Project-yqr\transitions"
output_dir = r"G:\Project-yqr\outputs"
os.makedirs(output_dir, exist_ok=True)

map_crs = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52)
periods = [(2009, 2015), (2015, 2018), (2009, 2018)]

# 透明类别（No Change）
transparent_tids = [1, 7, 13, 19, 25]
no_change_val = 0

# 高对比深色系颜色（按需扩展）
deep_colors = [
    "#8B0000",  # 深红
    "#006400",  # 深绿
    "#B22222",  # 火砖红
    "#483D8B",  # 深岩蓝
    "#D2691E",  # 巧克力棕
    "#800080",  # 紫色
    "#FF8C00",  # 暗橙
    "#2F4F4F",  # 深灰
    "#4B0082",  # 靛青
    "#A52A2A",  # 棕色
    "#6A5ACD",  # 深石蓝
    "#8B4513",  # 马鞍棕
    "#FF4500",  # 橙红
    "#551A8B",  # 深紫
    "#CD5C5C",  # 印度红
    "#008B8B",  # 深青色
    "#A0522D",  # 赭色
    "#8B008B",  # 深紫红
    "#556B2F",  # 暗橄榄绿
    "#191970"   # 午夜蓝
]


# 读取转换类型
def load_transition_types():
    transition_types = {}
    types_file = os.path.join(input_dir, "major_transition_types.txt")
    try:
        with open(types_file, 'r', encoding='utf-8') as f:
            for line in f:
                tid = int(line.split(":")[0].replace("ID ", "").strip())
                name = line.split(":")[1].strip()
                transition_types[tid] = name
        logger.info(f"📝 已加载转换类型: {len(transition_types)} 种")
    except Exception as e:
        logger.error(f"❌ 加载转换类型失败: {e}")
        return None
    return transition_types

# 读取栅格
def read_transition_tiff(path):
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodatavals[0] if src.nodatavals else 0
            data = np.where(data == nodata, np.nan, data)
            bounds = src.bounds
            return data, bounds
    except Exception as e:
        logger.error(f"❌ 加载 {path} 失败: {e}")
        return None, None

# 绘图
def plot_major_transition_maps():
    transition_types = load_transition_types()
    if transition_types is None:
        return

    datasets = []
    used_tids = set()
    for from_year, to_year in periods:
        path = os.path.join(input_dir, f"major_transition_{from_year}_to_{to_year}.tif")
        data, bounds = read_transition_tiff(path)
        if data is not None:
            # 替换透明类别为 0
            data_masked = data.copy()
            data_masked[np.isin(data_masked, transparent_tids)] = no_change_val
            valid_tids = np.unique(data_masked[~np.isnan(data_masked)]).astype(int)
            used_tids.update(valid_tids)
            datasets.append((from_year, to_year, data_masked, bounds))

    # 排序非透明类别
    non_transparent_tids = sorted([tid for tid in used_tids if tid != no_change_val])

    # 构造颜色映射：0=白色, 其他类别深色系
    all_tids = [no_change_val] + non_transparent_tids
    color_list = ['white'] + deep_colors[:len(non_transparent_tids)]
    cmap = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(boundaries=[t-0.5 for t in all_tids]+[all_tids[-1]+0.5], ncolors=len(all_tids))

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), subplot_kw={'projection': map_crs})
    if len(datasets) == 1:
        axes = [axes]

    for ax, (from_year, to_year, data_masked, bounds) in zip(axes, datasets):
        ax.set_extent([-10, 35, 35, 71], crs=ccrs.PlateCarree())
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        im = ax.imshow(data_masked, cmap=cmap, norm=norm, extent=extent,
                       transform=ccrs.PlateCarree(), origin='upper')
        ax.text(0.04, 0.96, f"{from_year} → {to_year}", transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='gray', linewidth=0.4, alpha=0.8)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator([0,10,20,30])
        gl.ylocator = mticker.FixedLocator([40,50,60,70])
        gl.top_labels = False
        gl.right_labels = False

    # 色带：只显示 No Change + 非透明类别
    cbar_ax = fig.add_axes([0.22, -0.05, 0.56, 0.045])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=all_tids)
    cb_labels = ['No Change'] + [transition_types[tid] for tid in non_transparent_tids]
    cb.ax.set_xticklabels(cb_labels, rotation=45, ha='right', fontsize=10)
    cb.set_label("Major Transition Types", fontsize=16, weight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "major_transition_maps_final.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    gc.collect()
    logger.info(f"✅ 完成绘图并保存到 {save_path}")

# 主程序
def main():
    plot_major_transition_maps()

if __name__ == "__main__":
    main()
