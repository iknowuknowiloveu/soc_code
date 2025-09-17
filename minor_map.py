import os
import random
import colorsys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import logging
import gc
from collections import Counter

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# 配置
# -----------------------------
input_dir = r"G:\Project-yqr\CLC_new\change_by_ACC"
output_dir = r"G:\Project-yqr\CLC_new\outputs"
os.makedirs(output_dir, exist_ok=True)

map_crs = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52)
periods = [(2009, 2015), (2015, 2018), (2009, 2018)]

# 前20用固定的深色系、均衡（红/橙/棕/紫/蓝/绿都有，避免明黄）
top_palette = [
    "#8B0000",  # 深红
    "#1F78B4",  # 深蓝
    "#2E8B57",  # 海洋绿
    "#6A3D9A",  # 深紫
    "#E6550D",  # 深橙
    "#A0522D",  # 赭棕
    "#483D8B",  # 深岩蓝
    "#008B8B",  # 深青
    "#B22222",  # 火砖红
    "#2F4F4F",  # 深灰青
    "#b15928",  # 棕橙
    "#004B87",  # 海军蓝
    "#556B2F",  # 暗橄榄绿
    "#7F2704",  # 深棕
    "#4B0082",  # 靛青
    "#264653",  # 深水鸭
    "#8B4513",  # 马鞍棕
    "#3F007D",  # 靛紫
    "#0B3D91",  # 深钴蓝
    "#006400",  # 深绿
]

# 随机色：深色系、避开亮黄；且不与已用颜色重复
def generate_deep_random_colors(n, existing_hex_set, seed=42):
    random.seed(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        # H, L, S：控制深色与对比度；避开亮黄（大约 45°~65°）
        h = random.random()
        if 0.12 < h < 0.20:  # 避免黄色/亮黄橙
            continue
        l = random.uniform(0.30, 0.55)  # 偏深
        s = random.uniform(0.55, 0.90)  # 比较饱和
        r, g, b = colorsys.hls_to_rgb(h, l, s)  # 注意 colorsys 用 HLS
        hexc = "#{:02X}{:02X}{:02X}".format(int(r*255), int(g*255), int(b*255))
        if hexc.lower() not in existing_hex_set:
            existing_hex_set.add(hexc.lower())
            out.append(hexc)
    return out

# -----------------------------
# 读取转换类型文件，自动识别 No Change
# -----------------------------
def load_transition_types(file_path):
    transition_types = {}
    transparent_tids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' not in line:
                    continue
                tid_str, name = line.split(":", 1)
                tid = int(tid_str.replace("ID", "").strip())
                name = name.strip()
                transition_types[tid] = name
                # 形如 "X → X" 视为 No Change
                if '→' in name:
                    a, b = name.split('→', 1)
                    if a.strip() == b.strip():
                        transparent_tids.add(tid)
        logger.info(f"📝 载入 {file_path}：共 {len(transition_types)} 类，其中 No Change={len(transparent_tids)}")
    except Exception as e:
        logger.error(f"❌ 加载类型失败: {e}")
        return {}, set()
    return transition_types, transparent_tids

# -----------------------------
# 读取栅格
# -----------------------------
def read_transition_raster(path):
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodatavals[0] if src.nodatavals else np.nan
            data = np.where(data == nodata, np.nan, data)
            bounds = src.bounds
            return data, bounds
    except Exception as e:
        logger.error(f"❌ 读取失败 {path}: {e}")
        return None, None

# -----------------------------
# 绘制（minor 专用）
# -----------------------------
def plot_minor_transition_maps():
    raster_prefix = "minor"
    type_file = os.path.join(input_dir, f"{raster_prefix}_transition_types.txt")

    transition_types, transparent_tids = load_transition_types(type_file)
    if not transition_types:
        return

    datasets = []
    all_counts_excl_nochange = Counter()  # 统计热门（排除 No Change）
    all_tids_seen = set()

    # 读取三期数据
    for fy, ty in periods:
        raster_file = os.path.join(input_dir, f"{raster_prefix}_transition_{fy}_to_{ty}.tif")
        data, bounds = read_transition_raster(raster_file)
        if data is None:
            continue

        # 统计热门（不含 No Change / 透明类别）
        flat = data[~np.isnan(data)].astype(int)
        if transparent_tids:
            flat = flat[~np.isin(flat, list(transparent_tids))]
        all_counts_excl_nochange.update(flat.tolist())

        # 将透明类别重编码为 0（No Change）
        data_rec = data.copy()
        if transparent_tids:
            data_rec[np.isin(data_rec, list(transparent_tids))] = 0

        # 收集 cmap 的完整类别集合（含 0）
        tids_here = np.unique(data_rec[~np.isnan(data_rec)]).astype(int)
        all_tids_seen.update(tids_here.tolist())

        datasets.append((fy, ty, data_rec, bounds))

    if not datasets:
        logger.error("❌ 没有可绘制的数据栅格")
        return

    # 前20热门（若不足20，按实际数量）
    top20_tids = [tid for tid, _ in all_counts_excl_nochange.most_common(20)]
    logger.info(f"Top-20 类别: {top20_tids}")

    # ——构建颜色映射——
    # 0 -> 白色
    color_map = {0: "#FFFFFF"}
    used_hex = set([c.lower() for c in ["#FFFFFF"] + top_palette])  # 用于避免重复

    # 给前20分配固定深色系
    for tid, col in zip(top20_tids, top_palette):
        color_map[tid] = col

    # 其余类别（除了 0 和 top20），分配深色随机色
    others = sorted([tid for tid in all_tids_seen if tid not in top20_tids and tid != 0])
    need_n = len(others)
    if need_n > 0:
        extra_colors = generate_deep_random_colors(need_n, used_hex, seed=42)
        for tid, col in zip(others, extra_colors):
            color_map[tid] = col

    # 保证对所有出现过的 tid 都有颜色
    for tid in all_tids_seen:
        if tid not in color_map:
            # 理论不该发生；兜底用深灰
            color_map[tid] = "#444444"

    # ——构建 cmap / norm（离散）——
    all_tids_sorted = sorted(all_tids_seen)  # 包含 0
    color_list = [color_map[tid] for tid in all_tids_sorted]
    cmap = mcolors.ListedColormap(color_list)
    cmap.set_bad(alpha=0)  # NaN 透明
    boundaries = [tid - 0.5 for tid in all_tids_sorted] + [all_tids_sorted[-1] + 0.5]
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=cmap.N)

    # ——绘图（严格沿用你的大类绘制方法）——
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), subplot_kw={'projection': map_crs})
    if len(datasets) == 1:
        axes = [axes]

    for ax, (fy, ty, data_rec, bounds) in zip(axes, datasets):
        ax.set_extent([-10, 35, 35, 71], crs=ccrs.PlateCarree())
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        im = ax.imshow(
            data_rec, cmap=cmap, norm=norm, extent=extent,
            transform=ccrs.PlateCarree(), origin='upper'
        )
        ax.text(0.04, 0.96, f"{fy} → {ty}", transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='gray', linewidth=0.4, alpha=0.8)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator([0, 10, 20, 30])
        gl.ylocator = mticker.FixedLocator([40, 50, 60, 70])
        gl.top_labels = False
        gl.right_labels = False

    # ——Colorbar：只显示 No Change + 前20 ——
    show_tids = [0] + top20_tids
    show_colors = [color_map[tid] for tid in show_tids]
    show_labels = ["No Change"] + [transition_types.get(tid, f"TID {tid}") for tid in top20_tids]

    # 专门为 colorbar 创建一个 cmap/norm
    show_cmap = mcolors.ListedColormap(show_colors)
    show_norm = mcolors.BoundaryNorm(
        boundaries=[i - 0.5 for i in range(len(show_tids) + 1)],
        ncolors=len(show_tids)
    )

    cbar_ax = fig.add_axes([0.22, -0.05, 0.56, 0.045])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=show_cmap, norm=show_norm),
        cax=cbar_ax, orientation='horizontal',
        ticks=range(len(show_tids))
    )
    cb.ax.set_xticklabels(show_labels, rotation=45, ha='right', fontsize=10)
    cb.set_label("Minor Transition Types (Top 20 shown)", fontsize=16, weight='bold')


    plt.tight_layout()
    save_path = os.path.join(output_dir, "minor_transition_maps_top20.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    gc.collect()
    logger.info(f"✅ 完成绘图并保存: {save_path}")

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    plot_minor_transition_maps()
