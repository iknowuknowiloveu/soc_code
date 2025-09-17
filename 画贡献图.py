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

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置冻结变量列表
variables = ['su', 'CLC', 'NPP', 'nfer_crop_no3', 'ndep_nhx', 'fd', 'prcptot']
years = [2015, 2018]
map_crs = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52)

# 字体设置
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16

# 颜色：中间白色，对应 0
color_palette = [
    "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",  # 蓝（负）
    "#FFFFFF",  # 中间白色
    "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026", "#67001f"  # 红（正）
]

def load_dataset(path):
    if not os.path.exists(path):
        logger.warning(f"❌ 文件不存在: {path}")
        return None
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        # 不掩码0，保持数据完整
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        return data, extent

def get_dynamic_color_bounds(contrib_type, datasets):
    # 原始边界
    base_bounds = {
        'abs': [-2.5, -1, -0.1, -0.01, -0.001, -0.000001, 0.000001, 0.001, 0.01, 0.05, 0.1, 1, 2],
        'rel': [-0.7, -0.2, -0.01, -0.005, -0.001, -0.000001, 0.000001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.4]
    }
    
    # 收集所有数据以计算最大值和最小值
    all_data = []
    for _, data, _ in datasets:
        all_data.append(data)
    all_data = np.concatenate([data.ravel() for data in all_data])
    data_max = np.nanmax(all_data)
    data_min = np.nanmin(all_data)
    
    # 调试日志
    logger.info(f"计算 {contrib_type} 数据范围: max={data_max:.3f}, min={data_min:.3f}")
    
    # 只替换最大值和最小值，并保留小数点后一位
    bounds = base_bounds[contrib_type].copy()
    logger.info(f"原始 {contrib_type} 边界: {bounds}")
    
    # 直接使用数据的最小值和最大值，四舍五入到小数点后一位
    bounds[0] = round(data_min, 1)
    bounds[-1] = round(data_max, 1)
    
    logger.info(f"调整后 {contrib_type} 边界: {bounds}")
    return bounds

def plot_contribution(contrib_type, title_label, freeze_var, output_dir):
    datasets = []
    for year in years:
        tif_path = os.path.join(output_dir, f'{contrib_type}_contrib_{freeze_var}_{year}.tif')
        result = load_dataset(tif_path)
        if result:
            data, extent = result
            datasets.append((year, data, extent))
            logger.info(f"✅ 加载 {contrib_type} {year}, max={np.nanmax(data):.3f}, min={np.nanmin(data):.3f}")

    if not datasets:
        logger.error(f"❌ 无有效数据: {contrib_type}")
        return

    # 动态调整 color_bounds 的最大值和最小值
    bounds = get_dynamic_color_bounds(contrib_type, datasets)
    cmap = mcolors.ListedColormap(color_palette[:len(bounds)-1])
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds)-1)

    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 6), subplot_kw={'projection': map_crs})
    if len(datasets) == 1:
        axes = [axes]

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

    cbar_ax = fig.add_axes([0.22, 0.015, 0.56, 0.035])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=bounds)
    labels = []
    for b in bounds:
        if abs(b) < 1e-8:
            labels.append('0')
        else:
            labels.append(str(b))
    cb.ax.set_xticklabels(labels, fontsize=14)
    cb.set_label(title_label, fontsize=16, weight='bold')
    cb.ax.tick_params(labelsize=14)

    # 保存为 JPG 文件
    output_path = os.path.join(output_dir, f'{contrib_type}_contrib_{freeze_var}.jpg')
    plt.savefig(output_path, format='jpg', dpi=300, bbox_inches='tight')
    logger.info(f"✅ 保存 {contrib_type} 贡献图到: {output_path}")

    plt.close(fig)  # 关闭图形以释放内存
    gc.collect()
    logger.info(f"✅ 成功绘制并保存 {contrib_type} 贡献图")

# 遍历冻结变量并绘制
for freeze_var in variables:
    logger.info(f"开始处理冻结变量: {freeze_var}")
    output_dir = fr'Project-yqr/new/results/weight/freeze/{freeze_var}/contribution'
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    # 分别绘制并保存
    plot_contribution('abs', f"Absolute Contribution of {freeze_var}", freeze_var, output_dir)
    plot_contribution('rel', f"Relative Contribution of {freeze_var}", freeze_var, output_dir)

logger.info("所有变量的贡献图绘制和保存完成！")