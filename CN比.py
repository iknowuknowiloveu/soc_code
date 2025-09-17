import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import logging
import gc
import matplotlib.font_manager as fm

# ===================== Logging =====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== Input Paths =====================
soc_dir = r"E:\Project-yqr\new\results\weight"   # SOC for three years
n_path = r"E:\Project-yqr\EU_TIFF_500m\EU_N.tif"  # N data (one year only)
years = [2009, 2015, 2018]

map_crs = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52)

# ===================== Color Settings =====================
bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 24]
colors = [
    "#006837", "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b",
    "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026", "#67001f"
]
cmap_soc = mcolors.ListedColormap(colors)
norm_soc = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(colors))

# CN ratio colorbar
bounds_cn = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
cmap_cn = plt.cm.viridis
norm_cn = mcolors.BoundaryNorm(boundaries=bounds_cn, ncolors=cmap_cn.N)

# ===================== Font =====================
try:
    arial_path = fm.findfont("Arial")
    plt.rcParams["font.family"] = "Arial"
    logger.info(f"✅ Using Arial font: {arial_path}")
except:
    plt.rcParams["font.family"] = "DejaVu Sans"
    logger.warning("⚠️ Arial font not found, using DejaVu Sans instead")

plt.rcParams["font.size"] = 16

# ===================== Helper Function =====================
def match_raster_to_ref(src_path, ref_profile):
    """Resample src_path raster to reference grid and mask NoData"""
    with rasterio.open(src_path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan  # Remove NoData
        matched = np.empty((ref_profile['height'], ref_profile['width']), dtype=np.float32)
        reproject(
            source=data,
            destination=matched,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile['transform'],
            dst_crs=ref_profile['crs'],
            resampling=Resampling.bilinear
        )
        return np.ma.masked_invalid(matched)

# ===================== Load SOC Data =====================
soc_datasets = []
ref_profile = None
for year in years:
    path = os.path.join(soc_dir, f"weight_pred_{year}.tif")
    if not os.path.exists(path):
        logger.warning(f"❌ File not found: {path}")
        continue
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        masked = np.ma.masked_invalid(data)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        soc_datasets.append((year, masked, extent))
        logger.info(f"✅ Loaded SOC {year}, shape={masked.shape}, max={np.nanmax(masked):.2f}")
        if ref_profile is None:
            ref_profile = src.profile

# ===================== Load & Match N Data =====================
n_matched = match_raster_to_ref(n_path, ref_profile)
logger.info(f"✅ N data aligned, shape={n_matched.shape}, max={np.nanmax(n_matched):.2f}")

# ===================== Calculate CN Ratio =====================
cn_datasets = []
for (year, soc_data, extent) in soc_datasets:
    cn_ratio = np.divide(soc_data, n_matched, out=np.full_like(soc_data, np.nan), where=(n_matched > 0))
    cn_datasets.append((year, np.ma.masked_invalid(cn_ratio), extent))
    logger.info(f"✅ CN ratio for {year} calculated, max={np.nanmax(cn_ratio):.2f}")

# ===================== Plot N Distribution =====================
fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': map_crs})
ax.set_extent([-10, 35, 35, 71], crs=ccrs.PlateCarree())
im = ax.imshow(n_matched, cmap=cmap_soc, norm=norm_soc,
               extent=extent, transform=ccrs.PlateCarree(), origin='upper')
ax.set_title("Nitrogen Distribution", fontsize=18, fontweight='bold')
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='gray', linewidth=0.4, alpha=0.8)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator([0, 10, 20, 30])
gl.ylocator = mticker.FixedLocator([40, 50, 60, 70])
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.1, ticks=bounds)
cbar.ax.set_xticklabels([str(b) for b in bounds], fontsize=12)
cbar.set_label("N (kg/m²)", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# ===================== Plot CN Ratio for 3 Years =====================
fig, axes = plt.subplots(1, len(cn_datasets), figsize=(18, 6), subplot_kw={'projection': map_crs})
for ax, (year, cn_data, extent) in zip(axes, cn_datasets):
    ax.set_extent([-10, 35, 35, 71], crs=ccrs.PlateCarree())
    im = ax.imshow(cn_data, cmap=cmap_cn, norm=norm_cn,
                   extent=extent, transform=ccrs.PlateCarree(), origin='upper')
    ax.set_title(f"CN Ratio {year}", fontsize=18, fontweight='bold')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='gray', linewidth=0.4, alpha=0.8)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([0, 10, 20, 30])
    gl.ylocator = mticker.FixedLocator([40, 50, 60, 70])
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

# Common colorbar
cbar_ax = fig.add_axes([0.25, 0.015, 0.5, 0.04])  # Moved further down
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', pad=0.01, ticks=bounds_cn)
cbar.ax.set_xticklabels([str(b) for b in bounds_cn], fontsize=12)
cbar.set_label("CN Ratio (SOC/N)", fontsize=14, weight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjusted to reserve space for colorbar
plt.show()

gc.collect()
logger.info("✅ Nitrogen distribution map & CN ratio maps generated successfully")