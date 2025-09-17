import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ----------------------------
# 文件路径
# ----------------------------
folder = r"G:\Project-yqr\CLC_new\change_by_ACC"
minor_types_file = os.path.join(folder, "minor_transition_types.txt")
major_types_file = os.path.join(folder, "major_transition_types.txt")

transitions = {
    "2009_to_2015": ("minor_transition_2009_to_2015.tif", "major_transition_2009_to_2015.tif"),
    "2015_to_2018": ("minor_transition_2015_to_2018.tif", "major_transition_2015_to_2018.tif"),
    "2009_to_2018": ("minor_transition_2009_to_2018.tif", "major_transition_2009_to_2018.tif")
}

cell_area = 0.25  # km² (500m * 500m)

# ----------------------------
# 读取类型映射
# ----------------------------
def read_type_file(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式：ID 1: label
            if line.startswith("ID"):
                parts = line.split(":", 1)
                id_str = parts[0].replace("ID", "").strip()
                label = parts[1].strip()
                try:
                    d[int(id_str)] = label
                except ValueError:
                    continue
    return d

minor_type_dict = read_type_file(minor_types_file)
major_type_dict = read_type_file(major_types_file)

# ----------------------------
# 找到自己变自己的 ID
# ----------------------------
def get_self_transition_ids(type_dict):
    self_ids = []
    for tid, label in type_dict.items():
        if "→" in label:
            parts = [p.strip() for p in label.split("→")]
            if len(parts) >= 2 and parts[0] == parts[1]:
                self_ids.append(tid)
    return self_ids

minor_self_ids = get_self_transition_ids(minor_type_dict)
major_self_ids = get_self_transition_ids(major_type_dict)

# ----------------------------
# 计算每个时间段面积前五名
# ----------------------------
def calc_top5(raster_file, type_dict, self_ids):
    with rasterio.open(raster_file) as src:
        arr = src.read(1)
    arr = arr.flatten()
    arr = arr[~np.isnan(arr)]
    arr = arr[~np.isin(arr, self_ids)]  # 排除自己变自己的
    if len(arr) == 0:
        return []
    unique, counts = np.unique(arr, return_counts=True)
    areas = counts * cell_area
    ids_sorted = unique[np.argsort(-areas)]
    areas_sorted = areas[np.argsort(-areas)]
    top5 = []
    for tid, area in zip(ids_sorted[:5], areas_sorted[:5]):
        label = type_dict.get(tid, "Unknown")
        top5.append((tid, label, area))
    return top5

# ----------------------------
# 绘制条形图
# ----------------------------
def plot_top5(transitions, minor_type_dict, major_type_dict,
              minor_self_ids, major_self_ids, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    for col, (period, (minor_file, major_file)) in enumerate(transitions.items()):
        top5_minor = calc_top5(os.path.join(folder, minor_file), minor_type_dict, minor_self_ids)
        top5_major = calc_top5(os.path.join(folder, major_file), major_type_dict, major_self_ids)

        # Minor
        ax = axes[0, col]
        ids = [str(t[0]) for t in top5_minor]
        areas = [t[2] for t in top5_minor]
        labels = [t[1] for t in top5_minor]
        bars = ax.bar(range(len(top5_minor)), areas, color="skyblue")
        ax.set_title(f"{period} Minor Top5")
        ax.set_xticks(range(len(top5_minor)))
        ax.set_xticklabels(labels, rotation=90, fontsize=10)
        ax.set_ylabel("Area (km²)")
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{areas[i]:.0f}", ha='center', va='bottom', fontsize=9)

        # Major
        ax = axes[1, col]
        ids = [str(t[0]) for t in top5_major]
        areas = [t[2] for t in top5_major]
        labels = [t[1] for t in top5_major]
        bars = ax.bar(range(len(top5_major)), areas, color="salmon")
        ax.set_title(f"{period} Major Top5")
        ax.set_xticks(range(len(top5_major)))
        ax.set_xticklabels(labels, rotation=90, fontsize=10)
        ax.set_ylabel("Area (km²)")
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{areas[i]:.0f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 图已保存：{save_path}")

# ----------------------------
# 输出到图片
# ----------------------------
save_path = os.path.join(folder, "transition_top5_vertical_labels.png")
plot_top5(transitions, minor_type_dict, major_type_dict, minor_self_ids, major_self_ids, save_path)

# ----------------------------
# 打印面积前五名（排除自己变自己的）
# ----------------------------
for period, (minor_file, major_file) in transitions.items():
    print(f"--- {period} Minor transitions Top 5 ---")
    top5_minor = calc_top5(os.path.join(folder, minor_file), minor_type_dict, minor_self_ids)
    for tid, label, area in top5_minor:
        print(f"{tid}: {label}: {area:.2f} km²")

    print(f"\n--- {period} Major transitions Top 5 ---")
    top5_major = calc_top5(os.path.join(folder, major_file), major_type_dict, major_self_ids)
    for tid, label, area in top5_major:
        print(f"{tid}: {label}: {area:.2f} km²")
    print()
