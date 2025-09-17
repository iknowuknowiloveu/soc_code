import os
import numpy as np
import rasterio

# ==============================
# 输入输出路径
# ==============================
input_dir = r"G:\Project-yqr\CLC_new"
output_dir = r"G:\Project-yqr\CLC_new\change_by_CHA"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# CLC 类别定义
# ==============================
clc_classes = {
    'Artificial surfaces': [(1, 'Continuous urban fabric'), (2, 'Discontinuous urban fabric'),
                            (3, 'Industrial or commercial units'), (4, 'Road and rail networks'),
                            (5, 'Port areas'), (6, 'Airports'), (7, 'Mineral extraction sites'),
                            (8, 'Dump sites'), (9, 'Construction sites'), (10, 'Green urban areas'),
                            (11, 'Sport and leisure facilities')],
    'Agricultural areas': [(12, 'Non-irrigated arable land'), (13, 'Permanently irrigated land'),
                           (14, 'Rice fields'), (15, 'Vineyards'), (16, 'Fruit trees and berry plantations'),
                           (17, 'Olive groves'), (18, 'Pastures'), (19, 'Annual crops with permanent crops'),
                           (20, 'Complex cultivation patterns'), (21, 'Agriculture with natural vegetation'),
                           (22, 'Agro-forestry areas')],
    'Forest and semi-natural areas': [(23, 'Broad-leaved forest'), (24, 'Coniferous forest'),
                                      (25, 'Mixed forest'), (26, 'Natural grasslands'),
                                      (27, 'Moors and heathland'), (28, 'Sclerophyllous vegetation'),
                                      (29, 'Transitional woodland-shrub'), (30, 'Beaches, dunes, sands'),
                                      (31, 'Bare rocks'), (32, 'Sparsely vegetated areas'),
                                      (33, 'Burnt areas'), (34, 'Glaciers and perpetual snow')],
    'Wetlands': [(35, 'Inland marshes'), (36, 'Peat bogs'), (37, 'Salt marshes'),
                 (38, 'Salines'), (39, 'Intertidal flats')],
    'Water bodies': [(40, 'Water courses'), (41, 'Water bodies'), (42, 'Coastal lagoons'),
                     (43, 'Estuaries'), (44, 'Sea and ocean')]
}

# ==============================
# 小类 → 大类映射
# ==============================
small_to_large = {}
small_id_map = {}
small_names_map = {}
cnt = 1
for cat, items in clc_classes.items():
    for cid, cname in items:
        small_to_large[cid] = cat
        small_id_map[cid] = cnt
        small_names_map[cid] = cname
        cnt += 1

# 大类连续编号
large_names = list(clc_classes.keys())
large_id_map = {name: i+1 for i, name in enumerate(large_names)}

# ==============================
# 生成大类、小类 TID 映射表
# ==============================
# 大类映射
major_transition_id = 1
major_transition_list = []
for from_cat in large_names:
    for to_cat in large_names:
        major_transition_list.append((major_transition_id, from_cat, to_cat))
        major_transition_id += 1

major_file = os.path.join(output_dir, "major_transition_types.txt")
with open(major_file, "w", encoding="utf-8") as f:
    for tid, f_name, t_name in major_transition_list:
        f.write(f"ID {tid}: {f_name} → {t_name}\n")

# 小类映射
minor_transition_id = 1
minor_transition_list = []
for from_cid in small_id_map:
    for to_cid in small_id_map:
        f_name = small_names_map[from_cid]
        t_name = small_names_map[to_cid]
        minor_transition_list.append((minor_transition_id, f_name, t_name))
        minor_transition_id += 1

minor_file = os.path.join(output_dir, "minor_transition_types.txt")
with open(minor_file, "w", encoding="utf-8") as f:
    for tid, f_name, t_name in minor_transition_list:
        f.write(f"ID {tid}: {f_name} → {t_name}\n")

print(f"✅ Major mapping saved: {major_file}")
print(f"✅ Minor mapping saved: {minor_file}")

# ==============================
# 转变计算函数（全栅格向量化）
# ==============================
def compute_changes(from_file, to_file, period_name):
    with rasterio.open(from_file) as src_from, rasterio.open(to_file) as src_to:
        arr_from = src_from.read(1).astype(np.int32)
        arr_to = src_to.read(1).astype(np.int32)
        profile = src_from.profile.copy()
        profile.update(dtype=np.int32, nodata=0, compress="lzw")

    # 异常值处理
    arr_from = np.where((arr_from < 1) | (arr_from > 44), 0, arr_from)
    arr_to = np.where((arr_to < 1) | (arr_to > 44), 0, arr_to)

    # --------------------
    # 小类 TID 向量化映射
    # --------------------
    arr_from_small = np.vectorize(lambda x: small_id_map.get(x, 0))(arr_from)
    arr_to_small = np.vectorize(lambda x: small_id_map.get(x, 0))(arr_to)
    minor_tid_map = np.zeros_like(arr_from_small, dtype=np.int32)
    mask_valid = (arr_from_small>0)&(arr_to_small>0)
    idx = np.ravel_multi_index((arr_from_small[mask_valid]-1, arr_to_small[mask_valid]-1),
                               (len(small_id_map), len(small_id_map)))
    minor_tid_map[mask_valid] = np.array([x[0]-1 for x in minor_transition_list])[idx]+1

    # --------------------
    # 大类 TID 向量化映射
    # --------------------
    arr_from_large = np.vectorize(lambda x: large_id_map.get(small_to_large.get(x, ""),0))(arr_from)
    arr_to_large = np.vectorize(lambda x: large_id_map.get(small_to_large.get(x, ""),0))(arr_to)
    major_tid_map = np.zeros_like(arr_from_large, dtype=np.int32)
    mask_valid_large = (arr_from_large>0)&(arr_to_large>0)
    idx_large = np.ravel_multi_index((arr_from_large[mask_valid_large]-1, arr_to_large[mask_valid_large]-1),
                                     (len(large_id_map), len(large_id_map)))
    major_tid_map[mask_valid_large] = np.array([x[0] for x in major_transition_list])[idx_large]

    # 保存栅格
    minor_out = os.path.join(output_dir, f"minor_transition_{period_name}.tif")
    major_out = os.path.join(output_dir, f"major_transition_{period_name}.tif")
    with rasterio.open(minor_out, "w", **profile) as dst:
        dst.write(minor_tid_map, 1)
    with rasterio.open(major_out, "w", **profile) as dst:
        dst.write(major_tid_map, 1)

    print(f"✅ Finished {period_name}, results saved.")

# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    files = {
        "09_15": (os.path.join(input_dir, "CHA0915_09.tif"),
                  os.path.join(input_dir, "CHA0915_15.tif")),
        "15_18": (os.path.join(input_dir, "CHA1518_15.tif"),
                  os.path.join(input_dir, "CHA1518_18.tif")),
        "09_18": (os.path.join(input_dir, "CHA0915_09.tif"),
                  os.path.join(input_dir, "CHA1518_18.tif"))
    }

    for period, (f_from, f_to) in files.items():
        compute_changes(f_from, f_to, period)
