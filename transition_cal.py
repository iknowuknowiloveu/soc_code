import os
import numpy as np
import rasterio
from rasterio.windows import Window
import logging

# -------------------------
# 日志配置
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# 配置
# -------------------------
input_dir = r"G:\Project-yqr\CLC_new"
output_dir = r"G:\Project-yqr\CLC_new\change_by_ACC"
tile_size = 5000  # 每块大小，可根据内存调节
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# ACC 值和中文名映射
# -------------------------
acc_to_name = {
    111:"Continuous urban fabric", 112:"Discontinuous urban fabric", 121:"Industrial or commercial units",
    122:"Road and rail networks and associated land", 123:"Port areas", 124:"Airports", 131:"Mineral extraction sites",
    132:"Dump sites", 133:"Construction sites", 141:"Green urban areas", 142:"Sport and leisure facilities",
    211:"Non irrigated arable land", 212:"Permanently irrigated land", 213:"Rice fields",
    221:"Vineyards", 222:"Fruit trees and berry plantations", 223:"Olive groves",
    231:"Pastures", 241:"Annual crops associated with permanent crops", 242:"Complex cultivation patterns",
    243:"Land principally occupied by agriculture, with significant areas of natural vegetation",
    244:"Agro forestry areas", 311:"Broadleaved forest", 312:"Coniferous forest", 313:"Mixed forest",
    321:"Natural grasslands", 322:"Moors and heathland", 323:"Sclerophyllous vegetation",
    324:"Transitional woodland shrub", 331:"Beaches, dunes, sands", 332:"Bare rocks", 333:"Sparsely vegetated areas",
    334:"Burnt areas", 335:"Glaciers and perpetual snow", 411:"Inland marshes", 412:"Peat bogs",
    421:"Salt marshes", 422:"Salines", 423:"Intertidal flats", 511:"Water courses", 512:"Water bodies",
    521:"Coastal lagoons", 522:"Estuaries", 523:"Sea and ocean", 999:"Nodata"
}

acc_values = sorted([k for k in acc_to_name.keys() if k != 999])

# -------------------------
# 原始 ACC -> 连续编号映射
# -------------------------
original_to_continuous = {v:i+1 for i,v in enumerate(acc_values)}
continuous_to_name = {i+1:acc_to_name[v] for i,v in enumerate(acc_values)}

# 大类提取函数
def get_major(acc_val):
    if acc_val == 999:  # Nodata
        return 0
    return acc_val // 100  # 百位表示大类

# -------------------------
# 保存文字映射表
# -------------------------
def save_transition_tables():
    # 小类
    minor_file = os.path.join(output_dir, "minor_transition_types.txt")
    tid = 1
    with open(minor_file, 'w', encoding='utf-8') as f:
        for from_id in acc_values:
            for to_id in acc_values:
                f.write(f"ID {tid}: {acc_to_name[from_id]} → {acc_to_name[to_id]}\n")
                tid += 1
    logger.info(f"📝 已保存小类转换映射表: {minor_file}")

    # 大类
    major_file = os.path.join(output_dir, "major_transition_types.txt")
    major_ids = sorted(set(get_major(v) for v in acc_values))
    tid = 1
    with open(major_file, 'w', encoding='utf-8') as f:
        for from_cat in major_ids:
            for to_cat in major_ids:
                f.write(f"ID {tid}: Class {from_cat} → Class {to_cat}\n")
                tid += 1
    logger.info(f"📝 已保存大类转换映射表: {major_file}")

# -------------------------
# tile-based 处理
# -------------------------
def process_tile(from_tile, to_tile):
    # ACC 原始值 -> 连续编号
    from_cont = np.vectorize(lambda x: original_to_continuous.get(x, 0))(from_tile)
    to_cont = np.vectorize(lambda x: original_to_continuous.get(x, 0))(to_tile)

    # 小类 TID
    n = len(acc_values)
    minor_lookup = np.zeros((n+2, n+2), dtype=np.float32)
    tid = 1
    for i in range(1,n+1):
        for j in range(1,n+1):
            minor_lookup[i,j] = tid
            tid += 1
    minor_tile = minor_lookup[from_cont, to_cont]

    # 大类 TID
    major_ids = sorted(set(get_major(v) for v in acc_values))
    major_map = {v:i+1 for i,v in enumerate(major_ids)}
    from_major = np.vectorize(lambda x: major_map.get(get_major(x),0))(from_tile)
    to_major = np.vectorize(lambda x: major_map.get(get_major(x),0))(to_tile)
    m = len(major_ids)
    major_lookup = np.zeros((m+2, m+2), dtype=np.float32)
    tid = 1
    for i in range(1,m+1):
        for j in range(1,m+1):
            major_lookup[i,j] = tid
            tid += 1
    major_tile = major_lookup[from_major, to_major]

    # Nodata 处理为 np.nan
    minor_tile[(from_tile==999)|(to_tile==999)|(from_cont==0)|(to_cont==0)] = np.nan
    major_tile[(from_tile==999)|(to_tile==999)|(from_major==0)|(to_major==0)] = np.nan

    return major_tile, minor_tile

# -------------------------
# 分块计算
# -------------------------
def compute_transition(from_path, to_path, major_out, minor_out):
    with rasterio.open(from_path) as src:
        profile = src.profile.copy()
        rows, cols = profile['height'], profile['width']
        profile.update(dtype=rasterio.float32, nodata=np.nan)

        with rasterio.open(major_out,'w',**profile) as major_dst, \
             rasterio.open(minor_out,'w',**profile) as minor_dst:
            for i in range(0, rows, tile_size):
                for j in range(0, cols, tile_size):
                    w = Window(j, i, min(tile_size, cols-j), min(tile_size, rows-i))
                    from_tile = src.read(1, window=w).astype(np.int32)
                    to_tile = rasterio.open(to_path).read(1, window=w).astype(np.int32)
                    major_tile, minor_tile = process_tile(from_tile, to_tile)
                    major_dst.write(major_tile, 1, window=w)
                    minor_dst.write(minor_tile, 1, window=w)

# -------------------------
# 主程序
# -------------------------
def main():
    save_transition_tables()
    periods = [(2009,2015),(2015,2018),(2009,2018)]
    for from_year, to_year in periods:
        logger.info(f"处理 {from_year} → {to_year}")
        from_path = os.path.join(input_dir, f"CLCACC_{from_year}.tif")
        to_path = os.path.join(input_dir, f"CLCACC_{to_year}.tif")
        major_out = os.path.join(output_dir, f"major_transition_{from_year}_to_{to_year}.tif")
        minor_out = os.path.join(output_dir, f"minor_transition_{from_year}_to_{to_year}.tif")
        compute_transition(from_path, to_path, major_out, minor_out)
        logger.info(f"✅ 完成 {from_year} → {to_year}")

if __name__ == "__main__":
    main()
