import pandas as pd

# ===============================
# 输入路径
# ===============================
clc_file = r"E:\Project-yqr\CLC_new\提取newclc.csv"
SOC_2009_PATH = r"E:\Project-yqr\组会\data\SOC\SOC_2009_filled.csv"
SOC_2015_PATH = r"E:\Project-yqr\组会\data\SOC\SOC_2015_filled.csv"
SOC_2018_PATH = r"E:\Project-yqr\组会\data\SOC\SOC_2018_filled.csv"

# 输出路径（避免覆盖原始数据）
SOC_2009_OUT = SOC_2009_PATH.replace(".csv", "_CLC_mapped.csv")
SOC_2015_OUT = SOC_2015_PATH.replace(".csv", "_CLC_mapped.csv")
SOC_2018_OUT = SOC_2018_PATH.replace(".csv", "_CLC_mapped.csv")

# ===============================
# 1. 建立 CLC 映射关系 (111–523 → 1–44)
# ===============================
clc_mapping = {
    111: 1, 112: 2, 121: 3, 122: 4, 123: 5, 124: 6,
    131: 7, 132: 8, 133: 9,
    141: 10, 142: 11,
    211: 12, 212: 13, 213: 14,
    221: 15, 222: 16, 223: 17,
    231: 18,
    241: 19, 242: 20, 243: 21, 244: 22,
    311: 23, 312: 24, 313: 25,
    321: 26, 322: 27, 323: 28, 324: 29,
    331: 30, 332: 31, 333: 32, 334: 33, 335: 34,
    411: 35, 412: 36,
    421: 37, 422: 38, 423: 39,
    511: 40, 512: 41,
    521: 42, 522: 43, 523: 44,
    999: 999   # nodata 保留
}

# ===============================
# 2. 读取提取的 CLC 数据
# ===============================
clc_df = pd.read_csv(clc_file)

# 把 111–523 映射成 1–44
for year in [2009, 2015, 2018]:
    col = f"CLCACC_{year}"
    clc_df[col] = clc_df[col].map(clc_mapping).fillna(999).astype(int)

# ===============================
# 3. 定义一个函数用于替换
# ===============================
def replace_clc(soc_path, year, out_path):
    soc_df = pd.read_csv(soc_path)
    merge_df = soc_df.merge(
        clc_df[["ID", f"CLCACC_{year}"]],
        on="ID",
        how="left"
    )
    # 用转换后的值替换原来的 CLC 列
    merge_df["CLC"] = merge_df[f"CLCACC_{year}"]
    merge_df.drop(columns=[f"CLCACC_{year}"], inplace=True)

    merge_df.to_csv(out_path, index=False)
    print(f"✅ 已保存: {out_path}")

# ===============================
# 4. 分别处理 2009 / 2015 / 2018
# ===============================
replace_clc(SOC_2009_PATH, 2009, SOC_2009_OUT)
replace_clc(SOC_2015_PATH, 2015, SOC_2015_OUT)
replace_clc(SOC_2018_PATH, 2018, SOC_2018_OUT)
