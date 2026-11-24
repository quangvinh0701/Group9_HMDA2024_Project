# %% [markdown]
# # HMDA 2024: kiểm tra bổ sung cho phần data engineering
#
# Notebook này không thay đổi kết quả phân tích nào; nó ghi thêm các con số về chính dữ liệu và bước chuẩn bị:
#
# 1. **Dấu vân tay của file gốc**: SHA-256 của CSV (và Transmittal Sheet), để người khác biết mình dùng đúng file.
# 2. **Chi phí bước chuyển CSV sang Parquet**: chạy lại `convert()` của `src/convert_to_parquet.py` vào một thư mục
#    tạm, đo thời gian, bộ nhớ cao nhất của tiến trình và kích thước file ra.
# 3. **Bản ghi trùng còn lại trong mẫu**: 26.755 bản ghi trùng trên 99 trường được tính trên toàn file; ở đây đếm
#    lại cách đó (cùng hàm băm như pipeline Pandas) rồi xem bao nhiêu bản trùng rơi vào core và usable.
# 4. **DTI thiếu tập trung ở lender nào**: trong các hồ sơ đã có quyết định, DTI thiếu (NA, Exempt, mã khác) chia
#    theo LEI, để biết phần thiếu là do cách lender báo cáo hay do người vay.
# 5. **Giá trị bất hợp lý của các trường chưa kiểm**: `loan_amount`, `interest_rate`, `loan_term` trong core. Chỉ
#    đếm và báo cáo, không loại bản ghi nào.
#
# Định nghĩa mẫu đọc từ `rules.yaml` (bản 1.1), giống hai pipeline chính.
#
# Usage: `python data_checks.py` (tự tìm file trong /kaggle/input hoặc thư mục HMDA_INPUT).

# %%
import gc
import glob
import hashlib
import json
import os
import sys
import threading
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq
import yaml

try:
    import psutil
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil"], check=True)
    import psutil


def find(pattern, roots):
    for r in roots:
        hits = sorted(h for h in glob.glob(os.path.join(r, "**", pattern), recursive=True) if os.path.isfile(h))
        if hits:
            return hits[0]
    return None


ROOTS = [os.environ["HMDA_INPUT"]] if os.environ.get("HMDA_INPUT") else ["/kaggle/input", "."]
RAW = find("hmda_2024_raw.parquet", ROOTS)
RULES = find("rules.yaml", ROOTS)
CSV = find("*public_lar_one_year*.csv", ROOTS)
ZIP = None if CSV else find("*public_lar_one_year*.zip", ROOTS)
TS = find("*public_ts_one_year*", ROOTS) or find("*_ts_*.txt", ROOTS) or find("*_ts_*.csv", ROOTS)
OUTROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("HMDA_OUT", "output")
OUT = f"{OUTROOT}/results/data_checks"
TMP = f"{OUTROOT}/tmp_convert" if os.path.isdir("/kaggle/working") else "/tmp/tmp_convert"
os.makedirs(OUT, exist_ok=True)
R = yaml.safe_load(open(RULES, encoding="utf-8"))
print("raw parquet:", RAW, "\nrules:", RULES, R["version"], "\ncsv:", CSV or ZIP, "\nts:", TS, "\nout:", OUT)
K = {"rules_version": R["version"]}

# %% [markdown]
# ## 1. SHA-256 của file gốc

# %%
def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


t0 = time.time()
SRC = CSV or ZIP
K["lar_file"] = os.path.basename(SRC)
K["lar_bytes"] = os.path.getsize(SRC)
K["lar_sha256"] = sha256(SRC)
if TS:
    K["ts_file"], K["ts_bytes"], K["ts_sha256"] = os.path.basename(TS), os.path.getsize(TS), sha256(TS)
K["sha_seconds"] = round(time.time() - t0, 1)
print(json.dumps({k: v for k, v in K.items() if k != "rules_version"}, indent=1))

# %% [markdown]
# ## 2. Chạy lại bước chuyển CSV sang Parquet và đo chi phí
# Mã của `read_header()` và `convert()` chép nguyên từ `src/convert_to_parquet.py` (giống notebook `HMDA_convert`). Một luồng phụ đọc
# RSS của tiến trình mỗi 0,1 giây để lấy bộ nhớ cao nhất.

# %%
def read_header(path: str) -> tuple[list[str], str]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().rstrip("\r\n")
    delimiter = "|" if first.count("|") > first.count(",") else ","
    return [c.strip().strip('"') for c in first.split(delimiter)], delimiter


def convert(src: str, dst: str, block_mb: int = 256) -> None:
    columns, delimiter = read_header(src)

    read_opts = pv.ReadOptions(block_size=block_mb << 20)
    parse_opts = pv.ParseOptions(delimiter=delimiter)
    # By default Arrow (and Pandas) turn "NA", "N/A", "NaN", "null" and "" into nulls.
    # In HMDA, "NA" means "not applicable", which is information, so nothing is nulled here.
    conv_opts = pv.ConvertOptions(
        column_types={c: pa.string() for c in columns},
        null_values=[],
        strings_can_be_null=False,
        quoted_strings_can_be_null=False,
    )

    reader = pv.open_csv(src, read_options=read_opts,
                         parse_options=parse_opts, convert_options=conv_opts)

    writer = None
    n_rows = 0
    t0 = time.perf_counter()
    for batch in reader:
        row_id = pa.array(range(n_rows, n_rows + batch.num_rows), type=pa.int64())
        table = pa.Table.from_batches([batch]).append_column("row_id", row_id)
        if writer is None:
            writer = pq.ParquetWriter(dst, table.schema, compression="snappy")
        writer.write_table(table)
        n_rows += batch.num_rows
        print(f"\r{n_rows:,} rows written", end="", flush=True)
    if writer is not None:
        writer.close()

    meta = pq.ParquetFile(dst).metadata
    print(f"\nDone in {time.perf_counter() - t0:.1f}s: {meta.num_rows:,} rows, "
          f"{meta.num_columns} columns (incl. row_id), {meta.num_row_groups} row groups")


csv_path = CSV
if not csv_path:
    with zipfile.ZipFile(ZIP) as z_:
        nm = [n for n in z_.namelist() if n.lower().endswith(".csv")][0]
        csv_path = z_.extract(nm, "/tmp/lar")
os.makedirs(TMP, exist_ok=True)
dst = os.path.join(TMP, "hmda_2024_raw_rerun.parquet")

proc = psutil.Process()
base_rss = proc.memory_info().rss
peak = {"rss": base_rss}
stop = threading.Event()


def sampler():
    while not stop.is_set():
        peak["rss"] = max(peak["rss"], proc.memory_info().rss)
        time.sleep(0.1)


th = threading.Thread(target=sampler, daemon=True)
th.start()
t0 = time.perf_counter()
convert(csv_path, dst)
n_conv = pq.ParquetFile(dst).metadata.num_rows
K["convert_seconds"] = round(time.perf_counter() - t0, 1)
stop.set(); th.join()
md = pq.ParquetFile(dst).metadata
md_in = pq.ParquetFile(RAW).metadata
K.update({
    "convert_rows": n_conv, "convert_columns": md.num_columns, "convert_row_groups": md.num_row_groups,
    "convert_parquet_bytes": os.path.getsize(dst), "input_parquet_bytes": os.path.getsize(RAW),
    "convert_peak_rss_gib": round(peak["rss"] / 2**30, 2), "convert_rss_before_gib": round(base_rss / 2**30, 2),
    "csv_bytes": os.path.getsize(csv_path), "cpu_count": os.cpu_count(),
    "ram_gib": round(psutil.virtual_memory().total / 2**30, 1), "pyarrow": pa.__version__,
    "same_rows_and_columns_as_input": md.num_rows == md_in.num_rows and md.num_columns == md_in.num_columns,
})
K["csv_to_parquet_ratio"] = round(K["csv_bytes"] / K["convert_parquet_bytes"], 2)
assert n_conv == md_in.num_rows and md.num_columns == 100, (n_conv, md.num_columns)
os.remove(dst)
print({k: K[k] for k in K if k.startswith(("convert", "input", "csv", "cpu", "ram", "pyarrow", "same"))})

# %% [markdown]
# ## 3. Bản ghi trùng trên 99 trường: toàn file, core và usable
# Băm mỗi dòng bằng `pd.util.hash_pandas_object` trên 99 trường gốc (bỏ `row_id`), như pipeline Pandas.
# "Số bản trùng" của một tập = số dòng trừ số mẫu khác nhau trong tập đó.

# %%
pf = pq.ParquetFile(RAW)
ALLCOLS = [c for c in pf.schema_arrow.names if c != "row_id"]
assert len(ALLCOLS) == 99
FILT = R["samples"]["core_filters"]
DTI = R["variables"]["debt_to_income_ratio"]
DTI_KEYS = list(DTI["ordinal_map"].keys())
USE = R["samples"]["usable"]
FLAGCOLS = ["row_id", "lei", "action_taken", "debt_to_income_ratio", "combined_loan_to_value_ratio", "income",
            "loan_amount", "interest_rate", "loan_term"]

t0 = time.time()
hashes, core_parts = [], []
for batch in pf.iter_batches(batch_size=200_000):
    tb = pa.Table.from_batches([batch])
    hashes.append(pd.util.hash_pandas_object(tb.select(ALLCOLS).to_pandas(), index=False).to_numpy())
    for f in FILT:
        col = tb[f["column"]]
        mask = pc.is_in(col, pa.array(f["in"])) if "in" in f else pc.equal(col, f["equals"])
        tb = tb.filter(mask)
    core_parts.append(tb.select(FLAGCOLS))
H = np.concatenate(hashes); del hashes
core = pa.concat_tables(core_parts).to_pandas(); del core_parts
gc.collect()
core["row_id"] = core["row_id"].astype("int64")
print(f"băm {len(H):,} dòng, core {len(core):,} dòng | {time.time() - t0:.0f}s")


def n_dup(h):
    return int(len(h) - len(np.unique(h)))


# usable theo rules.yaml: decisioned, DTI thuộc 19 mức, 0 < CLTV <= 250, income > 0
dti_norm = core["debt_to_income_ratio"].astype(str)
for ch in DTI["normalize"]["remove"]:
    dti_norm = dti_norm.str.replace(ch, "", regex=False)
core["dti_valid"] = dti_norm.isin(DTI_KEYS)
core["dti_raw"] = core["debt_to_income_ratio"].astype(str)
cltv = pd.to_numeric(core["combined_loan_to_value_ratio"], errors="coerce")
inc = pd.to_numeric(core["income"], errors="coerce")
dec_codes = R["samples"]["decisioned"]["action_taken_in"]
core["decisioned"] = core["action_taken"].astype(str).isin(dec_codes)
core["usable"] = (core["decisioned"] & core["dti_valid"] & (cltv > USE["cltv_range"]["gt"])
                  & (cltv <= USE["cltv_range"]["le"]) & (inc > USE["income_gt"]))
core["h"] = H[core["row_id"].to_numpy()]

all_u, all_c = np.unique(H, return_counts=True)
twin = pd.Series(all_c, index=all_u)
K.update({
    "records": int(len(H)), "dup_all": n_dup(H), "dup_all_patterns": int((all_c > 1).sum()),
    "core_rows": int(len(core)), "dup_core": n_dup(core["h"].to_numpy()),
    "core_rows_with_twin_anywhere": int((twin.reindex(core["h"].to_numpy()).to_numpy() > 1).sum()),
    "decisioned_rows": int(core["decisioned"].sum()),
    "dup_decisioned": n_dup(core.loc[core["decisioned"], "h"].to_numpy()),
    "usable_rows": int(core["usable"].sum()), "dup_usable": n_dup(core.loc[core["usable"], "h"].to_numpy()),
})
d_use = core.loc[core["usable"]]
dup_mask = d_use["h"].duplicated(keep=False)
K["usable_rows_in_dup_groups"] = int(dup_mask.sum())
K["usable_dup_action_mix"] = d_use.loc[dup_mask, "action_taken"].astype(str).value_counts().to_dict()
# số khớp kỳ vọng đã có trong báo cáo
FULL = K["records"] == 12259199            # file thật (không phải file tổng hợp để thử)
if FULL:
    assert K["dup_all"] == 26755, K["dup_all"]
    assert K["core_rows"] == 5775325 and K["decisioned_rows"] == 4514505 and K["usable_rows"] == 4100051
print({k: K[k] for k in K if k.startswith(("dup", "core", "decisioned", "usable", "records"))})

# %% [markdown]
# ## 4. DTI thiếu trong hồ sơ đã có quyết định, theo lender
# Thiếu = giá trị DTI không thuộc 19 mức công bố (thường là `NA` hoặc `Exempt`).

# %%
dec = core.loc[core["decisioned"]].copy()
dec["miss"] = ~dec["dti_valid"]
dec["exempt"] = dec["dti_raw"].eq("Exempt")
dec["na"] = dec["dti_raw"].eq("NA")
L = dec.groupby("lei", observed=True).agg(n=("miss", "size"), miss=("miss", "sum"), exempt=("exempt", "sum"),
                                          na=("na", "sum"))
L = L.sort_values("miss", ascending=False)
tot_miss, tot_n = int(L["miss"].sum()), int(L["n"].sum())
cum = L["miss"].cumsum() / tot_miss
L["miss_rate"] = L["miss"] / L["n"]
K.update({
    "lenders_all_file": int(pc.count_distinct(pq.read_table(RAW, columns=["lei"])["lei"]).as_py()),
    "lenders_core": int(core["lei"].nunique()), "lenders_decisioned": int(len(L)),
    "dec_dti_missing": tot_miss, "dec_dti_missing_rate": tot_miss / tot_n,
    "dec_dti_exempt": int(L["exempt"].sum()), "dec_dti_na": int(L["na"].sum()),
    "dec_dti_other": tot_miss - int(L["exempt"].sum()) - int(L["na"].sum()),
    "lenders_with_any_missing": int((L["miss"] > 0).sum()),
    "top10_share_of_missing": float(L["miss"].head(10).sum() / tot_miss),
    "top10_share_of_decisioned": float(L["n"].head(10).sum() / tot_n),
    "lenders_for_50pct_missing": int((cum < 0.5).sum() + 1), "lenders_for_80pct_missing": int((cum < 0.8).sum() + 1),
    "lenders_all_missing": int((L["miss_rate"] == 1).sum()),
    "share_missing_from_lenders_ge90": float(L.loc[L["miss_rate"] >= 0.9, "miss"].sum() / tot_miss),
    "share_decisioned_from_lenders_ge90": float(L.loc[L["miss_rate"] >= 0.9, "n"].sum() / tot_n),
    "lenders_ge90": int((L["miss_rate"] >= 0.9).sum()),
    "median_lender_missing_rate": float(L["miss_rate"].median()),
})
# không có DTI vì lender được miễn (Exempt) hay vì NA ở lender vẫn báo DTI cho phần lớn hồ sơ?
rest = L.loc[L["miss_rate"] < 0.9]
K["missing_at_lenders_lt90"] = int(rest["miss"].sum())
K["missing_rate_at_lenders_lt90"] = float(rest["miss"].sum() / rest["n"].sum())
if FULL:
    assert tot_miss == 251176, tot_miss      # khớp Table 1 của báo cáo
L.head(25).to_csv(f"{OUT}/dti_missing_top25_lenders.csv")
print({k: K[k] for k in K if "lender" in k or "missing" in k or k.startswith("dec_")})

# %% [markdown]
# ## 5. Giá trị bất hợp lý của các trường chưa kiểm (core)
# Chỉ đếm; không loại bản ghi nào (các phân tích dùng median, band hoặc không dùng các trường này).

# %%
amt = pd.to_numeric(core["loan_amount"], errors="coerce")
ir = pd.to_numeric(core["interest_rate"], errors="coerce")
term = pd.to_numeric(core["loan_term"], errors="coerce")
orig = core["action_taken"].astype(str).eq("1")
K.update({
    "amt_nonnumeric": int(amt.isna().sum()), "amt_le0": int((amt <= 0).sum()), "amt_max": float(amt.max()),
    "amt_gt5m": int((amt > 5e6).sum()), "amt_gt10m": int((amt > 1e7).sum()),
    "orig_rows": int(orig.sum()), "ir_orig_nonnumeric": int(ir[orig].isna().sum()),
    "ir_orig_le0": int((ir[orig] <= 0).sum()), "ir_orig_gt15": int((ir[orig] > 15).sum()),
    "ir_orig_max": float(ir[orig].max()),
    "term_nonnumeric": int(term.isna().sum()), "term_le0": int((term <= 0).sum()),
    "term_gt480": int((term > 480).sum()), "term_max": float(term.max()),
    "term_360_share": float((term == 360).mean()),
})
print({k: K[k] for k in K if k.startswith(("amt", "ir_", "term", "orig"))})

# %%
K = {k: (v.item() if hasattr(v, "item") else v) for k, v in K.items()}
json.dump(K, open(f"{OUT}/data_checks.json", "w"), indent=1, default=str)
for p in sorted(glob.glob(f"{OUT}/*")):
    b = open(p, "rb").read()
    print("@@@@ FILE", os.path.relpath(p, OUTROOT), hashlib.md5(b).hexdigest(), len(b))
    print(b.decode("utf-8"))
print("@@@@ END")
