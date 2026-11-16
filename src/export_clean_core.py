# %% [markdown]
# # HMDA 2024: xuất bảng sạch (clean core) ra Parquet và đối chiếu từng dòng giữa Pandas và Spark
#
# Hai notebook chính đều dựng bảng sạch rồi dùng ngay trong phiên chạy; Spark ghi nó ra Parquet ở thư mục tạm.
# Notebook này giữ bảng đó lại làm một sản phẩm dữ liệu có thể tải về:
# 1. Chạy đúng phần 1–2 của `pandas_pipeline.py` (đọc, lọc mẫu core, tạo biến) và giữ bảng Pandas.
# 2. Chạy đúng phần 1–2 của `spark_pipeline.py`; Spark ghi bảng sạch ra Parquet, chia thư mục theo `state_code`.
# 3. Đọc lại Parquet do Spark ghi và so **từng dòng, từng cột** với bảng Pandas (khóa `row_id`).
# 4. Chép bảng sạch vào output (`clean_core/`), kèm từ điển cột và manifest số dòng theo bang.
#
# Trên Kaggle: các ô phần 1–2 của hai pipeline được chép ngay sau ô này. Ngoài Kaggle:
# `HMDA_INPUT=... python export_clean_core.py` (script tự chạy các ô đó).

# %%
import os
import sys

IN_NOTEBOOK = "ipykernel" in sys.modules
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."


def _cells_until(fname, stop_marker):
    """Các ô code của một pipeline, từ đầu đến trước tiêu đề `stop_marker`."""
    src = open(os.path.join(_HERE, fname), encoding="utf-8").read()
    src = src[:src.index(stop_marker)]
    return [c_ for c_ in src.split("# %%") if c_.strip() and not c_.startswith(" [markdown]")]


if not IN_NOTEBOOK:
    for _c in _cells_until("pandas_pipeline.py", "# ## 3. Kiểm tra với expected.yaml"):
        exec(compile(_c, "pandas_pipeline.py", "exec"))

# %%
# giữ bảng Pandas, giải phóng phần còn lại trước khi khởi động Spark
import gc

PD_CORE = c.set_index("row_id").sort_index()
PD_WORK, PD_RES = WORK, RES
for _n in ("c", "raw", "b"):
    globals().pop(_n, None)
gc.collect()
print(f"bảng Pandas: {len(PD_CORE):,} dòng, {PD_CORE.shape[1] + 1} cột")

# %%
if not IN_NOTEBOOK:
    for _c in _cells_until("spark_pipeline.py", "# ## 3. Kiểm tra với expected.yaml"):
        exec(compile(_c, "spark_pipeline.py", "exec"))

# %% [markdown]
# ## Đối chiếu Parquet do Spark ghi với bảng Pandas, rồi xuất

# %%
import json
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads

t0 = time.time()
PART = R["source"]["partition_clean_by"]
dset = pads.dataset(CLEAN, format="parquet",
                    partitioning=pads.partitioning(pa.schema([(PART, pa.string())]), flavor="hive"))
spark_cols = [f_.name for f_ in dset.schema]
ids = dset.to_table(columns=["row_id"]).column("row_id").to_numpy()
order = np.argsort(ids, kind="stable")
assert len(ids) == len(PD_CORE) and np.array_equal(ids[order], PD_CORE.index.to_numpy()), "tập row_id khác nhau"


def same(a, b):
    """So hai cột đã xếp theo row_id: chuỗi và bool phải bằng hẳn; số thực sai số tương đối 1e-12; thiếu = thiếu."""
    a, b = pd.Series(a), pd.Series(b)
    na_a, na_b = a.isna().to_numpy(), b.isna().to_numpy()
    if not np.array_equal(na_a, na_b):
        return False, int((na_a != na_b).sum())
    m = ~na_a
    av, bv = a[m].to_numpy(), b[m].to_numpy()
    if np.issubdtype(np.asarray(av).dtype, np.floating) or np.issubdtype(np.asarray(bv).dtype, np.floating):
        av, bv = av.astype(float), bv.astype(float)
        bad = ~np.isclose(av, bv, rtol=1e-12, atol=0)
    else:
        bad = np.asarray(av.astype(str) != bv.astype(str))
    return not bad.any(), int(bad.sum())


rows = []
for col in spark_cols:
    if col == "row_id":
        continue
    sv = dset.to_table(columns=[col]).column(col).to_pandas().to_numpy()[order]
    if col not in PD_CORE.columns:
        rows.append({"column": col, "in_pandas": False, "agree": None, "n_diff": None})
        continue
    ok, nd = same(sv, PD_CORE[col].to_numpy())
    rows.append({"column": col, "in_pandas": True, "agree": ok, "n_diff": nd})
for col in PD_CORE.columns:
    if col not in spark_cols:
        rows.append({"column": col, "in_pandas": True, "in_spark": False, "agree": None, "n_diff": None})
CMP = pd.DataFrame(rows)
CMP["in_spark"] = CMP.get("in_spark", pd.Series(dtype=object)).fillna(True).astype(bool)
both_ = CMP["in_pandas"] & CMP["in_spark"]
print(CMP.to_string(index=False))
n_cmp = int(both_.sum())
assert CMP.loc[both_, "agree"].all(), "Pandas và Spark lệch ở bảng sạch"
print(f"{len(ids):,} dòng x {n_cmp} cột trùng khớp | {time.time() - t0:.0f}s")

# %%
OUTROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("HMDA_OUT", "output")
DST = os.path.join(OUTROOT, "clean_core")
META = os.path.join(OUTROOT, "results", "clean")
shutil.rmtree(DST, ignore_errors=True)
shutil.copytree(CLEAN, DST, ignore=shutil.ignore_patterns("*.crc", "_SUCCESS"))
for d_ in ("results/pandas", "results/spark", "figures"):   # bảng phụ mà phần 1–2 của hai pipeline đã ghi; ở đây không cần
    shutil.rmtree(os.path.join(OUTROOT, d_), ignore_errors=True)
os.makedirs(META, exist_ok=True)

out = pads.dataset(DST, format="parquet", partitioning=pads.partitioning(pa.schema([(PART, pa.string())]), flavor="hive"))
tab_counts = out.to_table(columns=[PART]).column(PART).to_pandas().value_counts().sort_index()
files = [os.path.join(r_, f_) for r_, _, fs in os.walk(DST) for f_ in fs]
manifest = pd.DataFrame({PART: tab_counts.index, "rows": tab_counts.to_numpy()})
manifest.to_csv(f"{META}/clean_core_manifest.csv", index=False)
dic = pd.DataFrame({"column": [f_.name for f_ in out.schema], "type": [str(f_.type) for f_ in out.schema]})
nulls = {c_: int(out.to_table(columns=[c_]).column(c_).null_count) for c_ in dic["column"]}
dic["nulls"] = dic["column"].map(nulls)
dic["source"] = np.where(dic["column"].isin(RAW_KEEP + [PART]), "raw HMDA field, kept as text", "derived (rules.yaml 1.1)")
dic.to_csv(f"{META}/clean_core_columns.csv", index=False)
CMP.to_csv(f"{META}/clean_core_pandas_vs_spark.csv", index=False)
summary = {"rows": int(tab_counts.sum()), "columns": len(dic), "partitions": int(len(tab_counts)),
           "files": len(files), "size_mb": round(sum(os.path.getsize(f_) for f_ in files) / 1e6, 1),
           "columns_compared_with_pandas": n_cmp, "all_agree": bool(CMP.loc[both_, "agree"].all()),
           "rules_version": R["version"]}
json.dump(summary, open(f"{META}/clean_core_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
assert summary["rows"] == len(PD_CORE)
