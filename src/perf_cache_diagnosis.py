# %% [markdown]
# # HMDA 2024: vì sao cache làm Spark chậm hơn ở kích thước lớn?
#
# Thí nghiệm hiệu năng chính cho thấy từ ×4 trở lên, chạy trên bảng đã cache lại chậm hơn đọc thẳng Parquet.
# Notebook này kiểm hai giả thuyết trên đúng bảng 15 cột đó:
# 1. **Cột:** đọc Parquet chỉ lấy các cột truy vấn cần (column pruning), còn cache giữ cả 15 cột.
#    Kiểm bằng cách cache riêng các cột W1 dùng.
# 2. **Bộ nhớ:** bảng cache không vừa vùng storage memory của Spark nên bị đẩy xuống đĩa.
#    Kiểm bằng thông tin lưu trữ của từng RDD đã cache (kích thước trong RAM, trên đĩa, số partition đã cache).
#
# Dữ liệu dựng bằng đúng hai ô đầu của `perf_pipeline.py` (trên Kaggle: hai ô đó được chép ngay sau ô này).
# Usage (ngoài Kaggle): `HMDA_INPUT=... python perf_cache_diagnosis.py`

# %%
import os
import sys

os.environ.setdefault("PERF_SIZES", "100%,x4,x8,x16")
IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:   # chạy hai ô đầu (cài đặt, dựng dữ liệu) của perf_pipeline.py
    _src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_pipeline.py"), encoding="utf-8").read()
    _cells = [c for c in _src.split("# %%") if not c.startswith(" [markdown]")]
    for _c in _cells[1:3]:
        exec(compile(_c, "perf_pipeline.py", "exec"))

# %% [markdown]
# ## Đo trên Spark

# %%
import statistics
import time

import pandas as pd
from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.master("local[*]").appName("hmda-cache-diag")
         .config("spark.driver.memory", f"{max(2, int(TOTAL_RAM / 1e9 * 0.6))}g")
         .config("spark.sql.shuffle.partitions", str(max(8, 2 * (os.cpu_count() or 2))))
         .config("spark.local.dir", f"{WORK}/spark_local").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
jsc = spark.sparkContext._jsc.sc()
storage_max = None
try:
    ms = jsc.getExecutorMemoryStatus()
    it = ms.values().iterator()
    storage_max = 0
    while it.hasNext():
        storage_max += it.next()._1()
except Exception as e:  # noqa: BLE001
    print("không đọc được bộ nhớ storage:", repr(e)[:100])
print("storage memory tối đa (GB):", None if storage_max is None else round(storage_max / 1e9, 2))

W1_COLS = ["decisioned", "loan_type", "loan_purpose", "y", "loan_amount"]


def w1(df):
    g = df.filter("decisioned").groupBy("loan_type", "loan_purpose").agg(
        F.count(F.lit(1)).alias("n"), F.avg("y").alias("rate"), F.avg("loan_amount").alias("mean_amount"))
    return g.collect()


def timed(fn, runs=3):
    fn()   # làm nóng
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def storage():
    mem = disk = cached = total = 0
    for r in jsc.getRDDStorageInfo():
        mem += r.memSize(); disk += r.diskSize(); cached += r.numCachedPartitions(); total += r.numPartitions()
    return mem / 1e9, disk / 1e9, cached, total


rows = []
for sz in SIZES:
    path = DATA[sz]["path"]
    base = spark.read.parquet(path)
    r = {"size": sz, "rows": DATA[sz]["rows"], "parquet_gb": DATA[sz]["disk_mb"] / 1e3}
    r["w1_nocache_s"] = timed(lambda: w1(spark.read.parquet(path)))
    for label, df in [("full", base), ("w1cols", base.select(*W1_COLS))]:
        df = df.cache()
        t = time.perf_counter(); df.count(); r[f"cache_{label}_build_s"] = time.perf_counter() - t
        mem, disk, cached, total = storage()
        r.update({f"cache_{label}_mem_gb": mem, f"cache_{label}_disk_gb": disk,
                  f"cache_{label}_partitions_cached": cached, f"cache_{label}_partitions": total})
        r[f"w1_cache_{label}_s"] = timed(lambda: w1(df))
        df.unpersist(blocking=True)
    rows.append(r)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
spark.stop()

D = pd.DataFrame(rows)
D["storage_max_gb"] = None if storage_max is None else storage_max / 1e9
OUTROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("HMDA_OUT", "output")
os.makedirs(f"{OUTROOT}/results/perf", exist_ok=True)
D.to_csv(f"{OUTROOT}/results/perf/perf_cache_diagnosis.csv", index=False)
pd.set_option("display.width", 250)
print(D.round(3).to_string(index=False))
