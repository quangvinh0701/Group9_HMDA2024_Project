# %% [markdown]
# # HMDA 2024: phân phối của các biến tài chính, tính bằng cả Pandas và PySpark
#
# Bổ sung phần mô tả mà đề bài liệt kê: phân phối loan amount, interest rate, thống kê income, và quan hệ
# loan amount với income. Hai chỉ số tài chính đã định nghĩa sẵn trong `rules.yaml` (mục 6) nhưng chưa dùng,
# payment-to-income và cost share, cũng được tính ở đây.
#
# Đầu vào là bảng sạch `clean_core/` (output của notebook *HMDA clean core export*, đã đối chiếu từng dòng giữa
# Pandas và Spark) cộng bốn cột thô của file Parquet gốc nối theo `row_id`: `interest_rate`, `loan_term`,
# `total_loan_costs`. Mọi khoảng hợp lệ lấy từ `rules.yaml` 1.1.
#
# Mỗi bảng được tính hai lần, một lần bằng Pandas và một lần bằng PySpark, rồi so khớp như phần kiểm tra tương
# đương chính: số đếm phải bằng hẳn, số thực sai số tương đối tối đa 1e-9.
#
# Usage (ngoài Kaggle): `HMDA_INPUT=<thư mục có hmda_2024_raw.parquet, rules.yaml và clean_core/> python descriptive_extra.py`

# %%
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import yaml

ROOTS = [os.environ["HMDA_INPUT"]] if os.environ.get("HMDA_INPUT") else ["/kaggle/input"]


def find(pattern, dirs=False):
    hits = sorted(h for r in ROOTS for h in glob.glob(os.path.join(r, "**", pattern), recursive=True)
                  if os.path.isdir(h) == dirs)
    assert hits, f"không thấy {pattern} trong {ROOTS}"
    return hits[0]


RAW = find("hmda_2024_raw.parquet")
CLEAN = find("clean_core", dirs=True)
R = yaml.safe_load(open(find("rules.yaml"), encoding="utf-8"))
OUTROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("HMDA_OUT", "output")
OUT = f"{OUTROOT}/results/descriptive"
os.makedirs(OUT, exist_ok=True)
print("raw:", RAW, "\nclean:", CLEAN, "\nrules:", R["version"], "\nout:", OUT)

V = R["variables"]
IR, TERM = V["interest_rate"]["valid_range"], V["loan_term"]["valid_range"]
TLC = V["total_loan_costs"]["valid_range"]
TYPES = {"1": "conventional", "2": "FHA", "3": "VA", "4": "USDA"}
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
INC_EDGES = [0, 50, 75, 100, 150, 200, 300]          # nghìn USD; băng cuối là >= 300
INC_LABELS = ["<50", "50-75", "75-100", "100-150", "150-200", "200-300", "300+"]
LA_W, LA_MAX_BIN = 25000, 60                          # loan amount: bin 25 nghìn USD, bin 60 = từ 1,5 triệu trở lên
IR_LO, IR_W, IR_MAX_BIN = 2.0, 0.125, 64              # interest rate: bin 1/8 điểm từ 2%, -1 = dưới 2%, 64 = từ 10%

# Biến, mẫu, đơn vị (giải thích trong báo cáo, Phụ lục C)
VARS = [
    ("loan_amount", "orig", "nghìn USD, khoản vay đã giải ngân"),
    ("income", "decisioned & income > 0", "nghìn USD/năm"),
    ("interest_rate", "orig & interest_rate hợp lệ", "%/năm"),
    ("rate_spread", "pricing", "điểm phần trăm so với APOR"),
    ("cltv", "decisioned & cltv hợp lệ", "%"),
    ("lti", "decisioned & income > 0", "lần thu nhập năm"),
    ("pti", "pricing & lãi suất, kỳ hạn hợp lệ & income > 0", "% thu nhập tháng (chỉ gốc + lãi)"),
    ("cost_share", "pricing & total_loan_costs hợp lệ", "% loan amount"),
]
t0 = time.time()

# %% [markdown]
# ## 1. Pandas

# %%
part = pads.partitioning(pa.schema([("state_code", pa.string())]), flavor="hive")
CC = ["row_id", "type", "orig", "decisioned", "pricing", "income_num", "loan_amount_num", "rs_num", "cltv",
      "cltv_valid", "lti"]
core = pads.dataset(CLEAN, format="parquet", partitioning=part).to_table(columns=CC)
ids = core.column("row_id")
rawt = pq.read_table(RAW, columns=["row_id", "interest_rate", "loan_term", "total_loan_costs"])
rawt = rawt.filter(pc.is_in(rawt.column("row_id"), value_set=ids))
D = core.to_pandas().merge(rawt.to_pandas(), on="row_id", how="left", validate="one_to_one")
del rawt
assert len(D) == len(ids)


def num(s):
    return pd.to_numeric(s.where(s.str.fullmatch(r"-?\d+(\.\d+)?", na=False)), errors="coerce")


D["ir"] = num(D["interest_rate"]); D["term"] = num(D["loan_term"]); D["tlc"] = num(D["total_loan_costs"])
ir_ok = D["ir"].between(IR["ge"], IR["le"])
term_ok = (D["term"] > TERM["gt"]) & (D["term"] <= TERM["le"])
inc_ok = D["income_num"] > 0
r_ = D["ir"] / 1200
pay = np.where(r_ > 0, D["loan_amount_num"] * r_ / (1 - (1 + r_) ** (-D["term"])), D["loan_amount_num"] / D["term"])
D["v_loan_amount"] = np.where(D["orig"], D["loan_amount_num"] / 1000, np.nan)
D["v_income"] = np.where(D["decisioned"] & inc_ok, D["income_num"], np.nan)
D["v_interest_rate"] = np.where(D["orig"] & ir_ok, D["ir"], np.nan)
D["v_rate_spread"] = np.where(D["pricing"], D["rs_num"], np.nan)
D["v_cltv"] = np.where(D["decisioned"] & D["cltv_valid"], D["cltv"], np.nan)
D["v_lti"] = np.where(D["decisioned"] & inc_ok, D["lti"], np.nan)
D["v_pti"] = np.where(D["pricing"] & ir_ok & term_ok & inc_ok, 100 * pay / (1000 * D["income_num"] / 12), np.nan)
D["v_cost_share"] = np.where(D["pricing"] & (D["tlc"] >= TLC["ge"]) & (D["loan_amount_num"] > 0),
                             100 * D["tlc"] / D["loan_amount_num"], np.nan)
D["grp"] = D["type"].astype(str)

# (a) thống kê tóm tắt theo loại vay và chung
rows = []
for v, _, _ in VARS:
    col = f"v_{v}"
    for g, s in [("all", D[col])] + [(t, D.loc[D["grp"] == t, col]) for t in TYPES.values()]:
        s = s.dropna()
        q = s.quantile(QS).to_numpy() if len(s) else [np.nan] * len(QS)
        rows.append({"variable": v, "group": g, "n": len(s), "mean": s.mean(), "sd": s.std(ddof=1),
                     **{f"p{int(round(100 * p)):02d}": x for p, x in zip(QS, q)}})
PD_SUM = pd.DataFrame(rows)

# (b) histogram
la = D.loc[D["orig"], ["grp", "loan_amount_num"]]
la_bin = np.minimum(np.floor(la["loan_amount_num"] / LA_W), LA_MAX_BIN).astype(int)
irv = D.loc[D["orig"] & ir_ok, ["grp", "ir"]]
ir_bin = np.where(irv["ir"] < IR_LO, -1, np.minimum(np.floor((irv["ir"] - IR_LO) / IR_W), IR_MAX_BIN)).astype(int)
PD_HIST = pd.concat([
    la.assign(variable="loan_amount", bin=la_bin).groupby(["variable", "grp", "bin"]).size(),
    irv.assign(variable="interest_rate", bin=ir_bin).groupby(["variable", "grp", "bin"]).size(),
]).rename("n").reset_index().rename(columns={"grp": "group"})

# (c) loan amount và income: băng thu nhập, khoản vay đã giải ngân có income > 0
O = D.loc[D["orig"] & inc_ok, ["grp", "income_num", "loan_amount_num", "lti"]].copy()
O["band"] = np.searchsorted(np.array(INC_EDGES[1:], dtype=float), O["income_num"].to_numpy(), side="right")
rows = []
for g, s in [("all", O)] + [(t, O[O["grp"] == t]) for t in TYPES.values()]:
    for b, sb in s.groupby("band"):
        rows.append({"group": g, "band": INC_LABELS[b], "band_idx": int(b), "n": len(sb),
                     "median_loan": sb["loan_amount_num"].median() / 1000, "median_lti": sb["lti"].median()})
PD_BAND = pd.DataFrame(rows)
rows = []
for g, s in [("all", O)] + [(t, O[O["grp"] == t]) for t in TYPES.values()]:
    rows.append({"group": g, "n": len(s),
                 "spearman": s["loan_amount_num"].rank().corr(s["income_num"].rank()),
                 "pearson_log": np.corrcoef(np.log(s["loan_amount_num"]), np.log(s["income_num"]))[0, 1]})
PD_RHO = pd.DataFrame(rows)
EXCL = {"interest_rate_orig_numeric_out_of_range": int((D["orig"] & D["ir"].notna() & ~ir_ok).sum()),
        "interest_rate_orig_not_numeric": int((D["orig"] & D["ir"].isna()).sum()),
        "orig": int(D["orig"].sum())}
print(f"Pandas xong | {time.time() - t0:.0f}s | {EXCL}")

# %% [markdown]
# ## 2. PySpark (cùng định nghĩa, viết lại bằng DataFrame API)

# %%
from pyspark.sql import SparkSession, Window, functions as F

t1 = time.time()
avail = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1e9)
spark = (SparkSession.builder.master("local[*]").appName("hmda-descriptive")
         .config("spark.driver.memory", f"{max(2, int(avail * 0.5))}g")
         .config("spark.sql.shuffle.partitions", str(max(8, 2 * (os.cpu_count() or 2))))
         .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
         .config("spark.sql.ansi.enabled", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

NUM_RE = r"^-?[0-9]+(\.[0-9]+)?$"


def snum(c):
    return F.when(F.col(c).rlike(NUM_RE), F.col(c).cast("double"))


S = (spark.read.parquet(CLEAN).select(*CC)
     .join(spark.read.parquet(RAW).select("row_id", "interest_rate", "loan_term", "total_loan_costs"), "row_id", "left")
     .withColumn("ir", snum("interest_rate")).withColumn("term", snum("loan_term")).withColumn("tlc", snum("total_loan_costs")))
ir_ok_s = F.col("ir").between(IR["ge"], IR["le"])
term_ok_s = (F.col("term") > TERM["gt"]) & (F.col("term") <= TERM["le"])
inc_ok_s = F.col("income_num") > 0
rr = F.col("ir") / 1200
pay_s = F.when(rr > 0, F.col("loan_amount_num") * rr / (1 - F.pow(1 + rr, -F.col("term")))) \
         .otherwise(F.col("loan_amount_num") / F.col("term"))
S = (S.withColumn("grp", F.col("type"))
     .withColumn("v_loan_amount", F.when(F.col("orig"), F.col("loan_amount_num") / 1000))
     .withColumn("v_income", F.when(F.col("decisioned") & inc_ok_s, F.col("income_num")))
     .withColumn("v_interest_rate", F.when(F.col("orig") & ir_ok_s, F.col("ir")))
     .withColumn("v_rate_spread", F.when(F.col("pricing"), F.col("rs_num")))
     .withColumn("v_cltv", F.when(F.col("decisioned") & F.col("cltv_valid"), F.col("cltv")))
     .withColumn("v_lti", F.when(F.col("decisioned") & inc_ok_s, F.col("lti")))
     .withColumn("v_pti", F.when(F.col("pricing") & ir_ok_s & term_ok_s & inc_ok_s,
                                 100 * pay_s / (1000 * F.col("income_num") / 12)))
     .withColumn("v_cost_share", F.when(F.col("pricing") & (F.col("tlc") >= TLC["ge"]) & (F.col("loan_amount_num") > 0),
                                        100 * F.col("tlc") / F.col("loan_amount_num")))
     .cache())
n_spark = S.count()
assert n_spark == len(D)
QARR = "array(" + ",".join(str(q) for q in QS) + ")"


def summary(col):
    aggs = [F.count(col).alias("n"), F.mean(col).alias("mean"), F.stddev_samp(col).alias("sd"),
            F.expr(f"percentile({col}, {QARR})").alias("q")]
    a = S.agg(*aggs).withColumn("group", F.lit("all"))
    b = S.groupBy(F.col("grp").alias("group")).agg(*aggs)
    return a.unionByName(b).withColumn("variable", F.lit(col[2:]))


sp = None
for v, _, _ in VARS:
    x = summary(f"v_{v}")
    sp = x if sp is None else sp.unionByName(x)
SP_SUM = sp.toPandas()
for i, p in enumerate(QS):
    SP_SUM[f"p{int(round(100 * p)):02d}"] = SP_SUM["q"].map(lambda a, i=i: a[i] if a is not None else np.nan)
SP_SUM = SP_SUM.drop(columns="q")

la_bin_s = F.least(F.floor(F.col("loan_amount_num") / LA_W), F.lit(LA_MAX_BIN)).cast("int")
ir_bin_s = F.when(F.col("ir") < IR_LO, F.lit(-1)).otherwise(
    F.least(F.floor((F.col("ir") - IR_LO) / IR_W), F.lit(IR_MAX_BIN))).cast("int")
SP_HIST = (S.filter("orig").groupBy(F.lit("loan_amount").alias("variable"), F.col("grp").alias("group"), la_bin_s.alias("bin")).count()
           .unionByName(S.filter(F.col("orig") & ir_ok_s).groupBy(F.lit("interest_rate").alias("variable"),
                        F.col("grp").alias("group"), ir_bin_s.alias("bin")).count())
           .withColumnRenamed("count", "n").toPandas())

band_s = sum((F.col("income_num") >= e).cast("int") for e in INC_EDGES[1:])
OS_ = S.filter(F.col("orig") & inc_ok_s).withColumn("band_idx", band_s)
OS_all = OS_.withColumn("group", F.lit("all"))
OS_g = OS_.withColumn("group", F.col("grp"))
OSU = OS_all.unionByName(OS_g).cache()
SP_BAND = (OSU.groupBy("group", "band_idx").agg(F.count("*").alias("n"),
           (F.expr("percentile(loan_amount_num, 0.5)") / 1000).alias("median_loan"),
           F.expr("percentile(lti, 0.5)").alias("median_lti")).toPandas())
SP_BAND["band"] = SP_BAND["band_idx"].map(lambda b: INC_LABELS[b])


def avg_rank(df, col):
    """Hạng trung bình (như pandas rank method='average') trong từng nhóm."""
    c = df.groupBy("group", col).count()
    w = Window.partitionBy("group").orderBy(col).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    c = c.withColumn("cum", F.sum("count").over(w)).withColumn(f"r_{col}", F.col("cum") - (F.col("count") - 1) / 2)
    return c.select("group", col, f"r_{col}")


RK = OSU.join(avg_rank(OSU, "loan_amount_num"), ["group", "loan_amount_num"]).join(avg_rank(OSU, "income_num"), ["group", "income_num"])
SP_RHO = (RK.groupBy("group").agg(F.count("*").alias("n"), F.corr("r_loan_amount_num", "r_income_num").alias("spearman"),
                                  F.corr(F.log("loan_amount_num"), F.log("income_num")).alias("pearson_log")).toPandas())
print(f"Spark xong | {time.time() - t1:.0f}s")

# %% [markdown]
# ## 3. So khớp Pandas với Spark, ghi kết quả

# %%
def compare(name, a, b, keys):
    m = a.merge(b, on=keys, how="outer", suffixes=("_pd", "_sp"), indicator=True)
    assert (m["_merge"] == "both").all(), f"{name}: khóa không khớp\n{m[m['_merge'] != 'both']}"
    out = []
    for c in [c for c in a.columns if c not in keys and c in b.columns]:
        x, y = m[f"{c}_pd"].astype(float), m[f"{c}_sp"].astype(float)
        both_nan = x.isna() & y.isna()
        rel = ((x - y).abs() / np.maximum(x.abs(), y.abs())).where(~both_nan, 0).fillna(np.inf)
        rel = rel.where((x - y).abs() > 1e-15, 0)
        exact = c == "n"
        ok = (x.eq(y) | both_nan).all() if exact else bool((rel <= 1e-9).all())
        out.append({"output": name, "column": c, "values": int(len(m)), "max_rel_diff": float(rel.max()), "agree": ok})
    return out


CMP = pd.DataFrame(
    compare("summary statistics", PD_SUM, SP_SUM, ["variable", "group"])
    + compare("histograms", PD_HIST, SP_HIST, ["variable", "group", "bin"])
    + compare("loan amount by income band", PD_BAND.drop(columns="band"), SP_BAND.drop(columns="band"), ["group", "band_idx"])
    + compare("loan amount vs income correlation", PD_RHO, SP_RHO, ["group"]))
print(CMP.to_string(index=False))
assert CMP["agree"].all(), "Pandas và Spark lệch"
SUMM = CMP.groupby("output", sort=False).agg(values=("values", "sum"), max_rel_diff=("max_rel_diff", "max"),
                                             agree=("agree", "all")).reset_index()
PD_SUM.to_csv(f"{OUT}/summary_stats.csv", index=False)
PD_HIST.sort_values(["variable", "group", "bin"]).to_csv(f"{OUT}/histograms.csv", index=False)
PD_BAND.sort_values(["group", "band_idx"]).to_csv(f"{OUT}/loan_by_income_band.csv", index=False)
PD_RHO.to_csv(f"{OUT}/loan_income_correlation.csv", index=False)
SUMM.to_csv(f"{OUT}/pandas_vs_spark.csv", index=False)
json.dump({**EXCL, "rules_version": R["version"], "variables": {v: {"sample": s, "unit": u} for v, s, u in VARS},
           "seconds": round(time.time() - t0, 1)}, open(f"{OUT}/meta.json", "w"), indent=1, ensure_ascii=False)
print(SUMM.to_string(index=False))
print(PD_SUM[PD_SUM["group"] == "all"].round(3).to_string(index=False))
spark.stop()
