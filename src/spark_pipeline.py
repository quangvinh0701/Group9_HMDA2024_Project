# %% [markdown]
# # HMDA 2024: pipeline PySpark (rules.yaml 1.1)
#
# Notebook này thay cho notebook PySpark cũ, viết lại theo `rules.yaml` 1.1 và bổ sung các phần còn thiếu
# (mô hình hai chiều, HHI, độ nhạy, đầu ra tương đương, phần dự đoán của Spark).
#
# **Chuẩn bị trên Kaggle**
# 1. Input gồm dataset `hmda-2024-one-year-raw` (`hmda_2024_raw.parquet`) và `hmda-config` (`rules.yaml` bản 1.1, `expected.yaml`).
#    Transmittal Sheet lấy từ dataset `hmda-2024-ffiec-files` (Kaggle nhập từ link FFIEC); thiếu file này thì notebook thử tải khi có Internet.
# 2. Nếu đã chạy notebook Pandas, thêm output của nó làm Input để phần 11 dùng xác suất out-of-fold của Pandas.
# 3. Settings: Accelerator = None. Ảnh Kaggle có sẵn PySpark 4.0 và Java 21, không cần Internet. Chạy *Run All*.
# 4. Kết quả ghi vào `/kaggle/working/results/spark/`; các file `eq_*.csv` và `key_numbers.json` có cùng định dạng
#    với bên Pandas để `check_equivalence.py` so sánh.
#
# Nguyên tắc: phần tính trên dữ liệu lớn (lọc, tạo biến, groupBy, join, window, vòng lặp hai chiều, MLlib) chạy bằng
# Spark; các thống kê cuối trên bảng nhỏ (vài trăm dòng, ví dụ τ hay Spearman giữa 767 lender) dùng cùng công thức
# Python với bên Pandas để hai bên ra đúng một con số.

# %%
# ===== 0. Cài đặt và khởi tạo Spark =====
import os, glob, json, time, shutil, math, hashlib, subprocess, sys
import numpy as np
import pandas as pd
import yaml
import matplotlib

try:
    get_ipython  # noqa: F821
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyspark  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyspark"], check=True)
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql import types as T

T_START = time.time()
RUN_DUPLICATES = True     # đếm dòng trùng trên 99 cột bằng xxhash64
RUN_MLLIB = True          # logistic regression MLlib, cross-fitting 5 fold (mục 11)


def find_file(pattern, required=True):
    """Tìm file ở mọi dataset/notebook đã Add Input (hoặc thư mục HMDA_INPUT khi chạy ngoài Kaggle)."""
    roots = [os.environ["HMDA_INPUT"]] if os.environ.get("HMDA_INPUT") else ["/kaggle/input"]
    hits = sorted(h for r_ in roots for h in glob.glob(os.path.join(r_, "**", pattern), recursive=True))
    assert hits or not required, f"Không thấy {pattern} trong {roots}: kiểm tra Add Input"
    return hits[0] if hits else None


RAW_SRC = find_file("hmda_2024_raw.parquet")
RULES_PATH = find_file("rules.yaml")
EXP_PATH = find_file("expected.yaml", required=False)
TS_PATH = find_file("*public_ts*.csv", required=False) or find_file("*public_ts*.zip", required=False)
INP = os.path.dirname(RAW_SRC)
OUTROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("HMDA_OUT", "output")
RES, FIG = f"{OUTROOT}/results/spark", f"{OUTROOT}/figures/spark"
WORK = os.environ.get("HMDA_WORK", "/tmp/hmda")
for d in (RES, FIG, WORK):
    os.makedirs(d, exist_ok=True)
R = yaml.safe_load(open(RULES_PATH, encoding="utf-8"))
assert float(R["version"]) >= 1.1, f"Cần rules.yaml 1.1 trở lên, đang là {R.get('version')}"
if TS_PATH is None and R["source"]["transmittal_sheet"].get("url"):   # chưa có TS trong Input: thử tải từ FFIEC (cần Internet)
    try:
        import urllib.request
        os.makedirs(os.environ.get("HMDA_WORK", "/tmp/hmda"), exist_ok=True)
        TS_PATH = os.path.join(os.environ.get("HMDA_WORK", "/tmp/hmda"), "ts_one_year.zip")
        urllib.request.urlretrieve(R["source"]["transmittal_sheet"]["url"], TS_PATH)
    except Exception as e_:
        print("không tải được Transmittal Sheet:", repr(e_)[:100]); TS_PATH = None
if TS_PATH and TS_PATH.endswith(".zip"):          # file TS tải từ FFIEC dạng zip: giải nén để Spark đọc
    import zipfile
    os.makedirs(os.environ.get("HMDA_WORK", "/tmp/hmda"), exist_ok=True)
    with zipfile.ZipFile(TS_PATH) as z_:
        nm_ = [n for n in z_.namelist() if n.lower().endswith((".csv", ".txt"))][0]
        TS_PATH = z_.extract(nm_, os.environ.get("HMDA_WORK", "/tmp/hmda"))
EXP = yaml.safe_load(open(EXP_PATH, encoding="utf-8")) if EXP_PATH else {}
RAW = f"{WORK}/hmda_2024_raw.parquet"
if not os.path.exists(RAW) or os.path.getsize(RAW) != os.path.getsize(RAW_SRC):
    shutil.copy(RAW_SRC, RAW)

# Bộ nhớ driver = 60% của min(RAM còn trống lúc khởi tạo, giới hạn cgroup của container; Kaggle giới hạn 30 GiB)
try:
    avail_gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
except (ValueError, OSError):
    avail_gb = 8
for p_ in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
        v_ = open(p_).read().strip()
        if v_.isdigit():
            avail_gb = min(avail_gb, int(v_) / 1e9)
    except OSError:
        pass
driver_gb = max(2, int(avail_gb * 0.6))
t0 = time.time()
spark = (SparkSession.builder.master("local[*]").appName("hmda-v1")
         .config("spark.driver.memory", f"{driver_gb}g")
         .config("spark.sql.shuffle.partitions", str(max(8, 2 * (os.cpu_count() or 2))))
         .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
         .config("spark.sql.ansi.enabled", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir(f"{WORK}/checkpoints")
SPARK_INIT_S = time.time() - t0
print(f"Spark {spark.version} | khởi tạo {SPARK_INIT_S:.1f}s | driver {driver_gb} GB | input {INP}")

V, TH, G = R["variables"], R["thresholds"], R["geography"]
LT_NAME, LP_NAME = R["codes"]["loan_type"], R["codes"]["loan_purpose"]
UNIVERSE = list(G["universe_states"])
KEY = {"rules_version": str(R["version"]), "engine": "spark", "spark_init_s": SPARK_INIT_S}
CHECKS = []


def exp_value(path):
    node = EXP
    for p in path.split("."):
        if not isinstance(node, dict) or p not in node:
            return None, None, None
        node = node[p]
    return (node.get("value"), node.get("status"), node.get("tol", 1e-9)) if isinstance(node, dict) else (node, None, 1e-9)


def check(label, value, path=None):
    ev, st, tol = exp_value(path) if path else (None, None, None)
    ok = None if (ev is None or value is None or st == "pending") else (abs(float(value) - float(ev)) <= tol * max(1.0, abs(float(ev))))
    CHECKS.append({"check": label, "value": value, "expected": ev, "expected_status": st, "match": ok})
    flag = "" if ok is None else ("  OK" if ok else "  <-- LỆCH")
    print(f"  {label:<44} {value!s:>16}   kỳ vọng {ev!s:>12} ({st}){flag}")


def save(pdf, name):
    pdf.to_csv(f"{RES}/{name}.csv", index=False)


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return ctr - half, ctr + half


def spearman(x, y):
    x, y = pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    m = x.notna() & y.notna()
    return float(x[m].rank(method="average").corr(y[m].rank(method="average")))


def show():
    if IN_NOTEBOOK:
        plt.show()
    plt.close("all")


def mapper(d):
    return F.create_map(*[F.lit(x) for kv in d.items() for x in kv])

# %% [markdown]
# ## 1. Đọc dữ liệu thô, bảng chất lượng dữ liệu, lọc mẫu core
# Số dòng sau 7 bước lọc tính trong **một** lần quét bằng các cờ tích lũy (không đếm lại từng bước).

# %%
t0 = time.time()
raw = spark.read.parquet(RAW)
ALLCOLS = [c for c in raw.columns if c != "row_id"]
FILT = R["samples"]["core_filters"]
DQ = R["data_quality"]["special_code_table"]

conds, acc = [], F.lit(True)
for f in FILT:
    col = F.col(f["column"])
    if f["step"] == "closed_end":
        open_end_exempt_expr = F.sum((acc & (col == "1111")).cast("long")).alias("open_end_exempt_removed")
    acc = acc & (col.isin(f["in"]) if "in" in f else (col == f["equals"]))
    conds.append(F.sum(acc.cast("long")).alias(f["step"]))
sc_exprs = [F.sum((F.col(f) == code).cast("long")).alias(f"{f}||{code}") for f in DQ["fields"] for code in DQ["codes"]]
row = raw.agg(F.count(F.lit(1)).alias("n_all"), *conds, open_end_exempt_expr, *sc_exprs).first().asDict()
n_all = row["n_all"]
step_counts = {f["step"]: row[f["step"]] for f in FILT}
open_end_exempt_removed = row["open_end_exempt_removed"]
FP = R["data_quality"]["field_profile"]
fp_exprs = ([F.sum((F.col(f) == code).cast("long")).alias(f"{f}||{code}") for f in ALLCOLS for code in FP["count_codes"]] +
            [F.sum(F.col(f).rlike(FP["numeric_pattern"]).cast("long")).alias(f"{f}||numeric") for f in ALLCOLS])
fp_row = raw.agg(*fp_exprs).first().asDict()
fprof = pd.DataFrame([{"field": f, **{(c_ if c_ else "(empty)"): int(fp_row[f"{f}||{c_}"]) for c_ in FP["count_codes"] + ["numeric"]}}
                      for f in ALLCOLS])
fprof["n"] = n_all
for c_ in [x if x else "(empty)" for x in FP["count_codes"]] + ["numeric"]:
    fprof[f"share_{c_}"] = fprof[c_] / n_all
dq = pd.DataFrame([{"field": f, "code": (code if code else "(empty)"), "count": int(row[f"{f}||{code}"])}
                   for f in DQ["fields"] for code in DQ["codes"]])
act_counts = {r["action_taken"]: r["count"] for r in raw.groupBy("action_taken").count().collect()}
n_dup = 0
if RUN_DUPLICATES:
    n_dup = int(raw.select(F.xxhash64(*ALLCOLS).alias("h")).groupBy("h").count()
                .agg(F.sum(F.col("count") - 1)).first()[0] or 0)
core = raw.filter(acc)
print(f"đọc {n_all:,} dòng, core {step_counts[FILT[-1]['step']]:,} | trùng {n_dup:,} | {time.time() - t0:.0f}s")

# %% [markdown]
# ## 2. Tạo biến (rules mục 6) và ghi bảng sạch phân vùng theo `state_code`
# `try_cast` biến `NA`, `Exempt`, chuỗi rỗng thành null; NaN (nếu có chuỗi "NaN") cũng đổi thành null để so sánh
# không bị lệch (Spark coi NaN lớn hơn mọi số).

# %%
t0 = time.time()


def to_num(name):
    x = F.expr(f"try_cast(`{name}` as double)")
    return F.when(F.isnan(x), F.lit(None).cast("double")).otherwise(x)


def band(col, spec, cuts=None):
    rs = lambda v: cuts[v] if isinstance(v, str) else float(v)
    expr = None
    for b in spec:
        lo, hi = rs(b["lo"]), rs(b["hi"])
        cond = F.lit(True)
        if lo != -math.inf:
            cond = cond & (col > F.lit(lo))
        if hi != math.inf:
            cond = cond & (col <= F.lit(hi))
        expr = F.when(cond, F.lit(b["label"])) if expr is None else expr.when(cond, F.lit(b["label"]))
    return expr


def map_flag(colname, mapping):
    m = {k: v for k, v in mapping.items() if v is not None}
    return mapper({k: int(v) for k, v in m.items()})[F.col(colname)].cast("int")


D, L = V["debt_to_income_ratio"], V["combined_loan_to_value_ratio"]
CUT = V["loan_to_income"]["cut_exact"]
RSV, IRV, LTV_, TCV = (V["rate_spread"]["valid_range"], V["interest_rate"]["valid_range"],
                       V["loan_term"]["valid_range"], V["total_loan_costs"]["valid_range"])
S_ = R["sampling"]
dti_map = mapper({str(k): int(v) for k, v in D["ordinal_map"].items()})
act = F.col("action_taken")
inc, amt = to_num("income"), to_num("loan_amount")
cltv = to_num("combined_loan_to_value_ratio")
tract = to_num("tract_to_msa_income_percentage")
sub, pay = F.col("submission_of_application"), F.col("initially_payable_to_institution")

feat = (core
  .withColumn("decisioned", act.isin(R["samples"]["decisioned"]["action_taken_in"]))
  .withColumn("denied", (act == "3"))
  .withColumn("orig", (act == "1"))
  .withColumn("y", (act == "3").cast("double"))
  .withColumn("dti_ordinal", dti_map[F.regexp_replace(F.col("debt_to_income_ratio"), "[% ]", "")].cast("double"))
  .withColumn("cltv", cltv)
  .withColumn("cltv_valid", F.coalesce((F.col("cltv") > L["valid_range"]["gt"]) & (F.col("cltv") <= L["valid_range"]["le"]), F.lit(False)))
  .withColumn("income_num", inc)
  .withColumn("income_is_na", F.col("income") == "NA")
  .withColumn("loan_amount_num", amt)
  .withColumn("lti", F.when(F.col("income_num") > 0, F.col("loan_amount_num") / (1000 * F.col("income_num"))))
  .withColumn("tract_ratio", F.when(tract != 0, tract) if V["lmi_tract"]["zero_means_missing"] else tract)
  .withColumn("lmi", F.col("tract_ratio") < V["lmi_tract"]["lmi_if_below"])
  .withColumn("jumbo", map_flag("conforming_loan_limit", V["jumbo"]["values"]))
  .withColumn("broker", map_flag("submission_of_application", V["broker"]["values"]))
  .withColumn("channel3", F.when(sub.isin("3", "1111") | pay.isin("3", "1111"), F.lit(None).cast("string"))
                           .when(pay == "2", "third_party").when((sub == "1") & (pay == "1"), "retail")
                           .when((sub == "2") & (pay == "1"), "broker"))
  .withColumn("rs_num", to_num("rate_spread"))
  .withColumn("rs_valid", F.coalesce((F.col("rs_num") >= RSV["ge"]) & (F.col("rs_num") <= RSV["le"]), F.lit(False)))
  .withColumn("rs_out_of_range", F.col("rs_num").isNotNull() & ~F.col("rs_valid"))
  .withColumn("ir_num", to_num("interest_rate"))
  .withColumn("ir_valid", F.coalesce((F.col("ir_num") >= IRV["ge"]) & (F.col("ir_num") <= IRV["le"]), F.lit(False)))
  .withColumn("term_num", to_num("loan_term"))
  .withColumn("term_valid", F.coalesce((F.col("term_num") > LTV_["gt"]) & (F.col("term_num") <= LTV_["le"]), F.lit(False)))
  .withColumn("tlc_num", to_num("total_loan_costs"))
  .withColumn("tlc_valid", F.coalesce(F.col("tlc_num") >= TCV["ge"], F.lit(False)))
  .withColumn("in_universe", F.col("state_code").isin(UNIVERSE))
  .withColumn("state_missing", F.col("state_code").isin(G["missing_values"]))
  .withColumn("county_valid", F.coalesce(F.col("county_code").rlike(G["county_code_format"]), F.lit(False)))
  .withColumn("msa", F.col("derived_msa_md"))
  .withColumn("type", mapper(LT_NAME)[F.col("loan_type")])
  .withColumn("purpose", mapper(LP_NAME)[F.col("loan_purpose")])
  .withColumn("sample_bucket", F.pmod(F.pmod(F.col("row_id") * F.lit(S_["hash_multiplier"]).cast("long"),
                                             F.lit(S_["hash_modulus"]).cast("long")), F.lit(S_["buckets"]).cast("long")))
)
for i in range(1, 5):
    feat = feat.withColumn(f"dr{i}", F.coalesce(F.expr(f"try_cast(denial_reason_{i} as int)"), F.lit(0)))
feat = (feat
  .withColumn("dti_band", band(F.col("dti_ordinal"), D["bands"]))
  .withColumn("cltv_band", F.when(F.col("cltv_valid"), band(F.col("cltv"), L["bands"])))
  .withColumn("lti_band", band(F.col("lti"), V["loan_to_income"]["bands"], CUT))
  .withColumn("usable", F.col("decisioned") & F.col("dti_ordinal").isNotNull() & F.col("cltv_valid")
                        & F.coalesce(F.col("income_num") > 0, F.lit(False)))
  .withColumn("pricing", F.col("orig") & F.col("rs_valid"))
  .withColumn("cost_share", F.when(F.col("pricing") & F.col("tlc_valid") & (F.col("loan_amount_num") > 0),
                                   F.col("tlc_num") / F.col("loan_amount_num")))
)
r_m, n_m = F.col("ir_num") / 1200, F.col("term_num")
pay_m = F.when(r_m == 0, F.col("loan_amount_num") / n_m).otherwise(F.col("loan_amount_num") * r_m / (1 - F.pow(1 + r_m, -n_m)))
feat = feat.withColumn("pti", F.when(F.col("pricing") & F.col("ir_valid") & F.col("term_valid") & (F.col("income_num") > 0),
                                     pay_m / (1000 * F.col("income_num") / 12)))

RAW_KEEP = ["row_id", "lei", "state_code", "county_code", "action_taken", "loan_type", "loan_purpose", "rate_spread",
            "income", "loan_amount"]
KEEPC = RAW_KEEP + [c_ for c_ in feat.columns if c_ not in ALLCOLS and c_ != "row_id"]
CLEAN = f"{WORK}/clean_core"
feat.select(*KEEPC).repartition(R["source"]["partition_clean_by"]).write.mode("overwrite") \
    .partitionBy(R["source"]["partition_clean_by"]).parquet(CLEAN)
c = spark.read.parquet(CLEAN).cache()
n_core = c.count()
print(f"bảng sạch: {n_core:,} dòng, {len(c.columns)} cột | {time.time() - t0:.0f}s")

# %% [markdown]
# ## 3. Kiểm tra với expected.yaml, bảng chất lượng dữ liệu

# %%
dec = c.filter("decisioned")
U = c.filter("usable").cache()
n_dec, n_us = dec.count(), U.count()
print("Cỡ mẫu:")
check("all_records", n_all, "rows.all_records")
for f in FILT:
    check(f"core_filters.{f['step']}", step_counts[f["step"]], f"core_filters.{f['step']}")
check("open_end_exempt_removed", open_end_exempt_removed, "core_filters.open_end_exempt_removed")
check("decisioned", n_dec, "samples.decisioned")
check("usable", n_us, "samples.usable")
check("dev_sample_1pct", c.filter(F.col("sample_bucket") < S_["dev_sample_bucket_lt"]).count(), "samples.dev_sample_1pct")
inc_row = dec.agg(F.sum(F.col("income_is_na").cast("long")).alias("na"), F.sum((F.col("income_num") <= 0).cast("long")).alias("np"),
                  F.sum((F.col("income_num") > V["income"]["flag_above"]).cast("long")).alias("big"),
                  F.sum((F.col("cltv").isNotNull() & ~F.col("cltv_valid")).cast("long")).alias("cltv_bad"),
                  F.sum(F.col("dti_ordinal").isNull().cast("long")).alias("dti_miss")).first()
print("Income (decisioned):")
check("income NA", inc_row["na"], "income.na_decisioned")
check("income <= 0", inc_row["np"], "income.non_positive_decisioned")
check("income > 10000", inc_row["big"], "income.above_10000_decisioned")
geo = c.agg(F.countDistinct("state_code").alias("st"), F.countDistinct("county_code").alias("cty"),
            F.countDistinct(F.when(F.col("county_valid"), F.col("county_code"))).alias("cty_valid"),
            F.countDistinct("msa").alias("msa"),
            F.countDistinct(F.when(F.col("msa") != R["special_codes"]["msa_nonmetro"], F.col("msa"))).alias("msa_real"),
            F.sum((~F.col("county_valid")).cast("long")).alias("cty_bad"),
            F.sum(F.col("tract_ratio").isNull().cast("long")).alias("tract_missing")).first()
print("Địa lý:")
check("state codes in core (kể cả NA)", geo["st"], "geography.state_codes_in_core")
check("county codes in core (kể cả NA)", geo["cty"], "geography.county_codes_in_core")
check("county codes hợp lệ (5 chữ số)", geo["cty_valid"], "geography.county_codes_valid")
check("MSA codes in core (kể cả 99999)", geo["msa"], "geography.msa_codes_in_core")
check("MSA thật (bỏ 99999)", geo["msa_real"], "geography.msa_codes_real")
for s_ in ["PR", "GU", "VI"]:
    check(f"usable {s_}", U.filter(F.col("state_code") == s_).count(), f"geography.usable_{s_}")
check("thiếu bang, usable", U.filter("state_missing").count(), "geography.missing_state_usable")
check("thiếu bang, decisioned", dec.filter("state_missing").count(), "geography.missing_state_decisioned")
o_ = c.filter("orig")
rs_row = o_.agg(F.sum(F.col("rs_num").isNotNull().cast("long")).alias("num"), F.sum((F.col("rate_spread") == "NA").cast("long")).alias("na"),
                F.sum((F.col("rate_spread") == "Exempt").cast("long")).alias("ex"), F.sum(F.col("rs_out_of_range").cast("long")).alias("out"),
                F.sum((F.abs("rs_num") > 10).cast("long")).alias("gt10")).first()
print("Rate spread (core, đã giải ngân):")
check("rate_spread dạng số", rs_row["num"], "rate_spread_pricing.numeric")
check("rate_spread NA", rs_row["na"], "rate_spread_pricing.na")
check("rate_spread Exempt", rs_row["ex"], "rate_spread_pricing.exempt")
check("rate_spread ngoài [-8, 15]", rs_row["out"], "rate_spread_pricing.outside_valid_range")
check("|rate_spread| > 10", rs_row["gt10"], "rate_spread_pricing.abs_above_10")
n_pricing, n_pricing_u = c.filter("pricing").count(), c.filter("pricing and usable").count()
check("pricing ∩ usable", n_pricing_u, "rate_spread_pricing.pricing_usable")
share_out = rs_row["out"] / max(rs_row["num"], 1)
print(f"  tỷ lệ ngoài khoảng: {share_out:.6%} (điều kiện dừng 0.01%: {'VƯỢT, báo Vinh' if share_out > 1e-4 else 'không vượt'})")
KEY.update({"all_records": n_all, "core": n_core, "decisioned": n_dec, "usable": n_us, "pricing": n_pricing,
            "pricing_usable": n_pricing_u, "duplicates": n_dup, "rs_out_of_range": rs_row["out"]})

save(dq, "dq_special_codes")
save(fprof, "dq_field_profile")
dq_rows = [
    {"item": "exact duplicate rows (99 fields, all records)", "n": n_dup},
    {"item": "decisioned: income NA", "n": inc_row["na"]},
    {"item": "decisioned: income <= 0", "n": inc_row["np"]},
    {"item": "decisioned: income > 10,000 (flag only)", "n": inc_row["big"]},
    {"item": "decisioned: CLTV numeric but outside (0, 250]", "n": inc_row["cltv_bad"]},
    {"item": "decisioned: DTI missing", "n": inc_row["dti_miss"]},
    {"item": "core: county_code not 5 digits", "n": geo["cty_bad"]},
    {"item": "core: tract ratio 0 (treated as missing)", "n": geo["tract_missing"]},
    {"item": "core originations: rate_spread outside [-8, 15]", "n": rs_row["out"]},
]
save(pd.DataFrame(dq_rows), "dq_summary")
print(pd.DataFrame(dq_rows).to_string(index=False))
miss = (c.groupBy("action_taken").agg(
            (100 * F.avg(F.col("dti_ordinal").isNull().cast("double"))).alias("dti_missing"),
            (100 * F.avg((~F.col("cltv_valid")).cast("double"))).alias("cltv_invalid"),
            (100 * F.avg((~F.coalesce(F.col("income_num") > 0, F.lit(False))).cast("double"))).alias("income_bad"),
            F.count(F.lit(1)).alias("n")).orderBy("action_taken").toPandas())
save(miss, "dq_missing_by_action")
print(miss.round(1).to_string(index=False))

# %% [markdown]
# ## 4. SQ1: kết quả hồ sơ, tỷ lệ từ chối theo loại vay và mục đích

# %%
TYPE_ORDER = ["conventional", "FHA", "VA", "USDA"]
PURP_ORDER = ["purchase", "refinance", "cash_out_refinance", "home_improvement", "other", "not_applicable"]


def denial_table(df, keys):
    g = df.groupBy(*keys).agg(F.count(F.lit(1)).alias("n_decisioned"), F.sum(F.col("denied").cast("long")).alias("n_denied")).toPandas()
    g["denial_rate"] = g["n_decisioned"].pipe(lambda n: g["n_denied"] / n)
    lo, hi = wilson(g["n_denied"], g["n_decisioned"])
    g["ci_low"], g["ci_high"] = lo, hi
    g["show_ci"] = g["n_decisioned"] < R["data_quality"]["wilson_interval"]["apply_when_denominator_below"]
    return g


mix = c.groupBy("type").pivot("action_taken").count().toPandas().set_index("type").fillna(0)
mix = mix.rename(columns=R["codes"]["action_taken"])
out_pct = mix.div(mix.sum(axis=1), axis=0).reindex(TYPE_ORDER); out_pct["n_applications"] = mix.sum(axis=1)
save(out_pct.reset_index(), "sq1_outcomes_by_type")
by_type = denial_table(dec, ["type"]).set_index("type").reindex(TYPE_ORDER).reset_index()
by_purp = denial_table(dec, ["purpose"]).set_index("purpose").reindex(PURP_ORDER).reset_index()
cross = denial_table(dec, ["type", "purpose"])
save(by_type, "sq1_denial_by_type"); save(by_purp, "sq1_denial_by_purpose"); save(cross, "sq1_denial_type_x_purpose")
amt_t = (c.filter("orig").groupBy("type").agg(F.count("loan_amount_num").alias("n"), F.avg("loan_amount_num").alias("mean"),
                                               F.expr("percentile(loan_amount_num, 0.5)").alias("median"),
                                               F.expr("percentile(loan_amount_num, 0.25)").alias("p25"),
                                               F.expr("percentile(loan_amount_num, 0.75)").alias("p75"))
           .toPandas().set_index("type").reindex(TYPE_ORDER).reset_index())
save(amt_t, "sq1_loan_amount_originated_by_type")
KEY["denial_rate_core"] = float(dec.agg(F.avg("y")).first()[0])
for _, r in by_type.iterrows():
    KEY[f"denial_rate_{r['type']}"] = float(r["denial_rate"])
print(f"tỷ lệ từ chối core: {KEY['denial_rate_core']:.4f}")
print(by_type[["type", "n_decisioned", "denial_rate"]].round(4).to_string(index=False))

# %% [markdown]
# ## 5. SQ2: rủi ro, lý do từ chối, Kitagawa, giá

# %%
KEYS = ["dti_band", "cltv_band", "lti_band"]
DTI_LAB = [b["label"] for b in D["bands"]]; CLTV_LAB = [b["label"] for b in L["bands"]]
LTI_LAB = [b["label"] for b in V["loan_to_income"]["bands"]]
inv = {int(v): k for k, v in D["ordinal_map"].items()}
curve = U.groupBy("dti_ordinal").agg(F.count(F.lit(1)).alias("n"), F.avg("y").alias("denial_rate")).orderBy("dti_ordinal").toPandas()
curve["dti"] = curve["dti_ordinal"].astype(int).map(inv)
save(curve, "sq2_denial_by_dti_19")


def jump_tests(cnt):
    """cnt: cột scope, level, n, k. Kiểm định bước nhảy DTI (rules mục 21). Cùng công thức với bên Pandas."""
    J = R["dti_jump_test"]; z = J["ci_z"]; lv = J["levels"]; l36, l49 = lv[0], lv[-2]
    rows, bunch = [], []
    for scope, g in cnt.groupby("scope"):
        g = g.set_index("level")
        if not all(l in g.index for l in lv):
            continue
        r = (g["k"] / g["n"]).to_dict(); n = g["n"].to_dict()
        base = (r[l49] - r[l36]) / (l49 - l36)
        for jp in J["jumps"]:
            coef = {}
            for l, c_ in [(jp["to"], 1.0), (jp["from"], -1.0), (l49, -1.0 / (l49 - l36)), (l36, 1.0 / (l49 - l36))]:
                coef[l] = coef.get(l, 0.0) + c_
            est = sum(c_ * r[l] for l, c_ in coef.items())
            se = math.sqrt(sum(c_ * c_ * r[l] * (1 - r[l]) / n[l] for l, c_ in coef.items()))
            rows.append({"scope": scope, "jump": jp["label"], "role": jp["role"], "rate_from": r[jp["from"]], "rate_to": r[jp["to"]],
                         "step": r[jp["to"]] - r[jp["from"]], "baseline_step": base, "excess": est, "se": se,
                         "ci_low": est - z * se, "ci_high": est + z * se, "n_from": n[jp["from"]], "n_to": n[jp["to"]]})
        bunch.append({"scope": scope, "n_49": n[l49], "mean_n_44_48": float(np.mean([n[l] for l in range(l49 - 5, l49)])),
                      "bunching_ratio": n[l49] / float(np.mean([n[l] for l in range(l49 - 5, l49)]))})
    return pd.DataFrame(rows), pd.DataFrame(bunch)


JL = R["dti_jump_test"]["levels"]
uj = U.filter(F.col("dti_ordinal").isin(JL))
c_all = uj.groupBy("dti_ordinal").agg(F.count(F.lit(1)).alias("n"), F.sum("y").alias("k")).withColumn("scope", F.lit("all"))
c_t = (uj.filter(F.col("loan_type").isin([x for x in R["dti_jump_test"]["by"] if x != "all"]))
         .groupBy(F.col("loan_type").alias("scope"), "dti_ordinal").agg(F.count(F.lit(1)).alias("n"), F.sum("y").alias("k")))
jcnt = (c_all.select("scope", "dti_ordinal", "n", "k").unionByName(c_t.select("scope", "dti_ordinal", "n", "k")).toPandas()
            .rename(columns={"dti_ordinal": "level"}))
jcnt["level"] = jcnt["level"].astype(int); jcnt["k"] = jcnt["k"].astype(float)
jt, jb = jump_tests(jcnt)
lo_, hi_ = wilson(jcnt["k"], jcnt["n"]); jcnt["rate"] = jcnt["k"] / jcnt["n"]; jcnt["ci_low"], jcnt["ci_high"] = lo_, hi_
jcnt["dti"] = jcnt["level"].map(inv)
save(jcnt.sort_values(["scope", "level"]), "sq2_dti_jump_table"); save(jt, "sq2_dti_jump_test"); save(jb, "sq2_dti_bunching")
print(jt[jt["scope"] == "all"][["jump", "role", "step", "baseline_step", "excess", "ci_low", "ci_high"]].round(4).to_string(index=False))
for _, r_ in jt.iterrows():
    tag = f"jump_{r_['scope']}_{r_['jump'].replace(' -> ', '_to_').replace('-', '')}"
    KEY[f"{tag}_excess"], KEY[f"{tag}_ci_low"], KEY[f"{tag}_ci_high"] = r_["excess"], r_["ci_low"], r_["ci_high"]
for _, r_ in jb.iterrows():
    KEY[f"bunching_49_{r_['scope']}"] = r_["bunching_ratio"]

# Chẩn đoán DTI dưới 20% (rules mục 22)
LL = R["dti_low_diagnostic"]["levels"]
ul = U.filter(F.col("dti_ordinal").isin(LL)).withColumn("dti", mapper({int(k): v for k, v in inv.items()})[F.col("dti_ordinal").cast("int")])
t1 = ul.groupBy("dti", "type", "purpose").agg(F.count(F.lit(1)).alias("n"), F.sum("y").alias("k")).toPandas()
t1["rate"] = t1["k"] / t1["n"]; t1["ci_low"], t1["ci_high"] = wilson(t1["k"], t1["n"])
save(t1, "sq2_low_dti_by_type_purpose")
RNX = {int(a): b for a, b in R["codes"]["denial_reason"].items() if a not in R["denial_reasons"]["exclude_codes"]}
t2 = (ul.filter("denied").groupBy("dti")
        .agg(*[F.avg(((F.col("dr1") == k) | (F.col("dr2") == k) | (F.col("dr3") == k) | (F.col("dr4") == k)).cast("double")).alias(nm)
               for k, nm in RNX.items()], F.count(F.lit(1)).alias("n_denied")).toPandas().set_index("dti"))
save(t2.reset_index(), "sq2_low_dti_reasons")
t3 = (ul.groupBy("dti").agg(F.count(F.lit(1)).alias("n"), F.avg("y").alias("denial_rate"),
                            *[F.expr(f"percentile({x}, 0.5)").alias(nm) for x, nm in
                              [("income_num", "median_income"), ("loan_amount_num", "median_loan_amount"), ("lti", "median_lti"), ("cltv", "median_cltv")]])
        .toPandas().set_index("dti"))
for col_ in ["type", "purpose"]:
    ct = ul.groupBy("dti", col_).count().toPandas().pivot(index="dti", columns=col_, values="count").fillna(0)
    t3 = t3.join(ct.div(ct.sum(axis=1), axis=0).add_prefix("share_"))
save(t3.reset_index(), "sq2_low_dti_profile")
print(t3[["n", "denial_rate", "median_income", "median_loan_amount", "median_lti"]].round(3).to_string())
for d_, r_ in t3.iterrows():
    KEY[f"low_dti_{d_}_denial_rate"] = float(r_["denial_rate"]); KEY[f"low_dti_{d_}_median_income"] = float(r_["median_income"])
for d_, r_ in t2.iterrows():
    KEY[f"low_dti_{d_}_reason_credit_history"] = float(r_["credit_history"]); KEY[f"low_dti_{d_}_reason_collateral"] = float(r_["collateral"])
for k, lab in [("dti_band", DTI_LAB), ("cltv_band", CLTV_LAB), ("lti_band", LTI_LAB)]:
    t_ = U.groupBy(k).agg(F.count(F.lit(1)).alias("n"), F.avg("y").alias("denial_rate")).toPandas().set_index(k).reindex(lab).reset_index()
    save(t_, f"sq2_denial_by_{k}")
save(U.groupBy("dti_band", "cltv_band").agg(F.count(F.lit(1)).alias("n"), F.avg("y").alias("rate")).toPandas(), "sq2_heatmap_dti_cltv")

RN = {int(k): v for k, v in R["codes"]["denial_reason"].items()}
excl = {int(x) for x in R["denial_reasons"]["exclude_codes"]}
REASONS = sorted(set(RN) - excl)
cites = [F.avg(((F.col("dr1") == r) | (F.col("dr2") == r) | (F.col("dr3") == r) | (F.col("dr4") == r)).cast("double")).alias(RN[r])
         for r in REASONS]
den = U.filter("denied")
for by in ["type", "purpose", "dti_band"]:
    t_ = den.groupBy(by).agg(*cites, F.count(F.lit(1)).alias("n_denied")).toPandas()
    save(t_, f"sq2_reasons_by_{by}")
    if by == "purpose":
        t_ = t_.set_index("purpose")
        KEY["reason_credit_history_cashout"] = float(t_.loc["cash_out_refinance", "credit_history"]) if "cash_out_refinance" in t_.index else None
        KEY["reason_credit_history_purchase"] = float(t_.loc["purchase", "credit_history"]) if "purchase" in t_.index else None


def kitagawa(df, col, a, b, y="y", keys=KEYS, min_n=TH["cell"]["min_n"]):
    """Phân rã Kitagawa trên bảng ô (Spark): rules mục 11."""
    sub_ = df.filter(F.col(col).isin(a, b))
    g = sub_.groupBy(*keys, col).agg(F.count(F.lit(1)).alias("n"), F.avg(y).alias("r"), F.sum(y).alias("s"))
    A = g.filter(F.col(col) == a).select(*keys, F.col("n").alias("nA"), F.col("r").alias("rA"))
    B = g.filter(F.col(col) == b).select(*keys, F.col("n").alias("nB"), F.col("r").alias("rB"))
    tot = g.groupBy(col).agg(F.sum("n").alias("N"), F.sum("s").alias("S")).collect()
    N = {r_[col]: r_["N"] for r_ in tot}; Sm = {r_[col]: r_["S"] for r_ in tot}
    t = A.join(B, keys, "inner").filter((F.col("nA") >= min_n) & (F.col("nB") >= min_n))
    t = t.crossJoin(t.agg(F.sum("nA").alias("TA"), F.sum("nB").alias("TB")))
    t = (t.withColumn("sA", F.col("nA") / F.col("TA")).withColumn("sB", F.col("nB") / F.col("TB"))
          .withColumn("dr", F.col("rA") - F.col("rB")))
    s = t.agg(F.sum(F.col("sA") * F.col("rA")).alias("mA"), F.sum(F.col("sB") * F.col("rB")).alias("mB"),
              F.sum((F.col("sA") - F.col("sB")) * (F.col("rA") + F.col("rB")) / 2).alias("comp"),
              F.sum(F.col("dr") * (F.col("sA") + F.col("sB")) / 2).alias("rate"),
              F.sum(F.col("sA") * F.col("dr")).alias("atA"), F.sum(F.col("sB") * F.col("dr")).alias("atB"),
              F.count(F.lit(1)).alias("cells"), F.first("TA").alias("TA"), F.first("TB").alias("TB"),
              F.min("nA").alias("minA"), F.min("nB").alias("minB"), F.min("dr").alias("drmin"), F.max("dr").alias("drmax"),
              F.sum((F.col("dr") > 0).cast("int")).alias("hi"), F.sum((F.col("dr") < 0).cast("int")).alias("lo")).first()
    if not s["cells"]:   # không có ô chung: không phân rã được
        return {"a": a, "b": b, "n_common_cells": 0, "coverage_a": 0.0, "coverage_b": 0.0, "coverage_ge_50pct": False,
                **{k: np.nan for k in ["gap", "composition", "rate", "rate_gap_at_a_mix", "rate_gap_at_b_mix", "min_cell_n_a",
                                       "min_cell_n_b", "within_cell_gap_min", "within_cell_gap_max", "cells_a_higher",
                                       "cells_a_lower", "raw_gap", "sign_flip"]}}
    res = {"a": a, "b": b, "gap": s["mA"] - s["mB"], "composition": s["comp"], "rate": s["rate"],
           "rate_gap_at_a_mix": s["atA"], "rate_gap_at_b_mix": s["atB"], "n_common_cells": s["cells"],
           "coverage_a": s["TA"] / N[a], "coverage_b": s["TB"] / N[b], "min_cell_n_a": s["minA"], "min_cell_n_b": s["minB"],
           "within_cell_gap_min": s["drmin"], "within_cell_gap_max": s["drmax"], "cells_a_higher": s["hi"], "cells_a_lower": s["lo"],
           "raw_gap": Sm[a] / N[a] - Sm[b] / N[b]}
    res["sign_flip"] = bool(np.sign(res["rate_gap_at_a_mix"]) != np.sign(res["rate_gap_at_b_mix"]))
    res["coverage_ge_50pct"] = bool(min(res["coverage_a"], res["coverage_b"]) >= 0.5)
    return res


kt = []
for p_ in R["kitagawa"]["pairs"]:
    r = kitagawa(U, p_["column"], p_["a"], p_["b"]); r["label"] = p_["label"]; r["column"] = p_["column"]; kt.append(r)
kt = pd.DataFrame(kt); save(kt, "sq2_kitagawa_denial")
for _, r in kt.iterrows():
    tag = f"{r['column']}_{r['a']}v{r['b']}"
    for k in ["gap", "composition", "rate", "rate_gap_at_a_mix", "rate_gap_at_b_mix", "within_cell_gap_min",
              "within_cell_gap_max", "cells_a_higher", "cells_a_lower", "n_common_cells", "min_cell_n_a"]:
        KEY[f"kitagawa_{tag}_{k}"] = r[k]
print(kt[["label", "gap", "composition", "rate", "rate_gap_at_a_mix", "rate_gap_at_b_mix", "n_common_cells"]].round(4).to_string(index=False))
cells_ = U.groupBy(*KEYS).agg(F.count(F.lit(1)).alias("n"), F.sum((F.col("loan_type") == "2").cast("long")).alias("fha"))
cr = cells_.agg(F.count(F.lit(1)).alias("k"), F.min("n").alias("mn"), F.min("fha").alias("mf")).first()
KEY.update({"n_risk_cells": cr["k"], "smallest_cell_n": cr["mn"], "min_fha_per_cell": cr["mf"]})
save(cells_.toPandas(), "sq2_risk_cells")
print(f"ô rủi ro: {cr['k']} | nhỏ nhất {cr['mn']} | FHA nhỏ nhất {cr['mf']}")
check("số ô rủi ro", KEY["n_risk_cells"], "risk_cells.n_cells")
check("ô rủi ro nhỏ nhất", KEY["smallest_cell_n"], "risk_cells.smallest_cell_n")
check("FHA ít nhất mỗi ô", KEY["min_fha_per_cell"], "risk_cells.min_fha_per_cell")
check("Kitagawa FHA vs conv: gap", KEY["kitagawa_loan_type_2v1_gap"], "kitagawa_fha_vs_conventional.gap")
check("Kitagawa FHA vs conv: composition", KEY["kitagawa_loan_type_2v1_composition"], "kitagawa_fha_vs_conventional.composition")
check("Kitagawa FHA vs conv: rate", KEY["kitagawa_loan_type_2v1_rate"], "kitagawa_fha_vs_conventional.rate")

# Giá
P = c.filter("pricing")
rs_desc = (P.groupBy("type").agg(F.count(F.lit(1)).alias("n"), F.avg("rs_num").alias("mean"),
                                 F.expr("percentile(rs_num, 0.5)").alias("median"), F.expr("percentile(rs_num, 0.25)").alias("p25"),
                                 F.expr("percentile(rs_num, 0.75)").alias("p75"), F.stddev_samp("rs_num").alias("sd"))
             .toPandas().set_index("type").reindex(TYPE_ORDER).reset_index())
save(rs_desc, "sq2_rate_spread_by_type")
PU = c.filter("pricing and usable").withColumnRenamed("rs_num", "rsy")
for k, lab in [("dti_band", DTI_LAB), ("cltv_band", CLTV_LAB), ("lti_band", LTI_LAB)]:
    save(PU.groupBy("type", k).agg(F.count(F.lit(1)).alias("n"), F.avg("rsy").alias("mean"),
                                   F.expr("percentile(rsy, 0.5)").alias("median")).toPandas(), f"sq2_rate_spread_by_{k}")
pk = []
for lt_code, lt_name in [("1", "conventional"), ("2", "FHA"), ("3", "VA")]:
    for a, b in [("32", "1"), ("31", "1")]:
        r = kitagawa(PU.filter(F.col("loan_type") == lt_code), "loan_purpose", a, b, y="rsy"); r["loan_type"] = lt_name; pk.append(r)
pk = pd.DataFrame(pk); save(pk, "sq2_kitagawa_rate_spread")
for _, r in pk.iterrows():
    for k in ["gap", "composition", "rate", "coverage_a", "coverage_b"]:
        KEY[f"pricing_{r['loan_type']}_{r['a']}v{r['b']}_{k}"] = float(r[k])
print(pk[["loan_type", "a", "b", "gap", "composition", "rate", "coverage_a", "coverage_b", "coverage_ge_50pct"]].round(3).to_string(index=False))
base = c.filter("orig and usable and rs_num is not null").withColumnRenamed("rs_num", "rsy")
sens = []
for name_, cond in [("all numeric", F.lit(True)), ("[-8, 15] (rules)", F.col("rs_valid")), ("|rs| <= 5", F.abs("rsy") <= 5)]:
    for lt_code, lt_name in [("1", "conventional"), ("2", "FHA"), ("3", "VA")]:
        r = kitagawa(base.filter(cond & (F.col("loan_type") == lt_code)), "loan_purpose", "32", "1", y="rsy")
        r.update(cut=name_, loan_type=lt_name); sens.append(r)
save(pd.DataFrame(sens), "appendix_rate_spread_cut_sensitivity")
cov_ev = (c.filter("orig and usable").filter(F.col("loan_purpose").isin("1", "32"))
           .groupBy("type", "purpose").agg(F.count(F.lit(1)).alias("n"), F.avg((F.col("cltv") > 80).cast("double")).alias("share_cltv_gt_80"))
           .toPandas())
save(cov_ev, "sq2_cltv_gt80_by_type_purpose")
print(cov_ev.round(3).to_string(index=False))

# %% [markdown]
# ## 6. Giá trị kỳ vọng p_ext (rules mục 7): broadcast join bảng tỷ lệ theo ô vào từng hồ sơ

# %%
EXTK = R["extended_cells"]["keys"]


def add_pext(df, min_n, ext_keys=None, risk_keys=None):
    ext_keys, risk_keys = ext_keys or EXTK, risk_keys or KEYS
    r_ext = df.groupBy(*ext_keys).agg(F.count(F.lit(1)).alias("n_ext"), F.avg("y").alias("r_ext"))
    r_risk = df.groupBy(*risk_keys).agg(F.avg("y").alias("r_risk"))
    return (df.join(F.broadcast(r_ext), ext_keys, "left").join(F.broadcast(r_risk), risk_keys, "left")
              .withColumn("p_ext", F.when(F.col("n_ext") >= min_n, F.col("r_ext")).otherwise(F.col("r_risk")))
              .withColumn("v_ext", F.col("p_ext") * (1 - F.col("p_ext")))
              .withColumn("fallback", F.col("n_ext") < min_n))


UE_PATH = f"{WORK}/ud_exp"
add_pext(U, R["extended_cells"]["min_n"]).write.mode("overwrite").parquet(UE_PATH)
UE = spark.read.parquet(UE_PATH).cache()
nat = UE.agg(F.sum("y").alias("O"), F.sum("p_ext").alias("E"), F.avg(F.col("fallback").cast("double")).alias("fb"),
             F.sum(F.when(~F.col("fallback"), F.col("y"))).alias("O_nf"), F.sum(F.when(~F.col("fallback"), F.col("p_ext"))).alias("E_nf")).first()
n_ext_cells = UE.select(*EXTK).distinct().count()
KEY.update({"national_O": nat["O"], "national_E": nat["E"], "national_oe": nat["O"] / nat["E"], "n_ext_cells": n_ext_cells,
            "fallback_share": nat["fb"], "O_nonfallback": nat["O_nf"], "E_nonfallback": nat["E_nf"]})
print(f"O = {nat['O']:,.0f} | E = {nat['E']:,.1f} | O/E = {nat['O'] / nat['E']:.4f} | ô mở rộng {n_ext_cells:,} | fallback {nat['fb']:.4%}")
print(f"chỉ các ô >= 30: O = {nat['O_nf']:,.0f}, E = {nat['E_nf']:,.1f} (phải bằng nhau)")
check("O usable denied", nat["O"], "national_oe.O_usable_denied")
check("E usable", nat["E"], "national_oe.E_usable")
check("O/E toàn quốc", nat["O"] / nat["E"], "national_oe.oe")
check("số ô mở rộng", n_ext_cells, "risk_cells.extended_cells")
check("tỷ lệ fallback", nat["fb"], "risk_cells.fallback_share")
fb = (UE.filter("fallback").groupBy("type", "purpose").agg(F.count(F.lit(1)).alias("n"), F.sum("y").alias("O"), F.sum("p_ext").alias("E"))
        .toPandas()); fb["O_minus_E"] = fb["O"] - fb["E"]
save(fb.sort_values("O_minus_E", ascending=False), "sq3_fallback_records_by_product")
YBAR = n_us and nat["O"] / n_us


def oe_table(df, key):
    g = df.groupBy(key).agg(F.count(F.lit(1)).alias("n"), F.sum("y").alias("O"), F.sum("p_ext").alias("E"),
                            F.sum("v_ext").alias("V")).toPandas()
    g[key] = g[key].astype(str); g = g.set_index(key)
    g["raw_rate"] = g["O"] / g["n"]; g["oe"] = g["O"] / g["E"]; g["sampling_var"] = g["V"] / g["E"] ** 2
    return g


def dispersion(t):
    x, w, s2 = t["oe"].to_numpy(float), t["n"].to_numpy(float), t["sampling_var"].to_numpy(float)
    if len(x) < 2:
        return {"groups": len(x)}
    var = float(np.var(x, ddof=1)); samp = float(np.mean(s2))
    mw = np.average(x, weights=w); varw = float(np.average((x - mw) ** 2, weights=w)); sampw = float(np.average(s2, weights=w))
    return {"groups": len(x), "mean_oe": float(x.mean()), "median_oe": float(np.median(x)), "sd_oe": var ** .5,
            "mean_sampling_var": samp, "tau2": var - samp, "tau": max(var - samp, 0) ** .5,
            "sd_oe_weighted": varw ** .5, "tau_weighted": max(varw - sampw, 0) ** .5,
            "share_outside_95": float((np.abs(x - 1) > 1.96 * np.sqrt(s2)).mean()),
            "share_outside_998": float((np.abs(x - 1) > 3.09 * np.sqrt(s2)).mean())}

# %% [markdown]
# ## 7. SQ3: địa lý (50 bang + DC; PR riêng; GU, VI chỉ đếm; thiếu vị trí riêng)

# %%
UG = UE.filter("in_universe")
ST = oe_table(UG, "state_code"); ST = ST[ST["n"] >= G["state_min_n"]]
lo, hi = wilson(ST["O"], ST["n"]); ST["ci_low"], ST["ci_high"] = lo, hi
ST["std_rate"] = ST["oe"] * YBAR
save(ST.sort_values("oe", ascending=False).reset_index(), "sq3_state_oe")
sep = oe_table(UE.filter(F.col("state_code").isin(G["report_separately"] + G["count_only"]) | F.col("state_missing")), "state_code")
save(sep.reset_index(), "sq3_state_separate_lines")
if "NA" in sep.index:
    check("O/E nhóm thiếu bang", float(sep.loc["NA", "oe"]), "geography.missing_state_usable_oe")
print(sep[["n", "O", "raw_rate", "oe"]].round(3).to_string())
ds = dispersion(ST)
KEY.update({"states_universe": len(ST), "tau_state": ds.get("tau"), "sd_state_raw_rate": float(ST["raw_rate"].std(ddof=1)),
            "sd_state_std_rate": float(ST["std_rate"].std(ddof=1))})
check("số bang trong phạm vi", len(ST), "geography.universe_states")

msa_n = UG.filter(F.col("msa") != R["special_codes"]["msa_nonmetro"]).groupBy("msa").count()
MSA_DF = msa_n.filter(F.col("count") >= TH["msa"]["min_usable"]).select("msa").cache()
MSA_SET = sorted(r["msa"] for r in MSA_DF.collect())
KEY["msa_common"] = len(MSA_SET)
check("tập MSA chung", len(MSA_SET), "geography.msa_common_set")
MT = oe_table(UG.join(F.broadcast(MSA_DF), "msa"), "msa")
dm = dispersion(MT)
med_loan = (c.filter("orig and in_universe").groupBy("msa").agg(F.expr("percentile(loan_amount_num, 0.5)").alias("median_loan_originated"))
             .toPandas().set_index("msa"))
MT = MT.join(med_loan)
save(MT.sort_values("oe", ascending=False).reset_index(), "sq3_msa_oe")
KEY.update({"tau_msa": dm.get("tau"), "msa_share_outside_95": dm.get("share_outside_95")})
CT = oe_table(UG.filter("county_valid"), "county_code"); CT = CT[CT["n"] >= G["county_display_min_n"]]
KEY.update({"counties_valid_core": geo["cty_valid"], "counties_display": len(CT)})
save(CT.reset_index(), "sq3_county_oe_display")

UL = UG.filter(F.col("lmi").isNotNull()).withColumn("lmi_group", F.when(F.col("lmi"), "LMI").otherwise("other"))
LM = oe_table(UL, "lmi_group")
LMt = pd.concat([oe_table(UL.filter(F.col("type") == t_), "lmi_group").assign(loan_type=t_) for t_ in TYPE_ORDER])
lmi_k = pd.DataFrame([{**kitagawa(UL, "lmi_group", "LMI", "other"), "scope": "all"}] +
                     [{**kitagawa(UL.filter(F.col("type") == t_), "lmi_group", "LMI", "other"), "scope": t_} for t_ in ["conventional", "FHA", "VA"]])
save(LM.reset_index(), "sq3_lmi_oe"); save(LMt.reset_index(), "sq3_lmi_oe_by_type"); save(lmi_k, "sq3_lmi_kitagawa")
KEY.update({"oe_lmi": float(LM.loc["LMI", "oe"]), "oe_non_lmi": float(LM.loc["other", "oe"])})
la = (c.filter("orig and in_universe and lmi is not null").withColumn("lmi_group", F.when(F.col("lmi"), "LMI").otherwise("other"))
       .groupBy("lmi_group").agg(F.expr("percentile(loan_amount_num, 0.5)").alias("median_loan_originated")).toPandas())
save(la, "sq3_lmi_loan_size")
print(f"bang: {len(ST)} | τ {ds.get('tau', float('nan')):.3f} | MSA chung {len(MSA_SET)} | τ {dm.get('tau', float('nan')):.3f} | "
      f"O/E LMI {KEY['oe_lmi']:.3f} vs {KEY['oe_non_lmi']:.3f}")
grpc = F.when(F.col("state_missing"), "missing_state").otherwise("has_state")
mp = dec.groupBy(grpc.alias("group"), "purpose").count().toPandas().pivot(index="group", columns="purpose", values="count").fillna(0)
save(mp.div(mp.sum(axis=1), axis=0).reset_index(), "dq_missing_location_purpose_mix")
mt_ = (dec.groupBy(grpc.alias("group"), F.col("tract_ratio").isNull().alias("tract_missing"))
          .agg(F.count(F.lit(1)).alias("n"), F.avg("y").alias("denial_rate")).orderBy("group", "tract_missing").toPandas())
save(mt_, "dq_missing_location_by_tract")
print(mt_.round(3).to_string(index=False))

# %% [markdown]
# ## 8. SQ4: lender (broadcast join Transmittal Sheet, window function cho loại vay chiếm ưu thế và ngũ phân vị)

# %%
dl = dec.groupBy("lei").agg(F.count(F.lit(1)).alias("n_dec"), F.sum(F.col("usable").cast("long")).alias("n_us")) \
        .withColumn("us_share", F.col("n_us") / F.col("n_dec")).cache()
tot_dec = dl.agg(F.sum("n_dec")).first()[0]
big812 = dl.filter(F.col("n_dec") >= TH["lender"]["min_decisioned"])
L767_DF = big812.filter(F.col("us_share") >= TH["lender"]["min_usable_share"]).select("lei").cache()
n812, n767 = big812.count(), L767_DF.count()
KEY.update({"lenders_812": n812, "lenders_812_share": big812.agg(F.sum("n_dec")).first()[0] / tot_dec, "lenders_767": n767})
check("lender >= 500 decisioned", n812, "lenders.min500_decisioned_only")
check("lender phân tích", n767, "lenders.analysis_set")

ts = (spark.read.option("header", True).csv(TS_PATH).select(*R["source"]["transmittal_sheet"]["columns"]).dropDuplicates(["lei"])
      if TS_PATH else None)
UL7 = UE.join(F.broadcast(L767_DF), "lei")
lo_sp = UL7.groupBy("lei").agg(F.count(F.lit(1)).alias("n"), F.sum("y").alias("O"), F.sum("p_ext").alias("E"), F.sum("v_ext").alias("V"))
if ts is not None:
    lo_sp = lo_sp.join(F.broadcast(ts), "lei", "left").withColumn("agency", mapper(R["codes"]["agency_code"])[F.col("agency_code")])
LO = lo_sp.toPandas().set_index("lei")
LO["raw_rate"] = LO["O"] / LO["n"]; LO["oe"] = LO["O"] / LO["E"]; LO["sampling_var"] = LO["V"] / LO["E"] ** 2
dlen = dispersion(LO)
KEY.update({"tau_lender": dlen["tau"], "tau_lender_weighted": dlen["tau_weighted"], "lender_share_outside_95": dlen["share_outside_95"],
            "lender_share_outside_998": dlen["share_outside_998"], "lender_median_oe": dlen["median_oe"]})
print(f"lender: τ {dlen['tau']:.3f} | ngoài 95% {dlen['share_outside_95']:.1%} | ngoài 99.8% {dlen['share_outside_998']:.1%}")

wt = []
for code_, nm in [("1", "conventional"), ("2", "FHA")]:
    t_ = oe_table(UE.filter(F.col("loan_type") == code_), "lei"); t_ = t_[t_["n"] >= 500]
    d_ = dispersion(t_); d_["scope"] = f"within {nm}"; wt.append(d_)
    KEY[f"tau_lender_within_{nm}"] = d_.get("tau")
    KEY[f"spearman_lender_oe_main_vs_within_{nm}"] = spearman(LO["oe"].reindex(t_.index), t_["oe"])

# Định nghĩa từ chối thay thế: 812 lender, tỷ lệ thô; ngũ phân vị bằng ntile trên ORDER BY (tỷ lệ, lei)
apps = (c.groupBy("lei").pivot("action_taken", ["1", "2", "3", "4", "5"]).count().fillna(0)
         .withColumn("rate_main", F.col("3") / (F.col("1") + F.col("2") + F.col("3")))
         .withColumn("rate_alt", F.col("3") / (F.col("1") + F.col("2") + F.col("3") + F.col("4") + F.col("5")))
         .withColumn("wd_inc_share", (F.col("4") + F.col("5")) / (F.col("1") + F.col("2") + F.col("3") + F.col("4") + F.col("5")))).cache()
s812 = (apps.join(big812.select("lei"), "lei")
            .withColumn("q_main", F.ntile(5).over(Window.orderBy("rate_main", "lei")))
            .withColumn("q_alt", F.ntile(5).over(Window.orderBy("rate_alt", "lei"))).toPandas())
KEY.update({"denial_def_spearman": spearman(s812["rate_main"], s812["rate_alt"]),
            "denial_def_share_changing_quintile": float((s812["q_main"] != s812["q_alt"]).mean())})
print(f"định nghĩa từ chối: Spearman {KEY['denial_def_spearman']:.3f} | đổi ngũ phân vị {KEY['denial_def_share_changing_quintile']:.3f}")

# Độ nhạy
sens = []
for mn in TH["sensitivity"]["cell_min_n"]:
    sens.append({"variant": f"cell_min_n={mn}", **dispersion(oe_table(add_pext(U, mn).join(F.broadcast(L767_DF), "lei"), "lei"))})
ALLO = oe_table(UE, "lei").join(dl.toPandas().set_index("lei"))
for nd in TH["sensitivity"]["lender_min_decisioned"]:
    sens.append({"variant": f"n_dec>={nd}, share>={TH['lender']['min_usable_share']}",
                 **dispersion(ALLO[(ALLO["n_dec"] >= nd) & (ALLO["us_share"] >= TH["lender"]["min_usable_share"])])})
for sh in TH["sensitivity"]["lender_min_usable_share"]:
    sens.append({"variant": f"n_dec>={TH['lender']['min_decisioned']}, share>={sh}",
                 **dispersion(ALLO[(ALLO["n_dec"] >= TH["lender"]["min_decisioned"]) & (ALLO["us_share"] >= sh)])})
den_all = c.filter("denied").groupBy("lei").count().toPandas().set_index("lei")["count"]
LO["den_usable_share"] = LO["O"] / den_all.reindex(LO.index)
for thr in (0.5, 0.3):
    keep_ = ~((den_all.reindex(LO.index) >= 10) & (LO["den_usable_share"] < thr))
    sens.append({"variant": f"drop lenders with >=10 denials and <{int(thr * 100)}% usable", **dispersion(LO[keep_])})
sens = pd.DataFrame(sens + [dict(variant="within " + w["scope"].split()[-1], **{k: v for k, v in w.items() if k != "scope"}) for w in wt])
save(sens, "sq4_sensitivity")
print(sens[["variant", "groups", "tau", "tau_weighted"]].round(3).to_string(index=False))

# Giá: phân tán rate spread đã chuẩn hóa rủi ro giữa lender và giữa MSA (SQ4); định nghĩa giống bên Pandas (đề xuất rules 1.1)
SPREAD_MIN_GROUP_N = int(R["spread_dispersion"]["min_group_n"])
PSs = c.filter("pricing and usable").select("lei", "msa", "in_universe", "rs_num", *KEYS, "loan_type", "loan_purpose")
mx = PSs.groupBy(*EXTK).agg(F.count(F.lit(1)).alias("nx"), F.avg("rs_num").alias("mx"))
mf = PSs.groupBy(*KEYS, "loan_type").agg(F.avg("rs_num").alias("mf"))
PSs = (PSs.join(F.broadcast(mx), EXTK).join(F.broadcast(mf), KEYS + ["loan_type"])
          .withColumn("sres", F.col("rs_num") - F.when(F.col("nx") >= R["extended_cells"]["min_n"], F.col("mx")).otherwise(F.col("mf"))))


def spread_tau(df, key):
    g_ = df.groupBy(key).agg(F.count(F.lit(1)).alias("n"), F.avg("sres").alias("mean"), F.var_samp("sres").alias("var")).toPandas()
    g_ = g_[g_["n"] >= SPREAD_MIN_GROUP_N]
    v = float(g_["mean"].var(ddof=1)); s2 = float((g_["var"] / g_["n"]).mean())
    return {"groups": len(g_), "sd_mean": v ** .5, "tau": max(v - s2, 0) ** .5}


sp_l = spread_tau(PSs.join(F.broadcast(L767_DF), "lei"), "lei")
sp_m = spread_tau(PSs.filter("in_universe").join(F.broadcast(MSA_DF), "msa"), "msa")
KEY.update({"spread_tau_lender": sp_l["tau"], "spread_tau_msa": sp_m["tau"], "spread_groups_lender": sp_l["groups"],
            "spread_groups_msa": sp_m["groups"], "spread_sd_lender": sp_l["sd_mean"], "spread_sd_msa": sp_m["sd_mean"],
            "lenders_zero_denials": int((LO["O"] == 0).sum())})
save(pd.DataFrame([dict(group="lender", **sp_l), dict(group="msa", **sp_m)]), "sq4_spread_dispersion")
check("lender không có hồ sơ từ chối (trong 767)", KEY["lenders_zero_denials"], "lenders.zero_denials_in_767")
print(f"rate spread chuẩn hóa: τ lender {sp_l['tau']:.3f} điểm % ({sp_l['groups']} nhóm) | τ MSA {sp_m['tau']:.3f} ({sp_m['groups']} nhóm)")

# Độ nhạy: DTI 5 nhóm (rules 1.1, risk_cells.sensitivity_dti_bands)
KA = ["dti_band_alt", "cltv_band", "lti_band"]; EA = KA + ["loan_type", "loan_purpose"]
TA = add_pext(U.withColumn("dti_band_alt", band(F.col("dti_ordinal"), D[R["risk_cells"]["sensitivity_dti_bands"]])),
              R["extended_cells"]["min_n"], EA, KA).cache()
LOA = oe_table(TA.join(F.broadcast(L767_DF), "lei"), "lei")
kA = kitagawa(TA, "loan_type", "2", "1", keys=KA)
cA = TA.groupBy(*KA).count().agg(F.count(F.lit(1)).alias("k"), F.min("count").alias("mn")).first()
natA = TA.agg(F.sum("y").alias("O"), F.sum("p_ext").alias("E")).first()
KEY.update({"dti5_n_risk_cells": cA["k"], "dti5_smallest_cell_n": cA["mn"], "dti5_national_oe": natA["O"] / natA["E"],
            "dti5_tau_lender": dispersion(LOA)["tau"], "dti5_spearman_lender_oe": spearman(LO["oe"], LOA["oe"].reindex(LO.index))})
for k in ["gap", "composition", "rate", "rate_gap_at_a_mix", "rate_gap_at_b_mix", "cells_a_higher", "cells_a_lower"]:
    KEY[f"dti5_kitagawa_loan_type_2v1_{k}"] = kA[k]
d5 = {k: v for k, v in KEY.items() if k.startswith("dti5_")}
save(pd.Series(d5).rename("value").rename_axis("key").reset_index(), "sq4_dti5_sensitivity")
print("DTI 5 nhóm:", {k: round(float(v), 4) for k, v in d5.items()})
TA.unpersist()

LOnf = oe_table(UL7.filter(~F.col("fallback")), "lei")
d_oe = (LOnf["oe"] - LO["oe"].reindex(LOnf.index)).abs()
KEY.update({"fallback_robust_max_abs_change": float(d_oe.max()), "fallback_robust_spearman": spearman(LO["oe"].reindex(LOnf.index), LOnf["oe"])})
print(f"bỏ fallback: |ΔO/E| lớn nhất {KEY['fallback_robust_max_abs_change']:.4f} | Spearman {KEY['fallback_robust_spearman']:.4f}")


def dominant(col, names, out):
    w_ = Window.partitionBy("lei").orderBy(F.desc("cnt"), F.col(col))
    return (dec.groupBy("lei", col).agg(F.count(F.lit(1)).alias("cnt"))
               .withColumn("rk", F.row_number().over(w_)).filter("rk = 1").join(dl.select("lei", "n_dec"), "lei")
               .select("lei", mapper(names)[F.col(col)].alias(out), (F.col("cnt") / F.col("n_dec")).alias(out + "_share")))


dom = (dominant("loan_type", LT_NAME, "dominant_loan_type").join(dominant("loan_purpose", LP_NAME, "dominant_purpose"), "lei")
       .join(apps.select("lei", F.col("wd_inc_share").alias("withdrawn_incomplete_share"), F.col("rate_main").alias("denial_rate_main"),
                         F.col("rate_alt").alias("denial_rate_alternative")), "lei").join(F.broadcast(L767_DF), "lei").toPandas().set_index("lei"))
LO = LO.join(dom); LO["n_usable"] = LO["n"]
ex_cols = R["observed_expected"]["extremes_table"]["columns"]
srt = LO.reset_index().sort_values(["oe", "lei"])
k_ = R["observed_expected"]["extremes_table"]["n_each_side"]
extremes = pd.concat([srt.head(k_).assign(side="lowest"), srt.tail(k_).iloc[::-1].assign(side="highest")])
save(extremes[["side"] + [x for x in ex_cols if x in extremes.columns]], "sq4_lender_extremes")
save(LO.reset_index(), "sq4_lender_oe")
if "agency" in LO:
    save(LO.groupby("agency")["oe"].agg(n="size", median="median").reset_index(), "sq4_oe_by_agency")

# %% [markdown]
# ## 9. Mô hình hai chiều lender + MSA (Spark) và HHI
# - Thành phần liên thông: lan truyền nhãn nhỏ nhất trên đồ thị lender - MSA bằng join lặp.
# - Ước lượng: bảng ô (lender, MSA, n, S); mỗi vòng a = Σ(S − n·b)/Σn theo lender rồi b = Σ(S − n·a)/Σn theo MSA, 40 vòng,
#   `localCheckpoint` mỗi vòng để cắt lineage.
# - Split-half: `row_number()` trên Window(partitionBy lei, msa; orderBy h, row_id), half = (thứ hạng − 1) mod 2.

# %%
t0 = time.time()
SPLIT_SEED = 2024
assert f"row_id + {SPLIT_SEED}" in R["two_way_effects"]["split_half"]["hash"]
TW = (UE.filter("in_universe").join(F.broadcast(MSA_DF), "msa").join(F.broadcast(L767_DF), "lei")
        .select("row_id", "lei", "msa", (F.col("y") - F.col("p_ext")).alias("r")))
pairs = TW.select("lei", "msa").distinct().cache()
lab_l = pairs.select("lei").distinct().withColumn("lab", F.concat(F.lit("L"), F.col("lei")))
lab_m = pairs.select("msa").distinct().withColumn("lab", F.concat(F.lit("M"), F.col("msa")))
for it in range(100):
    new_m = (pairs.join(lab_l, "lei").groupBy("msa").agg(F.min("lab").alias("l1")).join(lab_m, "msa")
                  .select("msa", F.least("l1", "lab").alias("lab")))
    new_l = (pairs.join(new_m, "msa").groupBy("lei").agg(F.min("lab").alias("m1")).join(lab_l, "lei")
                  .select("lei", F.least("m1", "lab").alias("lab")))
    changed = (new_l.join(lab_l.withColumnRenamed("lab", "old"), "lei").filter(F.col("lab") != F.col("old")).count()
               + new_m.join(lab_m.withColumnRenamed("lab", "old"), "msa").filter(F.col("lab") != F.col("old")).count())
    lab_l, lab_m = new_l.localCheckpoint(), new_m.localCheckpoint()
    if changed == 0:
        break
comp_sizes = TW.join(lab_l, "lei").groupBy("lab").count().orderBy(F.desc("count")).collect()
big_lab = comp_sizes[0]["lab"]
CS = TW.join(lab_l.filter(F.col("lab") == big_lab).select("lei"), "lei").cache()
n_comp = lab_l.select("lab").union(lab_m.select("lab")).distinct().count()
KEY.update({"twoway_components": n_comp, "twoway_lenders": CS.select("lei").distinct().count(),
            "twoway_msas": CS.select("msa").distinct().count(), "twoway_records": CS.count()})
print(f"liên thông: {n_comp} thành phần, hội tụ sau {it + 1} vòng | tập lớn nhất: {KEY['twoway_lenders']} lender, {KEY['twoway_msas']} MSA")


def fit(cells, iters=R["two_way_effects"]["iterations"]):
    cells = cells.cache()
    b = cells.select("msa").distinct().withColumn("b", F.lit(0.0))
    for _ in range(iters):
        a = (cells.join(F.broadcast(b), "msa").groupBy("lei")
                  .agg((F.sum(F.col("S") - F.col("n") * F.col("b")) / F.sum("n")).alias("a"))).localCheckpoint()
        b = (cells.join(F.broadcast(a), "lei").groupBy("msa")
                  .agg((F.sum(F.col("S") - F.col("n") * F.col("a")) / F.sum("n")).alias("b"))).localCheckpoint()
    nl = cells.groupBy("lei").agg(F.sum("n").alias("n"))
    nm = cells.groupBy("msa").agg(F.sum("n").alias("n"))
    return a.join(nl, "lei").toPandas().set_index("lei"), b.join(nm, "msa").toPandas().set_index("msa")


def cell_agg(df):
    return df.groupBy("lei", "msa").agg(F.count(F.lit(1)).alias("n"), F.sum("r").alias("S"))


def wsd(x, w):
    m = np.average(x, weights=w); return float(np.sqrt(np.average((x - m) ** 2, weights=w)))


def wcov(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w))


A_, B_ = fit(cell_agg(CS))
h = F.pmod((F.col("row_id") + F.lit(SPLIT_SEED)) * F.lit(S_["hash_multiplier"]).cast("long"), F.lit(S_["hash_modulus"]).cast("long"))
CSH = CS.withColumn("h", h).withColumn("half", (F.row_number().over(Window.partitionBy("lei", "msa").orderBy("h", "row_id")) - 1) % 2)
A0, B0 = fit(cell_agg(CSH.filter("half = 0")))
A1, B1 = fit(cell_agg(CSH.filter("half = 1")))
la_ = A_.join(A0["a"].rename("a0"), how="inner").join(A1["a"].rename("a1"), how="inner")
mb_ = B_.join(B0["b"].rename("b0"), how="inner").join(B1["b"].rename("b1"), how="inner")
mpl = CS.groupBy("lei").agg(F.countDistinct("msa").alias("k")).toPandas()["k"]
KEY.update({
    "twoway_lender_full_sd": wsd(A_["a"], A_["n"]), "twoway_msa_full_sd": wsd(B_["b"], B_["n"]),
    "twoway_lender_signal_sd": max(wcov(la_["a0"], la_["a1"], la_["n"]), 0) ** .5,
    "twoway_msa_signal_sd": max(wcov(mb_["b0"], mb_["b1"], mb_["n"]), 0) ** .5,
    "twoway_median_msas_per_lender": float(mpl.median()), "twoway_share_lenders_le2_msas": float((mpl <= 2).mean()),
    "twoway_lenders_both_halves": len(la_), "twoway_msas_both_halves": len(mb_)})
tw_out = {k: v for k, v in KEY.items() if k.startswith("twoway_")}
pd.Series(tw_out).to_csv(f"{RES}/sq4_two_way.csv", header=["value"])
A_.reset_index().rename(columns={"a": "effect"}).to_csv(f"{RES}/sq4_two_way_lender_effects.csv", index=False)
B_.reset_index().rename(columns={"b": "effect"}).to_csv(f"{RES}/sq4_two_way_msa_effects.csv", index=False)
print(json.dumps({k: (round(v * 100, 3) if "sd" in k else v) for k, v in tw_out.items()}, indent=1), f"| {time.time() - t0:.0f}s")

og = c.filter("orig and in_universe").join(F.broadcast(MSA_DF), "msa")
cnt_ = og.groupBy("msa", "lei").agg(F.count(F.lit(1)).alias("n"))
hhi = (cnt_.withColumn("share", F.col("n") / F.sum("n").over(Window.partitionBy("msa")))
           .groupBy("msa").agg((F.sum(F.col("share") ** 2) * 10000).alias("hhi")).toPandas().set_index("msa"))
H = hhi.join(MT["oe"])
save(H.reset_index(), "sq4_hhi_msa")
KEY.update({"hhi_median": float(H["hhi"].median()), "hhi_p10": float(H["hhi"].quantile(.1)), "hhi_p90": float(H["hhi"].quantile(.9)),
            "hhi_spearman_oe": spearman(H["hhi"], H["oe"]), "hhi_msas": len(H)})
print(f"HHI: trung vị {KEY['hhi_median']:.0f} | Spearman với O/E {KEY['hhi_spearman_oe']:.3f}")

# %% [markdown]
# ## 10. Sáu đầu ra cho kiểm tra tương đương (cùng định dạng với Pandas)
# Trung vị tính hai cách: `percentile` (chính xác) và `percentile_approx` (xấp xỉ) để đo sai số xấp xỉ.

# %%
steps = [("all_records", n_all)] + [(f["step"], step_counts[f["step"]]) for f in FILT] + [
    ("open_end_exempt_removed", open_end_exempt_removed), ("decisioned", n_dec), ("usable", n_us),
    ("pricing", n_pricing), ("pricing_usable", n_pricing_u)]
save(pd.DataFrame(steps, columns=["step", "rows"]), "eq_1_filter_counts")
save(dq, "eq_2_special_codes")
save(fprof[["field"] + [x if x else "(empty)" for x in FP["count_codes"]] + ["numeric"]], "eq_2b_field_profile")
a3 = pd.DataFrame(sorted(act_counts.items()), columns=["action_taken", "n"]); a3["share"] = a3["n"] / n_all
save(a3, "eq_3_action_taken")
rows4 = []
for var, src, df_, smp in [("income", "income_num", dec, "decisioned"), ("loan_amount", "loan_amount_num", dec, "decisioned"),
                           ("rate_spread", "rs_num", c.filter("pricing"), "pricing")]:
    t_ = (df_.filter(F.col(src).isNotNull()).groupBy("loan_type")
             .agg(F.count(F.lit(1)).alias("n"), F.avg(src).alias("mean"), F.expr(f"percentile({src}, 0.5)").alias("median_exact"),
                  F.expr(f"percentile_approx({src}, 0.5)").alias("median_approx")).toPandas())
    t_.insert(0, "variable", var); t_.insert(0, "sample", smp); rows4.append(t_)
save(pd.concat(rows4)[["sample", "variable", "loan_type", "n", "mean", "median_exact", "median_approx"]].sort_values(["sample", "variable", "loan_type"]),
     "eq_4_amounts_by_loan_type")
rc = (c.filter("orig").groupBy("loan_type").agg(F.sum(F.col("rs_num").isNotNull().cast("long")).alias("numeric"),
                                               F.sum((F.col("rate_spread") == "NA").cast("long")).alias("na"),
                                               F.sum((F.col("rate_spread") == "Exempt").cast("long")).alias("exempt"),
                                               F.sum(F.col("rs_out_of_range").cast("long")).alias("out_of_range"),
                                               F.sum(F.col("rs_valid").cast("long")).alias("valid")).orderBy("loan_type").toPandas())
save(rc, "eq_4b_rate_spread_counts")
s5 = (dec.groupBy("state_code").agg(F.count(F.lit(1)).alias("n_decisioned"), F.sum(F.col("denied").cast("long")).alias("n_denied"))
         .withColumn("denial_rate", F.col("n_denied") / F.col("n_decisioned")).orderBy("state_code").toPandas())
save(s5, "eq_5_denial_by_state")
oe_l = UE.groupBy("lei").agg(F.sum("y").alias("O"), F.sum("p_ext").alias("E")).withColumn("oe", F.col("O") / F.col("E"))
t6 = (dec.groupBy("lei").agg(F.count(F.lit(1)).alias("n_decisioned"), F.sum(F.col("denied").cast("long")).alias("n_denied"),
                             F.avg("income_num").alias("mean_income"), F.avg("loan_amount_num").alias("mean_loan_amount"),
                             F.sum(F.col("usable").cast("long")).alias("n_usable"))
         .withColumn("rank", F.row_number().over(Window.orderBy(F.desc("n_decisioned"), "lei"))).filter("rank <= 50")
         .withColumn("denial_rate", F.col("n_denied") / F.col("n_decisioned")).join(oe_l, "lei", "left").orderBy("rank").toPandas())
save(t6[["rank", "lei", "n_decisioned", "n_denied", "mean_income", "mean_loan_amount", "n_usable", "denial_rate", "O", "E", "oe"]],
     "eq_6_top50_lenders")
print("đã ghi eq_1 .. eq_6")

# %% [markdown]
# ## 11. Dự đoán, phần của Spark (rules mục 20)
# - Fold theo lender: `conv(substring(sha2(lei, 256), 1, 8), 16, 10) % 5`, phải trùng với fold bên Pandas.
# - Logistic regression bằng MLlib (cùng biến với bên Pandas), cross-fitting 5 fold, AUC trên dự đoán out-of-fold.
# - O/E theo lender dùng xác suất mô hình (MLlib, và của Pandas nếu có file `pred_oof.parquet`), so thứ hạng với O/E theo ô.

# %%
t0 = time.time()
PR = R["prediction"]
K = PR["folds"]["k"]
folds = UE.select("lei").distinct().withColumn("fold", (F.conv(F.substring(F.sha2(F.col("lei"), 256), 1, 8), 16, 10).cast("long") % K).cast("int"))
fpd = folds.toPandas().set_index("lei")["fold"]
py_fold = pd.Series({l: int(hashlib.sha256(l.encode()).hexdigest()[:8], 16) % K for l in fpd.index})
KEY["pred_fold_mismatch_vs_python"] = int((fpd != py_fold.reindex(fpd.index)).sum())
print("lei có fold khác với hashlib (phải bằng 0):", KEY["pred_fold_mismatch_vs_python"])
pd_folds = glob.glob(f"{OUTROOT}/results/pandas/pred_folds_by_lei.csv") + glob.glob("/kaggle/input/**/pred_folds_by_lei.csv", recursive=True)
if pd_folds:
    pf_ = pd.read_csv(pd_folds[0], dtype={"lei": str}).set_index("lei")["fold"]
    KEY["pred_fold_mismatch_vs_pandas"] = int((fpd.reindex(pf_.index) != pf_).sum())
    print("lei có fold khác bên Pandas (phải bằng 0):", KEY["pred_fold_mismatch_vs_pandas"])

LOCMP = LO[["oe"]].rename(columns={"oe": "oe_cells"})
if RUN_MLLIB:
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.ml.classification import LogisticRegression as SparkLR
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.functions import vector_to_array

    M = (UE.join(F.broadcast(folds), "lei")
           .withColumn("dti_s", F.col("dti_ordinal").cast("int").cast("string"))
           .withColumn("cltv_s", F.least(F.greatest(F.col("cltv"), F.lit(0.0)), F.lit(150.0)) / 100)
           .withColumn("log_lti", F.log(F.least(F.greatest(F.col("lti"), F.lit(0.05)), F.lit(20.0))))
           .withColumn("log_amt", F.log("loan_amount_num"))
           .withColumn("log_inc", F.log(F.greatest(F.col("income_num"), F.lit(1.0))))
           .select("row_id", "lei", "fold", "y", "dti_s", "cltv_band", "loan_type", "loan_purpose", "cltv_s", "log_lti", "log_amt", "log_inc"))
    NUM = ["cltv_s", "log_lti", "log_amt", "log_inc"]
    stats = M.agg(*[F.avg(x).alias(f"m_{x}") for x in NUM], *[F.stddev_samp(x).alias(f"s_{x}") for x in NUM]).first()
    for x in NUM:
        M = M.withColumn(x, (F.col(x) - stats[f"m_{x}"]) / stats[f"s_{x}"])
    CATS = ["dti_s", "cltv_band", "loan_type", "loan_purpose"]
    stages = [StringIndexer(inputCol=x, outputCol=x + "_i", stringOrderType="alphabetAsc") for x in CATS]
    stages += [OneHotEncoder(inputCols=[x + "_i" for x in CATS], outputCols=[x + "_v" for x in CATS])]
    stages += [VectorAssembler(inputCols=NUM + [x + "_v" for x in CATS], outputCol="features"),
               SparkLR(featuresCol="features", labelCol="y", maxIter=200, regParam=0.0, tol=1e-7)]
    M = M.cache()
    preds = []
    for k in range(K):
        model = Pipeline(stages=stages).fit(M.filter(F.col("fold") != k))
        preds.append(model.transform(M.filter(F.col("fold") == k))
                     .select("row_id", "lei", "fold", "y", vector_to_array("probability")[1].alias("p_lr_mllib")))
        print(f"  fold {k}: {time.time() - t0:.0f}s")
    OOF = preds[0]
    for p_ in preds[1:]:
        OOF = OOF.unionByName(p_)
    OOF.write.mode("overwrite").parquet(f"{RES}/pred_oof_mllib.parquet")
    OOF = spark.read.parquet(f"{RES}/pred_oof_mllib.parquet")
    ev = BinaryClassificationEvaluator(rawPredictionCol="p_lr_mllib", labelCol="y", numBins=0)
    KEY["pred_mllib_lr_roc_auc"] = ev.setMetricName("areaUnderROC").evaluate(OOF)
    KEY["pred_mllib_lr_pr_auc"] = ev.setMetricName("areaUnderPR").evaluate(OOF)
    cal = OOF.agg(F.avg("p_lr_mllib").alias("p"), F.avg("y").alias("y")).first()
    KEY["pred_mllib_lr_calibration_in_the_large"] = cal["p"] - cal["y"]
    g_ = OOF.join(F.broadcast(L767_DF), "lei").groupBy("lei").agg(F.sum("y").alias("O"), F.sum("p_lr_mllib").alias("E")).toPandas().set_index("lei")
    LOCMP["oe_lr_mllib"] = (g_["O"] / g_["E"]).reindex(LOCMP.index)
    KEY["pred_spearman_cells_vs_lr_mllib"] = spearman(LOCMP["oe_cells"], LOCMP["oe_lr_mllib"])
    print({k: round(v, 4) for k, v in KEY.items() if k.startswith("pred_")})

pd_oof = glob.glob(f"{OUTROOT}/results/pandas/pred_oof.parquet") + glob.glob("/kaggle/input/**/pred_oof.parquet", recursive=True)
if pd_oof:
    PO = spark.read.parquet(pd_oof[0]).select("row_id", "p_lr", "p_gbt")
    JO = UE.select("row_id", "lei", "y").join(PO, "row_id", "inner")
    KEY["pred_pandas_oof_rows_joined"] = JO.count()
    g2 = JO.join(F.broadcast(L767_DF), "lei").groupBy("lei").agg(F.sum("y").alias("O"), F.sum("p_lr").alias("E_lr"), F.sum("p_gbt").alias("E_gbt")).toPandas().set_index("lei")
    for m_ in ["lr", "gbt"]:
        LOCMP[f"oe_{m_}"] = (g2["O"] / g2[f"E_{m_}"]).reindex(LOCMP.index)
        KEY[f"pred_spearman_cells_vs_{m_}"] = spearman(LOCMP["oe_cells"], LOCMP[f"oe_{m_}"])
    if RUN_MLLIB:
        cmpp = OOF.join(PO, "row_id").agg(F.corr("p_lr_mllib", "p_lr").alias("r"), F.max(F.abs(F.col("p_lr_mllib") - F.col("p_lr"))).alias("mx")).first()
        KEY["pred_corr_mllib_vs_sklearn_lr"], KEY["pred_maxdiff_mllib_vs_sklearn_lr"] = cmpp["r"], cmpp["mx"]
save(LOCMP.reset_index(), "pred_lender_oe_comparison_spark")
print(f"dự đoán (Spark) xong | {time.time() - t0:.0f}s")

# %% [markdown]
# ## 12. Tổng kết

# %%
def clean_json(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


KEY["runtime_s"] = round(time.time() - T_START, 1)
json.dump({k: clean_json(v) for k, v in KEY.items()}, open(f"{RES}/key_numbers.json", "w"), indent=1, ensure_ascii=False)
pd.DataFrame(CHECKS).to_csv(f"{RES}/checks_vs_expected.csv", index=False)
bad = [x for x in CHECKS if x["match"] is False]
print(f"xong sau {KEY['runtime_s']:.0f}s | {len(CHECKS)} kiểm tra, {len(bad)} lệch so với expected.yaml")
for x in bad:
    print("  LỆCH:", x["check"], x["value"], "kỳ vọng", x["expected"])
