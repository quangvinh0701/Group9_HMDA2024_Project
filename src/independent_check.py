# %% [markdown]
# # HMDA 2024: kiểm tra độc lập bằng DuckDB (cài đặt thứ ba)
#
# Hai pipeline Pandas và PySpark cùng đọc `rules.yaml`, nên việc chúng khớp nhau chỉ chứng tỏ hai cài đặt nhất quán,
# chưa chứng tỏ luật được dịch đúng. Script này viết lại các phép tính chính bằng **DuckDB SQL**, với định nghĩa lấy
# trực tiếp từ tài liệu FFIEC và Phụ lục A của báo cáo (gõ thẳng trong code, **không đọc rules.yaml**), rồi:
#
# 1. đếm số hồ sơ và tổng `loan_amount` theo nhiều lát cắt của toàn file để so với HMDA Data Browser;
# 2. tính lại các con số chính của mẫu nghiên cứu (lọc mẫu, kết quả theo loại vay, tỷ lệ từ chối theo bang và theo DTI,
#    $O/E$ của từng bang và từng lender) và so với kết quả của pipeline Pandas.
#
# Usage: `python independent_check.py [RAW_PARQUET] [PANDAS_RESULTS_DIR] [OUT_DIR]`
# (mặc định tự tìm trong /kaggle/input hoặc thư mục HMDA_INPUT).

# %%
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # Kaggle thường có sẵn duckdb; nếu không thì cài
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb"], check=True)
    import duckdb


def find(pattern, roots):
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, "**", pattern), recursive=True))
        if hits:
            return hits[0]
    return None


ARGS = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]
ROOTS = [os.environ["HMDA_INPUT"]] if os.environ.get("HMDA_INPUT") else ["/kaggle/input", "."]
RAW = ARGS[0] if len(ARGS) > 0 else find("hmda_2024_raw.parquet", ROOTS)
PD_DIR = ARGS[1] if len(ARGS) > 1 else os.path.dirname(find("results/pandas/key_numbers.json", ROOTS) or "")
OUTROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("HMDA_OUT", "output")
OUT = ARGS[2] if len(ARGS) > 2 else f"{OUTROOT}/results/independent"
os.makedirs(OUT, exist_ok=True)
print("raw:", RAW, "\npandas results:", PD_DIR or "(không có)", "\nout:", OUT)

con = duckdb.connect()
con.execute(f"SET threads TO {os.cpu_count() or 2}")
con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{RAW}')")
t0 = time.time()

# %% [markdown]
# ## 1. Lát cắt của toàn file để so với HMDA Data Browser
# Data Browser trả về số hồ sơ (`count`) và tổng `loan_amount` (`sum`) cho mỗi tổ hợp `action_taken` × một chiều.

# %%
DIMS = {"loan_types": "loan_type", "loan_purposes": "loan_purpose", "lien_statuses": "lien_status",
        "construction_methods": "construction_method", "states": "state_code"}
sl = []
for dim, col in DIMS.items():
    q = con.execute(f"""
        SELECT '{dim}' AS dimension, {col} AS value, action_taken, COUNT(*) AS count,
               SUM(TRY_CAST(loan_amount AS DOUBLE)) AS sum_loan_amount
        FROM raw GROUP BY ALL""").df()
    sl.append(q)
slices = pd.concat(sl, ignore_index=True).sort_values(["dimension", "value", "action_taken"])
slices.to_csv(f"{OUT}/db_slices.csv", index=False)
print("lát cắt:", len(slices), "dòng |", f"{time.time() - t0:.0f}s")

# %% [markdown]
# ## 2. Mẫu nghiên cứu, viết lại từ định nghĩa
# Core: hồ sơ (action_taken 1-5), thế chấp hạng nhất, closed-end, không vì mục đích kinh doanh, không reverse,
# nhà site-built 1-4 căn, nhà ở chính. Decisioned: action_taken 1-3. Usable: DTI thuộc 19 giá trị công bố,
# 0 < CLTV <= 250, income > 0. Pricing: giải ngân, rate spread trong [-8, 15].

# %%
DTI_ORD = {"<20": 1, "20-<30": 2, "30-<36": 3, **{str(v): v - 32 for v in range(36, 50)}, "50-60": 18, ">60": 19}
dti_case = "CASE " + " ".join(f"WHEN d = '{k}' THEN {v}" for k, v in DTI_ORD.items()) + " END"
LTI_Q1, LTI_Q2 = 2.3550724637681157, 3.4036144578313254     # tam phân vị cố định (Phụ lục A)

con.execute(f"""
CREATE TABLE steps AS
SELECT row_id, lei, state_code, action_taken, loan_type, loan_purpose,
       (action_taken IN ('1','2','3','4','5')) AS s1,
       (lien_status = '1') AS s2, (open_end_line_of_credit = '2') AS s3,
       (business_or_commercial_purpose = '2') AS s4, (reverse_mortgage = '2') AS s5,
       (derived_dwelling_category = 'Single Family (1-4 Units):Site-Built') AS s6, (occupancy_type = '1') AS s7,
       debt_to_income_ratio, combined_loan_to_value_ratio, income, loan_amount, rate_spread,
       tract_to_msa_income_percentage
FROM raw""")
funnel = con.execute("""
SELECT COUNT(*) AS all_records, SUM(s1::INT) AS applications, SUM((s1 AND s2)::INT) AS first_lien,
       SUM((s1 AND s2 AND s3)::INT) AS closed_end, SUM((s1 AND s2 AND s3 AND s4)::INT) AS non_business,
       SUM((s1 AND s2 AND s3 AND s4 AND s5)::INT) AS non_reverse,
       SUM((s1 AND s2 AND s3 AND s4 AND s5 AND s6)::INT) AS site_built_1_4_unit,
       SUM((s1 AND s2 AND s3 AND s4 AND s5 AND s6 AND s7)::INT) AS principal_residence
FROM steps""").df().iloc[0].to_dict()

con.execute(f"""
CREATE TABLE core AS
SELECT row_id, lei, state_code, action_taken, loan_type, loan_purpose,
       action_taken IN ('1','2','3') AS decisioned, action_taken = '3' AS denied, action_taken = '1' AS orig,
       (SELECT {dti_case} FROM (SELECT replace(replace(debt_to_income_ratio, '%', ''), ' ', '') AS d)) AS dti,
       TRY_CAST(combined_loan_to_value_ratio AS DOUBLE) AS cltv, TRY_CAST(income AS DOUBLE) AS inc,
       TRY_CAST(loan_amount AS DOUBLE) AS amt, TRY_CAST(rate_spread AS DOUBLE) AS rs,
       NULLIF(TRY_CAST(tract_to_msa_income_percentage AS DOUBLE), 0) AS tract_pct
FROM steps WHERE s1 AND s2 AND s3 AND s4 AND s5 AND s6 AND s7""")
con.execute("""
CREATE TABLE usable AS
SELECT *, denied::INT AS y,
       CASE WHEN dti <= 2 THEN '<30' WHEN dti = 3 THEN '30-<36' WHEN dti <= 10 THEN '36-42'
            WHEN dti <= 17 THEN '43-49' WHEN dti = 18 THEN '50-60' ELSE '>60' END AS db,
       CASE WHEN cltv <= 80 THEN '<=80' WHEN cltv <= 90 THEN '80-90' WHEN cltv <= 95 THEN '90-95'
            WHEN cltv <= 100 THEN '95-100' ELSE '>100' END AS cb,
       amt / (1000 * inc) AS lti
FROM core WHERE decisioned AND dti IS NOT NULL AND cltv > 0 AND cltv <= 250 AND inc > 0""")
con.execute(f"""ALTER TABLE usable ADD COLUMN lb VARCHAR;
UPDATE usable SET lb = CASE WHEN lti <= {LTI_Q1} THEN 'T1' WHEN lti <= {LTI_Q2} THEN 'T2' ELSE 'T3' END""")
counts = con.execute("""
SELECT (SELECT SUM(decisioned::INT) FROM core) AS decisioned, (SELECT COUNT(*) FROM usable) AS usable,
       (SELECT COUNT(*) FROM core WHERE orig AND rs BETWEEN -8 AND 15) AS pricing""").df().iloc[0].to_dict()
funnel.update(counts)
print("đếm mẫu:", {k: int(v) for k, v in funnel.items()}, f"| {time.time() - t0:.0f}s")

# %% [markdown]
# ## 3. Các bảng để so với Pandas

# %%
TYPE = "CASE loan_type WHEN '1' THEN 'conventional' WHEN '2' THEN 'FHA' WHEN '3' THEN 'VA' WHEN '4' THEN 'USDA' END"
T = {}
T["outcomes_by_type"] = con.execute(f"""
SELECT {TYPE} AS type, COUNT(*) AS n_applications, AVG((action_taken='1')::INT) AS originated,
       AVG((action_taken='3')::INT) AS denied, AVG((action_taken='4')::INT) AS withdrawn
FROM core GROUP BY 1""").df()
T["loan_originated_by_type"] = con.execute(f"""
SELECT {TYPE} AS type, COUNT(amt) AS n, AVG(amt) AS mean, quantile_cont(amt, 0.5) AS median
FROM core WHERE orig GROUP BY 1""").df()
T["rate_spread_by_type"] = con.execute(f"""
SELECT {TYPE} AS type, COUNT(*) AS n, AVG(rs) AS mean, quantile_cont(rs, 0.5) AS median
FROM core WHERE orig AND rs BETWEEN -8 AND 15 GROUP BY 1""").df()
T["denial_by_state"] = con.execute("""
SELECT state_code, COUNT(*) AS n_decisioned, SUM(denied::INT) AS n_denied FROM core WHERE decisioned GROUP BY 1""").df()
T["denial_by_dti_19"] = con.execute("""
SELECT dti AS dti_ordinal, COUNT(*) AS n, AVG(y) AS denial_rate FROM usable GROUP BY 1""").df()
T["denial_type_x_purpose"] = con.execute(f"""
SELECT {TYPE} AS type, loan_purpose, COUNT(*) AS n_decisioned, SUM(denied::INT) AS n_denied
FROM core WHERE decisioned GROUP BY ALL""").df()

# xác suất kỳ vọng: tỷ lệ quốc gia của ô mở rộng (ô rủi ro × loại vay × mục đích), ô < 30 hồ sơ dùng ô rủi ro
con.execute("""
CREATE TABLE pe AS
WITH ext AS (SELECT db, cb, lb, loan_type, loan_purpose, COUNT(*) AS n_e, AVG(y) AS r_e FROM usable GROUP BY ALL),
     rc AS (SELECT db, cb, lb, AVG(y) AS r_c FROM usable GROUP BY ALL)
SELECT u.*, CASE WHEN ext.n_e >= 30 THEN ext.r_e ELSE rc.r_c END AS p
FROM usable u JOIN ext USING (db, cb, lb, loan_type, loan_purpose) JOIN rc USING (db, cb, lb)""")
T["oe_state"] = con.execute("SELECT state_code, COUNT(*) AS n, SUM(y) AS O, SUM(p) AS E FROM pe GROUP BY 1").df()
T["oe_lender"] = con.execute("SELECT lei, COUNT(*) AS n, SUM(y) AS O, SUM(p) AS E FROM pe GROUP BY 1").df()
T["oe_national"] = con.execute("SELECT SUM(y) / SUM(p) AS oe FROM pe").df()
# số liệu để quy đổi kết quả ra đô la trong phần diễn giải tài chính
MAG = con.execute("""
SELECT (SELECT quantile_cont(amt, 0.5) FROM core WHERE orig AND rs BETWEEN -8 AND 15 AND loan_type = '1' AND loan_purpose = '32')
           AS median_loan_conv_cashout,
       (SELECT quantile_cont(amt, 0.5) FROM core WHERE orig AND rs BETWEEN -8 AND 15 AND loan_type = '1' AND loan_purpose = '1')
           AS median_loan_conv_purchase,
       (SELECT quantile_cont(inc, 0.5) FROM usable WHERE loan_type = '1' AND dti = 17) AS median_income_conv_dti49,
       (SELECT COUNT(*) FROM usable WHERE loan_type = '1' AND dti = 17) AS n_conv_dti49""").df().iloc[0].to_dict()
json.dump({k: float(v) for k, v in MAG.items()}, open(f"{OUT}/ind_magnitudes.json", "w"), indent=1)
print("quy đổi:", MAG)
for k, v in T.items():
    v.to_csv(f"{OUT}/ind_{k}.csv", index=False)
json.dump({k: int(v) for k, v in funnel.items()}, open(f"{OUT}/ind_funnel.json", "w"), indent=1)
print("xong các bảng |", f"{time.time() - t0:.0f}s")

# %% [markdown]
# ## 4. So với kết quả của pipeline Pandas
# Đếm phải khớp tuyệt đối; trung bình, trung vị, tỷ lệ và $O/E$ so theo sai lệch tương đối $\le 10^{-9}$.

# %%
rows = []


def add(quantity, key, ind, pdv, exact):
    ind, pdv = float(ind), float(pdv)
    d = 0.0 if ind == pdv else abs(ind - pdv) / max(abs(ind), abs(pdv))
    rows.append({"quantity": quantity, "key": key, "independent": ind, "pandas": pdv, "rel_diff": d,
                 "agree": (ind == pdv) if exact else d <= 1e-9})


if PD_DIR and os.path.exists(os.path.join(PD_DIR, "eq_1_filter_counts.csv")):
    P = lambda n: pd.read_csv(os.path.join(PD_DIR, f"{n}.csv"), keep_default_na=False, na_values=[""])
    f1 = P("eq_1_filter_counts").set_index("step")["rows"]
    for k, v in funnel.items():
        if k in f1.index:
            add("sample sizes", k, v, f1[k], True)
    a = T["outcomes_by_type"].set_index("type"); b = P("sq1_outcomes_by_type").set_index("type")
    for t in a.index:
        add("applications by loan type", t, a.at[t, "n_applications"], b.at[t, "n_applications"], True)
        for col in ["originated", "denied", "withdrawn"]:
            add(f"share {col} by loan type", t, a.at[t, col], b.at[t, col], False)
    for name, pdname in [("loan_originated_by_type", "sq1_loan_amount_originated_by_type"), ("rate_spread_by_type", "sq2_rate_spread_by_type")]:
        a = T[name].set_index("type"); b = P(pdname).set_index("type")
        for t in a.index:
            add(f"{name}: n", t, a.at[t, "n"], b.at[t, "n"], True)
            add(f"{name}: mean", t, a.at[t, "mean"], b.at[t, "mean"], False)
            add(f"{name}: median", t, a.at[t, "median"], b.at[t, "median"], False)
    a = T["denial_by_state"].set_index("state_code"); b = P("eq_5_denial_by_state").set_index("state_code")
    for s in a.index:
        add("decisioned by state", s, a.at[s, "n_decisioned"], b.at[s, "n_decisioned"], True)
        add("denied by state", s, a.at[s, "n_denied"], b.at[s, "n_denied"], True)
    a = T["denial_by_dti_19"].set_index("dti_ordinal"); b = P("sq2_denial_by_dti_19"); b = b.set_index(b["dti_ordinal"].astype(int))
    for k in a.index:
        add("usable by DTI value", int(k), a.at[k, "n"], b.at[int(k), "n"], True)
        add("denial rate by DTI value", int(k), a.at[k, "denial_rate"], b.at[int(k), "denial_rate"], False)
    for name, pdname, key in [("oe_state", "sq3_state_oe", "state_code"), ("oe_lender", "sq4_lender_oe", "lei")]:
        a = T[name].set_index(key); b = P(pdname).set_index(key)
        for k in b.index:
            if k in a.index:
                add(f"{name}: n", k, a.at[k, "n"], b.at[k, "n"], True)
                add(f"{name}: O", k, a.at[k, "O"], b.at[k, "O"], True)
                add(f"{name}: E", k, a.at[k, "E"], b.at[k, "E"], False)
    kn = json.load(open(os.path.join(PD_DIR, "key_numbers.json")))
    if "national_oe" in kn:
        add("national O/E", "all", T["oe_national"]["oe"].iloc[0], kn["national_oe"], False)
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/independent_vs_pandas.csv", index=False)
    S = R.groupby("quantity", sort=False).agg(values=("agree", "size"), agree=("agree", "all"), max_rel_diff=("rel_diff", "max")).reset_index()
    S.to_csv(f"{OUT}/independent_vs_pandas_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))
    print(f"\n{len(R)} giá trị; {int((~R['agree']).sum())} lệch;", "Tất cả khớp" if R["agree"].all() else "CÓ LỆCH")
    if not R["agree"].all():
        print(R[~R["agree"]].head(20).to_string(index=False))
print(f"tổng thời gian {time.time() - t0:.0f}s")
