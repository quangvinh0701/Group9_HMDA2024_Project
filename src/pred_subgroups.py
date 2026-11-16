# %% [markdown]
# # HMDA 2024: đánh giá mô hình dự báo theo từng nhóm hồ sơ
#
# Dùng dự đoán out-of-fold mà notebook Pandas ghi ra (`pred_oof.parquet`, fold theo lender) và kiểm tra xem mô hình
# có dự đoán đều tay giữa các nhóm hay không: loại vay, mục đích vay, tract LMI và băng DTI. Với mỗi nhóm: số hồ sơ,
# tỷ lệ từ chối thực tế, trung bình dự đoán (CITL = dự đoán - thực tế), ROC-AUC và Brier.
# Các đặc điểm nhân khẩu học không dùng ở đây (xem hạn chế (vii) của báo cáo).
#
# Usage: `python pred_subgroups.py [PRED_OOF_PARQUET] [RAW_PARQUET] [OUT_DIR]`

# %%
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import duckdb
except ImportError:
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
OOF = ARGS[0] if len(ARGS) > 0 else find("pred_oof.parquet", ROOTS)
RAW = ARGS[1] if len(ARGS) > 1 else find("hmda_2024_raw.parquet", ROOTS)
OUTROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("HMDA_OUT", "output")
OUT = ARGS[2] if len(ARGS) > 2 else f"{OUTROOT}/results/predictive"
os.makedirs(OUT, exist_ok=True)
print("oof:", OOF, "\nraw:", RAW)

# %%
con = duckdb.connect()
d = con.execute(f"""
SELECT o.y, o.p_lr, o.p_gbt,
       CASE r.loan_type WHEN '1' THEN 'conventional' WHEN '2' THEN 'FHA' WHEN '3' THEN 'VA' WHEN '4' THEN 'USDA' END AS loan_type,
       CASE r.loan_purpose WHEN '1' THEN 'purchase' WHEN '2' THEN 'home improvement' WHEN '31' THEN 'refinance'
            WHEN '32' THEN 'cash-out refinance' WHEN '4' THEN 'other' ELSE 'not applicable' END AS purpose,
       CASE WHEN NULLIF(TRY_CAST(r.tract_to_msa_income_percentage AS DOUBLE), 0) IS NULL THEN 'tract unknown'
            WHEN TRY_CAST(r.tract_to_msa_income_percentage AS DOUBLE) < 80 THEN 'LMI tract' ELSE 'other tract' END AS tract,
       CASE replace(replace(r.debt_to_income_ratio, '%', ''), ' ', '')
            WHEN '<20' THEN '<30' WHEN '20-<30' THEN '<30' WHEN '30-<36' THEN '30-<36'
            WHEN '50-60' THEN '50-60' WHEN '>60' THEN '>60'
            ELSE CASE WHEN TRY_CAST(replace(r.debt_to_income_ratio, '%', '') AS INT) <= 42 THEN '36-42' ELSE '43-49' END
       END AS dti_band
FROM read_parquet('{OOF}') o JOIN read_parquet('{RAW}') r USING (row_id)""").df()
print(len(d), "hồ sơ")

# %%
rows = []
for dim in ["all", "loan_type", "purpose", "tract", "dti_band"]:
    groups = [("all", d)] if dim == "all" else list(d.groupby(dim))
    for g, x in groups:
        r = {"dimension": dim, "group": g, "n": len(x), "observed": x["y"].mean()}
        for m in ["lr", "gbt"]:
            p = x[f"p_{m}"]
            r[f"{m}_predicted"] = p.mean()
            r[f"{m}_citl"] = p.mean() - x["y"].mean()
            r[f"{m}_brier"] = float(np.mean((p - x["y"]) ** 2))
            r[f"{m}_auc"] = roc_auc_score(x["y"], p) if x["y"].nunique() == 2 else np.nan
        rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/pred_subgroups.csv", index=False)
pd.set_option("display.width", 220)
print(S.round(4).to_string(index=False))
