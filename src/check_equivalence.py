"""
Kiểm tra tương đương Pandas và PySpark (rules.yaml mục 19).

Đọc các file eq_*.csv và key_numbers.json do hai notebook ghi ra, so từng giá trị, xuất:
  equivalence_summary.csv / .md : bảng Quantity | Pandas | PySpark | Agree | Explanation (đưa vào báo cáo)
  equivalence_details.csv       : từng giá trị đã so, kèm sai lệch tương đối

Usage:
  python check_equivalence.py [PANDAS_DIR] [SPARK_DIR] [OUT_DIR]
Mặc định tìm results/pandas và results/spark trong /kaggle/working, /kaggle/input hoặc thư mục hiện tại.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

TOL = 1e-9          # sai số tương đối cho trung bình, tỷ lệ, tổng
TOL_ITER = 1e-6     # mô hình hai chiều (vòng lặp)


def locate(engine):
    pats = [f"/kaggle/working/results/{engine}", f"/kaggle/input/**/results/{engine}", f"results/{engine}",
            f"**/results/{engine}"]
    for p in pats:
        hits = [h for h in glob.glob(p, recursive=True) if os.path.exists(os.path.join(h, "eq_1_filter_counts.csv"))]
        if hits:
            return hits[0]
    raise SystemExit(f"Không tìm thấy results/{engine}/eq_1_filter_counts.csv")


ARGS = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]   # bỏ tham số của Jupyter
PD_DIR = ARGS[0] if len(ARGS) > 0 else locate("pandas")
SP_DIR = ARGS[1] if len(ARGS) > 1 else locate("spark")
OUT = ARGS[2] if len(ARGS) > 2 else ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
os.makedirs(OUT, exist_ok=True)
print("pandas:", PD_DIR, "\nspark :", SP_DIR)


def read(d, name):
    return pd.read_csv(os.path.join(d, f"{name}.csv"), dtype=str, keep_default_na=False)


def num(s):
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def rel(a, b):
    if pd.isna(a) and pd.isna(b):
        return 0.0
    if pd.isna(a) or pd.isna(b):
        return np.inf
    return abs(a - b) / max(1.0, abs(a), abs(b)) if max(abs(a), abs(b)) >= 1 else abs(a - b)


details, summary = [], []


def compare(name, quantity, keys, exact=(), approx=(), report_only=(), tol=TOL):
    a, b = read(PD_DIR, name), read(SP_DIR, name)
    m = a.merge(b, on=keys, how="outer", suffixes=("_pd", "_sp"), indicator=True)
    only = m[m["_merge"] != "both"]
    n_vals, worst, bad, notes = 0, 0.0, 0, []
    if len(only):
        bad += len(only)
        notes.append(f"{len(only)} nhóm chỉ có ở một bên: {only[keys].head(3).to_dict('records')}")
    both = m[m["_merge"] == "both"]
    for col in list(exact) + list(approx):
        x, y = num(both[f"{col}_pd"]), num(both[f"{col}_sp"])
        for k_, xv, yv in zip(both[keys].astype(str).agg("|".join, axis=1), x, y):
            d = 0.0 if (col in exact and xv == yv) else rel(xv, yv)
            ok = (xv == yv) if col in exact else d <= tol
            details.append({"output": name, "key": k_, "column": col, "pandas": xv, "spark": yv, "rel_diff": d, "agree": ok})
            n_vals += 1; worst = max(worst, d if np.isfinite(d) else 1e9); bad += (not ok)
    for col in report_only:
        x, y = num(both[f"{col}_pd"]), num(both[f"{col}_sp"])
        diffs = [rel(xv, yv) for xv, yv in zip(x, y) if not (pd.isna(xv) and pd.isna(yv))]
        if diffs:
            notes.append(f"{col}: sai lệch tương đối lớn nhất {max(diffs):.2e} (chỉ báo cáo)")
    if bad == 0:
        expl = "khớp tuyệt đối" if worst == 0 else f"khớp trong dung sai, sai lệch tương đối lớn nhất {worst:.1e} (thứ tự cộng số thực)"
    else:
        expl = f"LỆCH {bad} giá trị"
    summary.append({"Quantity": quantity, "Values compared": n_vals, "Agree": bad == 0,
                    "Max relative difference": worst, "Explanation": "; ".join([expl] + notes)})


compare("eq_1_filter_counts", "(1) Row counts after each filter", ["step"], exact=["rows"])
compare("eq_2_special_codes", "(2) Special-code counts, 10 fields", ["field", "code"], exact=["count"])
compare("eq_2b_field_profile", "(2b) Field profile, all 99 fields (codes and numeric values)", ["field"],
        exact=["NA", "Exempt", "1111", "8888", "9999", "(empty)", "numeric"])
compare("eq_3_action_taken", "(3) action_taken counts and shares", ["action_taken"], exact=["n"], approx=["share"])
compare("eq_4_amounts_by_loan_type", "(4) Mean and median of income, loan amount, rate spread by loan type",
        ["sample", "variable", "loan_type"], exact=["n"], approx=["mean", "median_exact"])
compare("eq_4b_rate_spread_counts", "(4b) Rate spread: numeric, NA, Exempt, out of range", ["loan_type"],
        exact=["numeric", "na", "exempt", "out_of_range", "valid"])
compare("eq_5_denial_by_state", "(5) Denial rate by state (all codes, including NA)", ["state_code"],
        exact=["n_decisioned", "n_denied"], approx=["denial_rate"])
compare("eq_6_top50_lenders", "(6) Aggregates of the 50 largest lenders", ["rank", "lei"],
        exact=["n_decisioned", "n_denied", "n_usable"], approx=["mean_income", "mean_loan_amount", "denial_rate", "O", "E", "oe"])

# sai số của percentile_approx so với trung vị chính xác (chỉ bên Spark có)
sp4 = read(SP_DIR, "eq_4_amounts_by_loan_type")
ap = [rel(x, y) for x, y in zip(num(sp4["median_exact"]), num(sp4["median_approx"]))]
summary.append({"Quantity": "(4) percentile_approx vs exact median (Spark)", "Values compared": len(ap), "Agree": True,
                "Max relative difference": max(ap) if ap else np.nan,
                "Explanation": f"chỉ báo cáo: percentile_approx lệch tối đa {max(ap):.2e} so với percentile chính xác" if ap else ""})

# key_numbers.json
kp = json.load(open(os.path.join(PD_DIR, "key_numbers.json"), encoding="utf-8"))
ks = json.load(open(os.path.join(SP_DIR, "key_numbers.json"), encoding="utf-8"))
SKIP = {"engine", "runtime_s", "spark_init_s", "rules_version"}
shared = sorted((set(kp) & set(ks)) - SKIP)
n_bad, worst = 0, 0.0
for k in shared:
    a, b = kp[k], ks[k]
    if a is None or b is None or isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        ok = a == b; d = 0.0 if ok else np.inf
    else:
        d = rel(float(a), float(b)); ok = d <= (TOL_ITER if k.startswith("twoway_") else TOL)
    details.append({"output": "key_numbers", "key": k, "column": "value", "pandas": a, "spark": b, "rel_diff": d, "agree": ok})
    n_bad += (not ok); worst = max(worst, d if np.isfinite(d) else 1e9)
summary.append({"Quantity": "Key analysis numbers (SQ1-SQ4, two-way, HHI)", "Values compared": len(shared), "Agree": n_bad == 0,
                "Max relative difference": worst,
                "Explanation": ("khớp trong dung sai" if n_bad == 0 else f"LỆCH {n_bad} số") +
                               f"; khóa chỉ có một bên: {sorted((set(kp) ^ set(ks)) - SKIP)[:8]}"})
if kp.get("rules_version") != ks.get("rules_version"):
    summary.append({"Quantity": "rules.yaml version", "Values compared": 1, "Agree": False, "Max relative difference": np.nan,
                    "Explanation": f"Pandas {kp.get('rules_version')} khác Spark {ks.get('rules_version')}"})

S = pd.DataFrame(summary)
Dt = pd.DataFrame(details)
S.to_csv(os.path.join(OUT, "equivalence_summary.csv"), index=False)
Dt.to_csv(os.path.join(OUT, "equivalence_details.csv"), index=False)
with open(os.path.join(OUT, "equivalence_summary.md"), "w", encoding="utf-8") as f:
    f.write("| Quantity | Values compared | Agree | Max relative difference | Explanation |\n|---|---|---|---|---|\n")
    for _, r in S.iterrows():
        f.write(f"| {r['Quantity']} | {r['Values compared']} | {'yes' if r['Agree'] else 'NO'} | {r['Max relative difference']:.1e} | {r['Explanation']} |\n")
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
print(S.to_string(index=False))
bad = Dt[~Dt["agree"].astype(bool)]
if len(bad):
    print("\nCác giá trị lệch (tối đa 20):\n", bad.head(20).to_string(index=False))
print("\nTất cả khớp" if S["Agree"].all() else "\nCÓ LỆCH: xem equivalence_details.csv")
