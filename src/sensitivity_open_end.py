# %% [markdown]
# # HMDA 2024: độ nhạy khi đưa lại các hồ sơ open-end "Exempt"
#
# Ở bước lọc closed-end, mẫu chính chỉ giữ `open_end_line_of_credit = 2` và loại 216.601 hồ sơ có mã `1111`
# (lender được miễn báo cáo trường này). Proposal hứa kiểm tra xem kết quả có đổi không khi giữ lại nhóm này.
# Script này:
# 1. tạo một bản `rules.yaml` chỉ khác đúng một chỗ: bước closed_end nhận cả `2` và `1111`;
# 2. chạy lại toàn bộ pipeline Pandas với bản luật đó (trên Kaggle: các ô của `pandas_pipeline.py` được chép ngay sau ô này);
# 3. so các con số chính và xếp hạng $O/E$ của lender với lần chạy chính.
#
# Usage (ngoài Kaggle): `HMDA_INPUT=... python sensitivity_open_end.py [MAIN_PANDAS_RESULTS] [OUT_DIR]`

# %%
import glob
import json
import os
import subprocess
import sys

import yaml

ROOTS = [os.environ["HMDA_INPUT"]] if os.environ.get("HMDA_INPUT") else ["/kaggle/input"]


def find(pattern):
    for r in ROOTS:
        hits = sorted(glob.glob(os.path.join(r, "**", pattern), recursive=True))
        if hits:
            return hits[0]
    return None


ARGS = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]
MAIN_RES = ARGS[0] if len(ARGS) > 0 else os.path.dirname(find("results/pandas/key_numbers.json") or "")
_SENS = {"main_res": MAIN_RES, "out": ARGS[1] if len(ARGS) > 1 else None}   # giữ riêng: pipeline chạy sau có thể ghi đè tên biến
SENS_IN = "/tmp/hmda_sens_input"
os.makedirs(SENS_IN, exist_ok=True)
for pat in ["hmda_2024_raw.parquet", "expected.yaml", "*public_ts*.csv", "*public_ts*.zip"]:
    src = find(pat)
    if src and not os.path.exists(os.path.join(SENS_IN, os.path.basename(src))):
        os.symlink(src, os.path.join(SENS_IN, os.path.basename(src)))
R = yaml.safe_load(open(find("rules.yaml"), encoding="utf-8"))
step = next(f for f in R["samples"]["core_filters"] if f["step"] == "closed_end")
assert step == {"step": "closed_end", "column": "open_end_line_of_credit", "equals": "2"}, step
step.pop("equals"); step["in"] = ["2", "1111"]
yaml.safe_dump(R, open(os.path.join(SENS_IN, "rules.yaml"), "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
os.environ["HMDA_INPUT"] = SENS_IN            # pipeline Pandas sẽ đọc bản luật này
IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    os.environ.setdefault("HMDA_OUT", "output_sens_open_end")
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(here, "pandas_pipeline.py")], check=True)
print("luật độ nhạy:", step, "| kết quả chính:", MAIN_RES)

# %% [markdown]
# ## So với lần chạy chính
# (Trên Kaggle, ô này nằm sau toàn bộ các ô của `pandas_pipeline.py`.)

# %%
import pandas as pd
from scipy.stats import spearmanr

OUTROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("HMDA_OUT", "output_sens_open_end")
SENS_RES = f"{OUTROOT}/results/pandas"
OUT = _SENS["out"] or f"{OUTROOT}/results/sensitivity"
os.makedirs(OUT, exist_ok=True)
MAIN_RES = _SENS["main_res"]
km = json.load(open(os.path.join(MAIN_RES, "key_numbers.json")))
ks = json.load(open(os.path.join(SENS_RES, "key_numbers.json")))
KEYS = ["core", "decisioned", "usable", "denial_rate_conventional", "denial_rate_FHA",
        "kitagawa_loan_type_2v1_gap", "kitagawa_loan_type_2v1_composition", "kitagawa_loan_type_2v1_rate_gap_at_a_mix",
        "kitagawa_loan_type_2v1_rate_gap_at_b_mix", "kitagawa_loan_purpose_32v1_gap", "kitagawa_loan_purpose_32v1_rate",
        "jump_1_49_to_5060_excess", "jump_1_42_to_43_excess", "bunching_49_1", "pricing_conventional_32v1_gap",
        "tau_state", "tau_msa", "tau_lender", "lender_median_oe", "twoway_lender_signal_sd", "twoway_msa_signal_sd",
        "pred_p_lr_roc_auc"]
rows = [{"key": k, "main": km[k], "sensitivity": ks[k]} for k in KEYS if k in km and k in ks]
rows.append({"key": "twoway_signal_sd_ratio", "main": km["twoway_lender_signal_sd"] / km["twoway_msa_signal_sd"],
             "sensitivity": ks["twoway_lender_signal_sd"] / ks["twoway_msa_signal_sd"]})
lm = pd.read_csv(os.path.join(MAIN_RES, "sq4_lender_oe.csv"))[["lei", "oe"]]
ls_ = pd.read_csv(os.path.join(SENS_RES, "sq4_lender_oe.csv"))[["lei", "oe"]]
both = lm.merge(ls_, on="lei", suffixes=("_main", "_sens"))
rows.append({"key": "lenders_in_both", "main": len(lm), "sensitivity": len(ls_), "note": len(both)})
rows.append({"key": "spearman_lender_oe_main_vs_sens", "main": 1.0, "sensitivity": spearmanr(both["oe_main"], both["oe_sens"])[0]})
T = pd.DataFrame(rows)
T["change"] = T["sensitivity"] - T["main"]
T.to_csv(f"{OUT}/open_end_exempt.csv", index=False)
pd.set_option("display.width", 200)
print(T.to_string(index=False))
