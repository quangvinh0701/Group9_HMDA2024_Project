"""
So số đếm của chính nhóm với HMDA Data Browser (FFIEC), ngoài cột action_taken đã so trong pipeline.

Đầu vào:
  results/independent/db_slices.csv          số hồ sơ và tổng loan_amount theo action_taken x một chiều,
                                             do independent_check.py đếm trên toàn file thô
  results/external/data_browser_slices_2024.csv  cùng các lát cắt lấy từ Data Browser API (xem fetch() bên dưới)
Đầu ra:
  results/external/data_browser_comparison.csv   từng lát cắt: count và sum của hai bên, khớp hay không
  in ra bảng tóm tắt theo chiều.

Usage:
  python compare_data_browser.py [--fetch]     # --fetch: tải lại dữ liệu Data Browser (cần Internet)
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "results/external/data_browser_slices_2024.csv")
OURS = os.path.join(ROOT, "results/independent/db_slices.csv")
API = "https://ffiec.cfpb.gov/v2/data-browser-api/view/"
ACTIONS = "actions_taken=1,2,3,4,5,6,7,8&years=2024"
DIMS = {"loan_types": "1,2,3,4", "loan_purposes": "1,2,31,32,4,5", "lien_statuses": "1,2", "construction_methods": "1,2"}
STATES = ("AL,AK,AZ,AR,CA,CO,CT,DE,DC,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,"
          "OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY,PR,GU,VI").split(",")


def fetch():
    import json
    import urllib.request
    rows = []
    get = lambda q: json.load(urllib.request.urlopen(API + q, timeout=60))["aggregations"]
    for dim, vals in DIMS.items():
        for a in get(f"nationwide/aggregations?{ACTIONS}&{dim}={vals}"):
            rows.append([dim, a[dim], a["actions_taken"], a["count"], int(a["sum"])])
    for s in STATES:
        for a in get(f"aggregations?states={s}&{ACTIONS}"):
            rows.append(["states", s, a["actions_taken"], a["count"], int(a["sum"])])
    lines = sorted(",".join(map(str, r)) for r in rows)
    open(DB, "w").write("dimension,value,action_taken,count,sum\n" + "\n".join(lines) + "\n")


if "--fetch" in sys.argv:
    fetch()
db = pd.read_csv(DB, dtype={"value": str, "action_taken": str}, keep_default_na=False)
ours = pd.read_csv(OURS, dtype={"value": str, "action_taken": str}, keep_default_na=False)
ours = ours[ours["value"] != "NA"]        # Data Browser không có nhóm "không có bang"
m = db.merge(ours, on=["dimension", "value", "action_taken"], how="outer", suffixes=("_db", "_ours"))
m["count_ours"] = m["count_ours"].fillna(0).astype("int64")
m["sum_loan_amount"] = m["sum_loan_amount"].fillna(0).round().astype("int64")
m["count_db"] = m["count_db"].fillna(0).astype("int64")
m["sum"] = m["sum"].fillna(0).astype("int64")
m["agree"] = (m["count_db"] == m["count_ours"]) & (m["sum"] == m["sum_loan_amount"])
m = m.rename(columns={"sum": "sum_db", "sum_loan_amount": "sum_ours"})
m.to_csv(os.path.join(ROOT, "results/external/data_browser_comparison.csv"), index=False)
S = m.assign(nonzero=m["count_db"] > 0).groupby("dimension").agg(
    slices=("agree", "size"), nonzero_slices=("nonzero", "sum"), agree=("agree", "all"),
    records=("count_db", "sum")).reset_index()
print(S.to_string(index=False))
print(f"\n{int(m['agree'].sum())}/{len(m)} lát cắt khớp cả số hồ sơ lẫn tổng loan_amount")
sys.exit(0 if m["agree"].all() else 1)
