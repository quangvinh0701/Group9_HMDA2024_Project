"""
Tạo một file Parquet GIẢ LẬP có đúng cấu trúc hmda_2024_raw.parquet (99 cột string + row_id)
để chạy thử hai notebook trước khi chạy trên dữ liệu thật. Số liệu là ngẫu nhiên, không dùng để phân tích.

Usage: python gen_synth.py OUT_DIR [N_ROWS]
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = sys.argv[1] if len(sys.argv) > 1 else "synth"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
CHUNK = 100_000
rng = np.random.default_rng(7)
os.makedirs(OUT, exist_ok=True)

COLS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "columns.txt")).read().strip().split(", ")
assert len(COLS) == 99

STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
          "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()
FIPS = {s: f"{i + 1:02d}" for i, s in enumerate(STATES)}
FIPS.update({"PR": "72", "GU": "66", "VI": "78"})
ALLST = STATES + ["PR", "GU", "VI"]
w_st = rng.gamma(1.2, 1.0, len(STATES)); w_st = w_st / w_st.sum() * 0.975
P_ST = np.concatenate([w_st, [0.02, 0.0025, 0.0005]])
P_ST = P_ST / P_ST.sum()

# MSA: 1-3 mỗi bang, PR có 1 MSA, GU/VI không có MSA
msa_of = {}
code = 10000
for s in ALLST:
    k = 0 if s in ("GU", "VI") else (1 if s == "PR" else int(rng.integers(1, 4)))
    msa_of[s] = [str(code + 20 * j) for j in range(k)]
    code += 100
msa_eff = {m: rng.normal(0, 0.15) for ms in msa_of.values() for m in ms}
st_eff = {s: rng.normal(0, 0.1) for s in ALLST}

# Lender
NL = 140
LEIS = np.array([("5493" + "".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"), 16))) for _ in range(NL)])
w_l = rng.pareto(1.3, NL) + 0.2; w_l = w_l / w_l.sum()
l_eff = rng.normal(0, 0.7, NL)
zero_deny = set(rng.choice(NL, 3, replace=False))        # lender không báo hồ sơ từ chối
bad_usable = set(rng.choice(NL, 4, replace=False))       # hồ sơ từ chối hay thiếu DTI

DTI_CATS = np.array(["<20%", "20%-<30%", "30%-<36%"] + [str(v) for v in range(36, 50)] + ["50%-60%", ">60%"])
P_DTI = np.array([0.04, 0.13, 0.14] + [0.035] * 14 + [0.11, 0.04]); P_DTI /= P_DTI.sum()
DTI_EFF = np.array([0.4, 0.05, 0, *[0.0] * 14, 1.3, 3.8])


def pick(vals, p, n):
    return np.asarray(vals)[rng.choice(len(vals), n, p=np.asarray(p) / np.sum(p))]


def chunk(n, start):
    d = {}
    li = rng.choice(NL, n, p=w_l)
    st = pick(ALLST, P_ST, n)
    st_na = rng.random(n) < 0.002
    msa = np.array(["99999"] * n, dtype=object)
    cty = np.array(["NA"] * n, dtype=object)
    for i in range(n):
        ms = msa_of[st[i]]
        if ms and rng.random() > 0.15:
            msa[i] = ms[int(rng.integers(len(ms)))]
        cty[i] = FIPS[st[i]] + f"{int(rng.integers(1, 40)) * 2 + 1:03d}" if rng.random() > 0.01 else "NA"
    st = st.astype(object)
    st[st_na] = "NA"; msa[st_na] = "99999"; cty[st_na] = "NA"

    lien = pick(["1", "2"], [0.75, 0.25], n)
    openend = pick(["2", "1", "1111"], [0.92, 0.06, 0.02], n)
    bus = pick(["2", "1", "1111"], [0.93, 0.05, 0.02], n)
    rev = pick(["2", "1", "1111"], [0.995, 0.003, 0.002], n)
    dwell = pick(["Single Family (1-4 Units):Site-Built", "Single Family (1-4 Units):Manufactured",
                  "Multifamily:Site-Built"], [0.92, 0.05, 0.03], n)
    occ = pick(["1", "2", "3"], [0.9, 0.03, 0.07], n)
    ltype = pick(["1", "2", "3", "4"], [0.62, 0.23, 0.14, 0.01], n)
    purp = pick(["1", "31", "32", "2", "4", "5"], [0.6, 0.12, 0.18, 0.04, 0.0598, 0.0002], n)

    group = pick(["app", "6", "7", "8"], [0.885, 0.1, 0.005, 0.01], n)
    wd = rng.random(n)
    # DTI, CLTV, income, amount
    dti_i = rng.choice(len(DTI_CATS), n, p=P_DTI)
    dti = DTI_CATS[dti_i].astype(object)
    u = rng.random(n)
    dti[u < 0.05] = "NA"; dti[(u >= 0.05) & (u < 0.07)] = "Exempt"
    base_cltv = np.where(purp == "1", np.where(ltype == "2", 96.5, np.where(ltype == "3", 100.0, 85.0)), 70.0)
    base_cltv = np.where((purp == "32") & (ltype == "3"), 90.0, base_cltv)
    cltv = np.clip(base_cltv + rng.normal(0, 9, n), 5, 130)
    cltv = np.where((purp == "32") & np.isin(ltype, ["1", "2"]), np.minimum(cltv, 80.0), cltv)
    cltv_s = np.char.mod("%.3f", np.round(cltv, 3)).astype(object)
    u = rng.random(n)
    cltv_s[u < 0.03] = "NA"; cltv_s[(u >= 0.03) & (u < 0.05)] = "Exempt"; cltv_s[(u >= 0.05) & (u < 0.051)] = "300.0"
    inc = np.round(np.exp(rng.normal(4.6, 0.6, n))).astype(int)
    inc_s = inc.astype(str).astype(object)
    u = rng.random(n)
    inc_s[u < 0.04] = "NA"; inc_s[(u >= 0.04) & (u < 0.045)] = "0"; inc_s[(u >= 0.045) & (u < 0.046)] = "-5"
    inc_s[(u >= 0.046) & (u < 0.0462)] = "12000"; inc_s[(u >= 0.0462) & (u < 0.0464)] = "1111"
    amt = np.exp(rng.normal(12.5, 0.6, n))
    amt = (np.floor(amt / 10000) * 10000 + 5000).astype(np.int64)
    lti = amt / (1000 * np.maximum(inc, 1))

    # quyết định
    lmi_ratio = np.round(np.clip(rng.normal(110, 35, n), 20, 400), 2)
    lmi_ratio[rng.random(n) < 0.02] = 0
    lmi = (lmi_ratio > 0) & (lmi_ratio < 80)
    logit = (-2.3 + DTI_EFF[dti_i] + np.where(cltv > 100, 0.6, np.where(cltv <= 80, 0.3, 0))
             + np.where(lti > 3.4, 0.4, 0) + np.where(ltype == "2", 0.5, np.where(ltype == "3", 0.1, 0))
             + np.select([purp == "32", purp == "31", purp == "2", purp == "4", purp == "5"], [1.4, 0.8, 1.8, 1.8, 2.5], 0)
             + l_eff[li] + np.array([msa_eff.get(m, 0.0) for m in msa]) + np.array([st_eff.get(s, 1.5) for s in st])
             + np.where(lmi, 0.25, 0))
    p = 1 / (1 + np.exp(-logit))
    p[np.isin(li, list(zero_deny))] = 0
    action = np.where(wd < 0.17, "4", np.where(wd < 0.21, "5", "0")).astype(object)
    dec = action == "0"
    den = dec & (rng.random(n) < p)
    action[den] = "3"
    rest = dec & ~den
    action[rest] = np.where(rng.random(rest.sum()) < 0.05, "2", "1")
    action[group != "app"] = group[group != "app"]

    # trường bị NA theo action
    na_dc = np.isin(action, ["4", "5", "6", "7", "8"])
    dti[na_dc] = "NA"; cltv_s[na_dc] = "NA"
    bad = den & np.isin(li, list(bad_usable)) & (rng.random(n) < 0.8)
    dti[bad] = "NA"

    reasons = np.array([str(k) for k in range(1, 10)])
    r1 = np.where(action == "3", pick(reasons, [3, 0.3, 2.5, 1.7, 0.7, 0.7, 1.6, 0.02, 1.2], n), "10").astype(object)
    r2 = np.where((action == "3") & (rng.random(n) < 0.3), pick(reasons, [1] * 9, n), "").astype(object)
    blank = np.array([""] * n, dtype=object)

    orig = np.isin(action, ["1", "2"])
    rs = np.select([ltype == "1", ltype == "2", ltype == "3"], [0.2, 0.45, -0.4], 0.3) + np.where(purp == "32", 0.3, 0) \
        + rng.normal(0, 0.6, n)
    rs_s = np.char.mod("%.3f", np.round(rs, 3)).astype(object)
    u = rng.random(n)
    rs_s[u < 0.012] = "NA"; rs_s[(u >= 0.012) & (u < 0.0121)] = "Exempt"
    out_vals = np.array(["3960", "-385", "100", "72.5", "-20", "16.2", "-9.5", "14.9", "-7.9"], dtype=object)
    k = rng.random(n) < 0.0002
    rs_s[k] = out_vals[rng.integers(0, len(out_vals), k.sum())]
    rs_s[~orig] = "NA"
    ir_s = np.where(orig, np.char.mod("%.3f", np.round(6.5 + rng.normal(0, 0.6, n), 3)), "NA").astype(object)
    ir_s[orig & (rng.random(n) < 0.001)] = "0"
    term = pick(["360", "180", "240", "NA", "Exempt"], [0.85, 0.08, 0.05, 0.015, 0.005], n)
    tlc = np.where(action == "1", np.char.mod("%.2f", np.round(amt * 0.02 + rng.normal(0, 800, n), 2)), "NA").astype(object)
    pv = np.where(rng.random(n) < 0.05, "NA", (amt / np.clip(cltv, 20, 130) * 100 // 10000 * 10000 + 5000).astype(np.int64).astype(str)).astype(object)
    sub = pick(["1", "2", "3", "1111"], [0.8, 0.18, 0.01, 0.01], n)
    pay = pick(["1", "2", "3", "1111"], [0.9, 0.07, 0.02, 0.01], n)
    cll = pick(["C", "NC", "U", "NA"], [0.95, 0.04, 0.005, 0.005], n)
    tract = np.where(cty == "NA", "NA", np.char.add(cty.astype(str), "000100")).astype(object)

    d.update({
        "activity_year": np.full(n, "2024", dtype=object), "lei": LEIS[li].astype(object),
        "derived_msa_md": msa, "state_code": st, "county_code": cty, "census_tract": tract,
        "conforming_loan_limit": cll, "derived_dwelling_category": dwell, "action_taken": action,
        "loan_type": ltype, "loan_purpose": purp, "lien_status": lien, "reverse_mortgage": rev,
        "open_end_line_of_credit": openend, "business_or_commercial_purpose": bus,
        "loan_amount": amt.astype(str).astype(object), "combined_loan_to_value_ratio": cltv_s,
        "interest_rate": ir_s, "rate_spread": rs_s, "total_loan_costs": tlc, "loan_term": term,
        "property_value": pv, "occupancy_type": occ, "income": inc_s, "debt_to_income_ratio": dti,
        "submission_of_application": sub, "initially_payable_to_institution": pay,
        "denial_reason_1": r1, "denial_reason_2": r2, "denial_reason_3": blank, "denial_reason_4": blank,
        "tract_to_msa_income_percentage": np.char.mod("%.2f", lmi_ratio).astype(object),
        "applicant_age": pick(["25-34", "35-44", "45-54", "8888"], [0.3, 0.3, 0.3, 0.1], n),
        "co_applicant_age": pick(["35-44", "9999"], [0.4, 0.6], n),
        "derived_sex": pick(["Male", "Female", "Joint"], [0.4, 0.3, 0.3], n),
    })
    for c in COLS:
        if c not in d:
            d[c] = np.full(n, "NA", dtype=object)
    tab = {c: pa.array(d[c].tolist(), type=pa.string()) for c in COLS}
    tab["row_id"] = pa.array(np.arange(start, start + n, dtype=np.int64))
    return pa.table(tab)


path = os.path.join(OUT, "hmda_2024_raw.parquet")
writer = None
done = 0
last = None
while done < N:
    n = min(CHUNK, N - done)
    t = chunk(n, done)
    if done + n >= N:  # thêm 40 dòng trùng hoàn toàn (khác row_id) ở cuối file
        dup = t.slice(0, 40).to_pydict()
        dup["row_id"] = list(range(done + n, done + n + 40))
        t = pa.concat_tables([t, pa.table({c: pa.array(dup[c], type=t.schema.field(c).type) for c in t.column_names})])
    writer = writer or pq.ParquetWriter(path, t.schema, compression="snappy")
    writer.write_table(t)
    done += n
writer.close()

# Transmittal Sheet giả lập
with open(os.path.join(OUT, "2024_public_ts_one_year_csv.csv"), "w") as f:
    f.write("activity_year,calendar_quarter,lei,tax_id,agency_code,respondent_name,respondent_state,"
            "respondent_city,respondent_zip_code,lar_count\n")
    for i, lei in enumerate(LEIS):
        f.write(f"2024,4,{lei},00-0000000,{['1','2','3','5','7','9'][i % 6]},Lender {i:03d},CA,City,90000,0\n")
print("wrote", path, pq.ParquetFile(path).metadata.num_rows, "rows")
