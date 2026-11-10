"""
Build figures, tables and number macros for the HMDA 2024 report.

Every number printed in the report comes from the Kaggle run outputs:
  results/pandas       (HMDA v1 pandas, Version 1)
  results/spark        (HMDA v1 pyspark, Version 1)
  results/perf         (HMDA v1 performance, Version 1)
  results/equivalence  (HMDA v1 check equivalence, Version 1)
  results/external     (FFIEC Data Browser counts, verification only)

Usage: python build_report_assets.py RESULTS_DIR OUT_DIR
(RESULTS_DIR is either the raw Kaggle download layout shown above or the repository's results/ folder;
set HMDA_STYLE=acm for figures in the ACM template font.)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

if len(sys.argv) < 3:
    sys.exit("Usage: python build_report_assets.py RESULTS_DIR OUT_DIR")
RES, OUT = sys.argv[1], sys.argv[2]
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

P = os.path.join(RES, "pandas", "results", "pandas")
S = os.path.join(RES, "spark", "results", "spark")
PF = os.path.join(RES, "perf", "results", "perf")
EQ = os.path.join(RES, "equivalence")
# repository layout: results/pandas, results/spark, results/perf directly under RESULTS_DIR
if not os.path.isdir(P):
    P, S, PF = os.path.join(RES, "pandas"), os.path.join(RES, "spark"), os.path.join(RES, "perf")

# ---------------------------------------------------------------- fonts / style
for f in ["lmroman10-regular.otf", "lmroman10-bold.otf", "lmroman10-italic.otf"]:
    p = os.path.join("/usr/share/texmf/fonts/opentype/public/lm", f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
# keep only the 10 pt design size of Latin Modern Roman (other optical sizes look wrong at figure scale)
font_manager.fontManager.ttflist = [fe for fe in font_manager.fontManager.ttflist
                                    if not (fe.name == "Latin Modern Roman" and "lmroman10-" not in fe.fname)]
# HMDA_STYLE=acm: figures in Linux Libertine to match the ACM (acmart) template text font
STYLE = os.environ.get("HMDA_STYLE", "lm")
FONT = "Latin Modern Roman"
if STYLE == "acm":
    for f in ["LinLibertine_R.otf", "LinLibertine_RB.otf", "LinLibertine_RI.otf"]:
        p = os.path.join("/usr/share/fonts/opentype/linux-libertine", f)
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
    FONT = "Linux Libertine O"
if FONT not in {fe.name for fe in font_manager.fontManager.ttflist}:   # font not installed on this machine
    print(f"font {FONT!r} not found, using DejaVu Serif"); FONT = "DejaVu Serif"
plt.rcParams.update({
    "font.family": FONT, "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "mathtext.fontset": "cm",
})
if STYLE == "acm":
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
                         "mathtext.bf": FONT + ":bold"})
C_CONV, C_FHA, C_VA, C_ALL = "#1f4e79", "#c55a11", "#548235", "#404040"
C_PD, C_SP = "#c55a11", "#1f4e79"


def rd(path, **kw):
    return pd.read_csv(path, **kw)


kp = json.load(open(os.path.join(P, "key_numbers.json")))
ks = json.load(open(os.path.join(S, "key_numbers.json")))

# ---------------------------------------------------------------- macros
MAC = {}


def fmt_int(x):
    return f"{int(round(x)):,}"


def fmt(x, d=1):
    return f"{x:,.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pp(x, d=1):          # rate difference in percentage points, signed
    v = 100 * x
    s = f"{v:+.{d}f}"
    return s.replace("-", "$-$") if v < 0 else s


def sg(x, d=2):            # signed plain number with a proper minus sign
    t = f"{x:.{d}f}"
    return t.replace("-", "$-$") if x < 0 else t


def mac(name, value):
    assert name.isalpha(), name
    MAC[name] = value


# data size and samples
eq1 = rd(os.path.join(P, "eq_1_filter_counts.csv")).set_index("step")["rows"]
for k, n in [("all_records", "nAll"), ("applications", "nApps"), ("first_lien", "nFirstLien"),
             ("closed_end", "nClosedEnd"), ("non_business", "nNonBusiness"), ("non_reverse", "nNonReverse"),
             ("site_built_1_4_unit", "nSiteBuilt"), ("principal_residence", "nCore"),
             ("open_end_exempt_removed", "nOpenEndExempt"), ("decisioned", "nDecisioned"),
             ("usable", "nUsable"), ("pricing", "nPricing"), ("pricing_usable", "nPricingUsable")]:
    mac(n, fmt_int(eq1[k]))
mac("shareUsable", pct(eq1["usable"] / eq1["decisioned"]))
mac("nDup", fmt_int(kp["duplicates"]))

eq3 = rd(os.path.join(P, "eq_3_action_taken.csv")).set_index("action_taken")
mac("nOriginated", fmt_int(eq3.loc[1, "n"]))
mac("nPurchased", fmt_int(eq3.loc[6, "n"]))
mac("nPreapproval", fmt_int(eq3.loc[7, "n"] + eq3.loc[8, "n"]))
mac("rawDenialAll", pct(eq3.loc[3, "n"] / (eq3.loc[1, "n"] + eq3.loc[2, "n"] + eq3.loc[3, "n"])))
db = json.load(open(os.path.join(RES, "external", "data_browser_action_taken_2024.json")))
dbc = {int(a["actions_taken"]): a["count"] for a in db["aggregations"]}
assert all(dbc[i] == eq3.loc[i, "n"] for i in range(1, 9)), "Data Browser mismatch"
mac("dbMatch", "all eight")

# denial rates
for t, n in [("core", "drCore"), ("conventional", "drConv"), ("FHA", "drFHA"), ("VA", "drVA"), ("USDA", "drUSDA")]:
    mac(n, pct(kp[f"denial_rate_{t}"]))
pur = rd(os.path.join(P, "sq1_denial_by_purpose.csv")).set_index("purpose")
for t, n in [("purchase", "drPurchase"), ("refinance", "drRefi"), ("cash_out_refinance", "drCashout"),
             ("home_improvement", "drHomeImp"), ("other", "drOtherPurpose")]:
    mac(n, pct(pur.loc[t, "denial_rate"]))
txp = rd(os.path.join(P, "sq1_denial_type_x_purpose.csv")).set_index(["type", "purpose"])
mac("drFHACashout", pct(txp.loc[("FHA", "cash_out_refinance"), "denial_rate"]))
mac("drConvCashout", pct(txp.loc[("conventional", "cash_out_refinance"), "denial_rate"]))
mac("drFHAPurchase", pct(txp.loc[("FHA", "purchase"), "denial_rate"]))
mac("drConvPurchase", pct(txp.loc[("conventional", "purchase"), "denial_rate"]))

out_t = rd(os.path.join(P, "sq1_outcomes_by_type.csv")).set_index("type")
amt = rd(os.path.join(P, "sq1_loan_amount_originated_by_type.csv")).set_index("type")
rs_t = rd(os.path.join(P, "sq2_rate_spread_by_type.csv")).set_index("type")
dt = rd(os.path.join(P, "sq1_denial_by_type.csv")).set_index("type")

# DTI
d19 = rd(os.path.join(P, "sq2_denial_by_dti_19.csv"))
band = rd(os.path.join(P, "sq2_denial_by_dti_band.csv")).set_index("dti_band")
mac("drDTIlowTwenty", pct(d19.loc[d19.dti == "<20", "denial_rate"].iloc[0]))
mac("drDTItwentyThirty", pct(d19.loc[d19.dti == "20-<30", "denial_rate"].iloc[0]))
mac("drDTIthirtyThirtysix", pct(d19.loc[d19.dti == "30-<36", "denial_rate"].iloc[0]))
mac("drDTIfiftySixty", pct(band.loc["50-60", "denial_rate"]))
mac("drDTIoverSixty", pct(band.loc[">60", "denial_rate"]))
mid = d19[(d19.dti_ordinal >= 3) & (d19.dti_ordinal <= 17)]
mac("drDTImidMin", pct(mid.denial_rate.min()))
mac("drDTImidMax", pct(mid.denial_rate.max()))
jt = rd(os.path.join(P, "sq2_dti_jump_test.csv"))
jt["scope"] = jt["scope"].astype(str)


def jget(scope, jump, col):
    return jt.loc[(jt.scope == scope) & (jt.jump == jump), col].iloc[0]


mac("jumpConvMain", pp(jget("1", "49 -> 50-60", "excess")))
mac("jumpConvMainLo", pct(jget("1", "49 -> 50-60", "ci_low")))
mac("jumpConvMainHi", pct(jget("1", "49 -> 50-60", "ci_high")))
mac("drConvFortynine", pct(jget("1", "49 -> 50-60", "rate_from")))
mac("drConvFiftySixty", pct(jget("1", "49 -> 50-60", "rate_to")))
mac("jumpConvFortythree", pp(jget("1", "42 -> 43", "excess"), 2))
mac("jumpFHAMain", pp(jget("2", "49 -> 50-60", "excess")))
mac("jumpVAMain", pp(jget("3", "49 -> 50-60", "excess")))
mac("jumpAllMain", pp(jget("all", "49 -> 50-60", "excess")))
mac("drFHAFiftySixty", pct(jget("2", "49 -> 50-60", "rate_to")))
bun = rd(os.path.join(P, "sq2_dti_bunching.csv"))
bun["scope"] = bun["scope"].astype(str)
bset = bun.set_index("scope")["bunching_ratio"]
mac("bunchConv", fmt(bset["1"], 2))
mac("bunchFHA", fmt(bset["2"], 2))
mac("bunchVA", fmt(bset["3"], 2))
lowr = rd(os.path.join(P, "sq2_low_dti_reasons.csv")).set_index("dti")
lowp = rd(os.path.join(P, "sq2_low_dti_profile.csv")).set_index("dti")
mac("lowDTIcredit", pct(lowr.loc["<20", "credit_history"], 0))
mac("midDTIcredit", pct(lowr.loc["20-<30", "credit_history"], 0))
mac("lowDTIincome", fmt_int(lowp.loc["<20", "median_income"]))
mac("midDTIincome", fmt_int(lowp.loc["20-<30", "median_income"]))
mac("lowDTIcltv", fmt_int(lowp.loc["<20", "median_cltv"]))
mac("lowDTInonPurchase", pct(1 - lowp.loc["<20", "share_purchase"], 0))
mac("midDTInonPurchase", pct(1 - lowp.loc["20-<30", "share_purchase"], 0))
cl = rd(os.path.join(P, "sq2_denial_by_cltv_band.csv")).set_index("cltv_band")
mac("drCLTVlow", pct(cl.loc["<=80", "denial_rate"]))
mac("drCLTVmid", pct(cl.loc["90-95", "denial_rate"]))
mac("drCLTVhigh", pct(cl.loc[">100", "denial_rate"]))
gt80 = rd(os.path.join(P, "sq2_cltv_gt80_by_type_purpose.csv")).set_index(["type", "purpose"])
mac("fhaCashoutGtEighty", pct(gt80.loc[("FHA", "cash_out_refinance"), "share_cltv_gt_80"]))
mac("convCashoutLeEighty", pct(1 - gt80.loc[("conventional", "cash_out_refinance"), "share_cltv_gt_80"], 0))
reas = rd(os.path.join(P, "sq2_reasons_by_purpose.csv")).set_index("purpose")
mac("creditCashout", pct(reas.loc["cash_out_refinance", "credit_history"]))
mac("creditPurchase", pct(reas.loc["purchase", "credit_history"]))
mac("dtiReasonPurchase", pct(reas.loc["purchase", "debt_to_income"]))

# Kitagawa
kd = rd(os.path.join(P, "sq2_kitagawa_denial.csv")).set_index("label")
f = kd.loc["FHA vs conventional"]
mac("kgFHAgap", pp(f.gap)); mac("kgFHAcomp", pp(f.composition)); mac("kgFHArate", pp(f.rate))
mac("kgFHAatFHA", pp(f.rate_gap_at_a_mix)); mac("kgFHAatConv", pp(f.rate_gap_at_b_mix))
mac("kgFHAmin", pp(f.within_cell_gap_min, 0)); mac("kgFHAmax", pp(f.within_cell_gap_max, 0))
mac("kgFHAhigher", fmt_int(f.cells_a_higher)); mac("kgFHAlower", fmt_int(f.cells_a_lower))
mac("kgFHAcompShare", pct(f.composition / f.gap, 0))
mac("minFHAcell", fmt_int(f.min_cell_n_a))
c = kd.loc["cash-out vs purchase"]
mac("kgCOgap", pp(c.gap)); mac("kgCOcomp", pp(c.composition)); mac("kgCOrate", pp(c.rate))
mac("kgCOhigher", fmt_int(c.cells_a_higher))
r = kd.loc["refinance vs purchase"]
mac("kgRFgap", pp(r.gap)); mac("kgRFcomp", pp(r.composition)); mac("kgRFrate", pp(r.rate))
lk = rd(os.path.join(P, "sq3_lmi_kitagawa.csv"))
la = lk[lk.scope == "all"].iloc[0]
mac("kgLMIgap", pp(la.gap)); mac("kgLMIcomp", pp(la.composition)); mac("kgLMIrate", pp(la.rate))
mac("kgLMIhigher", fmt_int(la.cells_a_higher))
lt_ = lk[lk.scope.isin(["conventional", "FHA", "VA"])]
mac("lmiTypeMin", fmt(100 * lt_.rate.min(), 1)); mac("lmiTypeMax", fmt(100 * lt_.rate.max(), 1))
ks_rs = rd(os.path.join(P, "sq2_kitagawa_rate_spread.csv"))
cv = ks_rs[(ks_rs.loan_type == "conventional") & (ks_rs.a == 32)].iloc[0]
mac("psCOconvGap", fmt(cv.gap, 2)); mac("psCOconvComp", fmt(cv.composition, 2)); mac("psCOconvRate", fmt(cv.rate, 2))
mac("psCOconvCov", pct(cv.coverage_b, 0))
vv = ks_rs[(ks_rs.loan_type == "VA") & (ks_rs.a == 32)].iloc[0]
mac("psCOvaGap", sg(vv.gap)); mac("psCOvaComp", sg(vv.composition)); mac("psCOvaRate", sg(vv.rate))
fh = ks_rs[(ks_rs.loan_type == "FHA") & (ks_rs.a == 32)].iloc[0]
mac("psCOfhaCov", pct(fh.coverage_b, 0))
cut = rd(os.path.join(P, "appendix_rate_spread_cut_sensitivity.csv"))
c5 = cut[(cut.cut == "|rs| <= 5") & (cut.loan_type == "conventional")].iloc[0]
mac("psCOconvGapFive", fmt(c5.gap, 2))
mac("rsOut", fmt_int(kp["rs_out_of_range"]))

# geography
mac("nStates", fmt_int(kp["states_universe"]))
mac("tauState", fmt(kp["tau_state"], 3)); mac("sdStateRaw", pct(kp["sd_state_raw_rate"], 2))
mac("sdStateStd", pct(kp["sd_state_std_rate"], 2))
mac("nMSA", fmt_int(kp["msa_common"])); mac("tauMSA", fmt(kp["tau_msa"], 2))
mac("msaOutside", pct(kp["msa_share_outside_95"], 0))
mac("oeLMI", fmt(kp["oe_lmi"], 3)); mac("oeNonLMI", fmt(kp["oe_non_lmi"], 3))
lmi = rd(os.path.join(P, "sq3_lmi_oe.csv")).set_index("lmi_group")
mac("drLMI", pct(lmi.loc["LMI", "raw_rate"])); mac("drNonLMI", pct(lmi.loc["other", "raw_rate"]))
lsz = rd(os.path.join(P, "sq3_lmi_loan_size.csv")).set_index("lmi_group")
mac("loanLMI", fmt_int(lsz.loc["LMI", "median_loan_originated"])); mac("loanNonLMI", fmt_int(lsz.loc["other", "median_loan_originated"]))
sep = rd(os.path.join(P, "sq3_state_separate_lines.csv"), keep_default_na=False).set_index("state_code")
mac("nNoState", fmt_int(sep.loc["NA", "n"])); mac("drNoState", pct(sep.loc["NA", "raw_rate"]))
mac("oeNoState", fmt(sep.loc["NA", "oe"], 2)); mac("oePR", fmt(sep.loc["PR", "oe"], 2))
mlb = rd(os.path.join(P, "dq_missing_location_by_tract.csv"))
mac("noStateNoTract", fmt_int(mlb[(mlb.group == "missing_state") & (mlb.tract_missing)]["n"].iloc[0]))
mpm = rd(os.path.join(P, "dq_missing_location_purpose_mix.csv")).set_index("group")
mac("noStatePurchase", pct(mpm.loc["missing_state", "purchase"], 0))
st = rd(os.path.join(P, "sq3_state_oe.csv")).sort_values("oe")
mac("stateLowName", st.iloc[0].state_code); mac("stateLowOE", fmt(st.iloc[0].oe, 2))
mac("stateHighName", st.iloc[-1].state_code); mac("stateHighOE", fmt(st.iloc[-1].oe, 2))
mac("fallbackShare", pct(kp["fallback_share"], 2)); mac("nExtCells", fmt_int(kp["n_ext_cells"]))
mac("nationalOE", fmt(kp["national_oe"], 4))

# lenders
mac("nLenderEightTwelve", fmt_int(kp["lenders_812"])); mac("lenderShare", pct(kp["lenders_812_share"]))
mac("nLender", fmt_int(kp["lenders_767"]))
mac("tauLender", fmt(kp["tau_lender"], 2)); mac("tauLenderW", fmt(kp["tau_lender_weighted"], 2))
mac("lenderOutside", pct(kp["lender_share_outside_95"], 0)); mac("lenderOutsideHigh", pct(kp["lender_share_outside_998"], 0))
mac("lenderMedianOE", fmt(kp["lender_median_oe"], 2))
mac("tauLenderConv", fmt(kp["tau_lender_within_conventional"], 2)); mac("tauLenderFHA", fmt(kp["tau_lender_within_FHA"], 2))
mac("rhoConv", fmt(kp["spearman_lender_oe_main_vs_within_conventional"], 3))
mac("rhoFHA", fmt(kp["spearman_lender_oe_main_vs_within_FHA"], 3))
mac("rhoDef", fmt(kp["denial_def_spearman"], 3)); mac("quintChange", pct(kp["denial_def_share_changing_quintile"], 0))
mac("rhoDTIfive", fmt(kp["dti5_spearman_lender_oe"], 4)); mac("rhoFallback", fmt(kp["fallback_robust_spearman"], 4))
mac("nZeroDen", fmt_int(kp["lenders_zero_denials"]))
mac("spreadTauLender", fmt(kp["spread_tau_lender"], 2)); mac("spreadTauMSA", fmt(kp["spread_tau_msa"], 2))
mac("nSpreadLender", fmt_int(kp["spread_groups_lender"]))
mac("twLenders", fmt_int(kp["twoway_lenders"])); mac("twMSA", fmt_int(kp["twoway_msas"]))
mac("twRecords", fmt_int(kp["twoway_records"]))
mac("twLenderSD", pct(kp["twoway_lender_signal_sd"], 1)); mac("twMSASD", pct(kp["twoway_msa_signal_sd"], 1))
mac("twLenderFullSD", pct(kp["twoway_lender_full_sd"], 2)); mac("twMSAFullSD", pct(kp["twoway_msa_full_sd"], 2))
mac("twRatio", fmt(kp["twoway_lender_signal_sd"] / kp["twoway_msa_signal_sd"], 1))
mac("twMedianMSA", fmt_int(kp["twoway_median_msas_per_lender"]))
mac("hhiMedian", fmt_int(kp["hhi_median"])); mac("hhiPten", fmt_int(kp["hhi_p10"])); mac("hhiPninety", fmt_int(kp["hhi_p90"]))
mac("rhoHHI", fmt(kp["hhi_spearman_oe"], 2).replace("-", "$-$"))
sens = rd(os.path.join(P, "sq4_sensitivity.csv"))
mac("tauSensMin", fmt(sens.tau.min(), 2)); mac("tauSensMax", fmt(sens.tau.max(), 2))
ag = rd(os.path.join(P, "sq4_oe_by_agency.csv")).set_index("agency")
mac("agHUD", fmt(ag.loc["HUD", "median"], 2)); mac("agCFPB", fmt(ag.loc["CFPB", "median"], 2))
mac("nHUD", fmt_int(ag.loc["HUD", "n"]))

# prediction
pm = rd(os.path.join(P, "pred_metrics.csv")).set_index("model")
mac("aucLR", fmt(pm.loc["p_lr", "roc_auc"], 3)); mac("aucGB", fmt(pm.loc["p_gbt", "roc_auc"], 3))
mac("prLR", fmt(pm.loc["p_lr", "pr_auc"], 3)); mac("prGB", fmt(pm.loc["p_gbt", "pr_auc"], 3))
mac("brLR", fmt(pm.loc["p_lr", "brier"], 4)); mac("brGB", fmt(pm.loc["p_gbt", "brier"], 4))
mac("slLR", fmt(pm.loc["p_lr", "calibration_slope"], 3)); mac("slGB", fmt(pm.loc["p_gbt", "calibration_slope"], 3))
mac("aucMLlib", fmt(ks["pred_mllib_lr_roc_auc"], 3)); mac("corrMLlib", fmt(ks["pred_corr_mllib_vs_sklearn_lr"], 4))
mac("rhoPredLR", fmt(kp["pred_spearman_cells_vs_lr"], 3)); mac("rhoPredGB", fmt(kp["pred_spearman_cells_vs_gbt"], 3))
pmo = rd(os.path.join(P, "pred_missingness_by_outcome.csv")).set_index("field")
mac("missTermDen", pct(pmo.loc["term_num", "denied"])); mac("missTermApp", pct(pmo.loc["term_num", "approved"]))
mac("missDTIDen", pct(pmo.loc["dti_ordinal", "denied"])); mac("missDTIApp", pct(pmo.loc["dti_ordinal", "approved"]))
mac("baseRate", pct(kp["national_O"] / eq1["usable"]))

# runtime
mac("rtPandas", fmt_int(kp["runtime_s"] / 60)); mac("rtSpark", fmt_int(ks["runtime_s"] / 60))
mac("sparkInitMain", fmt(ks["spark_init_s"], 1))

# ---------------------------------------------------------------- performance
pmem = rd(os.path.join(PF, "perf_memory.csv"))
psum = rd(os.path.join(PF, "perf_summary.csv"))
meta = json.load(open(os.path.join(PF, "perf_meta.json")))
SIZES = ["10%", "25%", "50%", "100%", "x2", "x4", "x8", "x16"]
rows = {s: meta["sizes"][s]["rows"] for s in SIZES}


def med(engine, size, measure):
    q = psum[(psum.engine == engine) & (psum["size"] == size) & (psum.measure == measure)]
    return q["median"].iloc[0] if len(q) else np.nan


perf = []
for s in SIZES:
    load = med("pandas", s, "load")
    d = {"size": s, "rows": rows[s]}
    for w in ["W1", "W2", "W3"]:
        d[f"pd_{w}"] = med("pandas", s, w)
        d[f"pd_{w}_cold"] = load + d[f"pd_{w}"] if not np.isnan(load) else np.nan
        d[f"sp_{w}_nocache"] = med("spark", s, f"{w}_nocache")
        d[f"sp_{w}_cache"] = med("spark", s, f"{w}_cache")
    d["sp_init"] = med("spark", s, "init")
    d["sp_cache_build"] = med("spark", s, "cache_build")
    d["pd_load"] = load
    m1 = pmem[(pmem.engine == "pandas") & (pmem["size"] == s)]
    m2 = pmem[(pmem.engine == "spark") & (pmem["size"] == s)]
    d["pd_mem"] = m1.peak_rss_gb.iloc[0] if len(m1) else np.nan
    d["pd_status"] = m1.status.iloc[0] if len(m1) else ""
    d["sp_mem"] = m2.peak_rss_gb.iloc[0] if len(m2) else np.nan
    perf.append(d)
perf = pd.DataFrame(perf).set_index("size")
perf.to_csv(os.path.join(OUT, "perf_table_built.csv"))
oom = meta["pandas_oom"][0]
mac("oomSize", "16"); mac("oomRows", fmt(oom["rows"] / 1e6, 1)); mac("oomPeak", fmt(oom["peak_rss_gb"], 1))
mac("budget", fmt(meta["env"]["memory_budget_gb"], 0)); mac("ramTotal", fmt(meta["env"]["ram_total_gb"], 1))
mac("pdMemEight", fmt(perf.loc["x8", "pd_mem"], 1)); mac("spMemEight", fmt(perf.loc["x8", "sp_mem"], 1))
mac("spMemSixteen", fmt(perf.loc["x16", "sp_mem"], 1)); mac("pdMemFull", fmt(perf.loc["100%", "pd_mem"], 1))
mac("spMemTen", fmt(perf.loc["10%", "sp_mem"], 1)); mac("pdMemTen", fmt(perf.loc["10%", "pd_mem"], 2))
w0 = pmem[pmem["size"] == "W0"].set_index("label")
mac("wzPyarrowPeak", fmt(w0.loc["pandas_W0_pyarrow", "peak_rss_gb"], 1))
mac("wzCPeak", fmt(w0.loc["pandas_W0_c", "peak_rss_gb"], 1))
mac("wzSparkPeak", fmt(w0.loc["spark_W0", "peak_rss_gb"], 1))
mac("wzSparkTime", fmt(med("spark", "W0", "W0_spark"), 1))
mac("wzSparkInit", fmt(med("spark", "W0", "init"), 1))
mac("initMin", fmt(perf.sp_init.min(), 1)); mac("initMax", fmt(perf.sp_init.max(), 1))
mac("cbFull", fmt(perf.loc["100%", "sp_cache_build"], 1)); mac("cbEight", fmt(perf.loc["x8", "sp_cache_build"], 1))
mac("pdPerMillion", fmt((perf.loc["x8", "pd_mem"] - perf.loc["100%", "pd_mem"]) / ((rows["x8"] - rows["100%"]) / 1e6), 2))
eqp = meta["equality_100pct"]
mac("perfEqThree", f"{max(eqp['W3_oe_sum_rel_diff'], eqp['W3_lender_sd_rel_diff']):.1e}".replace("e-", r"\times10^{-") + "}")


def speed(a, b):
    return a / b


mac("spWOneEight", fmt(speed(perf.loc["x8", "pd_W1_cold"], perf.loc["x8", "sp_W1_nocache"]), 1))
mac("spWThreeEight", fmt(speed(perf.loc["x8", "pd_W3_cold"], perf.loc["x8", "sp_W3_nocache"]), 1))
mac("spWThreeTen", fmt(speed(perf.loc["10%", "pd_W3_cold"], perf.loc["10%", "sp_W3_nocache"]), 2))
mac("spWOneTen", fmt(speed(perf.loc["10%", "pd_W1_cold"], perf.loc["10%", "sp_W1_nocache"]), 2))

# performance table (W1 and W3; Pandas cold vs Spark no cache; speedup = T_pandas / T_spark)
lines = []
for s in SIZES:
    d = perf.loc[s]
    lab = s.replace("x", r"$\times$") if s.startswith("x") else s.replace("%", r"\%")
    cells = [lab, f"{d.rows / 1e6:.1f}"]
    for w in ["W1", "W3"]:
        a, b = d[f"pd_{w}_cold"], d[f"sp_{w}_nocache"]
        if np.isnan(a):
            cells += ["OOM", f"{b:.2f}", "Spark", "--"]
        else:
            cells += [f"{a:.2f}", f"{b:.2f}", "Pandas" if a < b else "Spark", f"{a / b:.2f}"]
    pm_ = "OOM" if d.pd_status == "oom_budget" else f"{d.pd_mem:.1f}"
    cells += [pm_, f"{d.sp_mem:.1f}"]
    lines.append(" & ".join(cells) + r" \\")
open(os.path.join(TAB, "perf_main.tex"), "w").write("\n".join(lines) + "\n")

# warm / cache table
lines = []
for s in SIZES:
    d = perf.loc[s]
    lab = s.replace("x", r"$\times$") if s.startswith("x") else s.replace("%", r"\%")
    def g(v):
        return "OOM" if np.isnan(v) else f"{v:.2f}"
    cells = [lab, g(d.pd_load), g(d.pd_W1), g(d.pd_W2), g(d.pd_W3), f"{d.sp_init:.1f}", f"{d.sp_cache_build:.1f}",
             f"{d.sp_W1_cache:.2f}", f"{d.sp_W1_nocache:.2f}", f"{d.sp_W2_cache:.2f}", f"{d.sp_W2_nocache:.2f}",
             f"{d.sp_W3_cache:.2f}", f"{d.sp_W3_nocache:.2f}"]
    lines.append(" & ".join(cells) + r" \\")
open(os.path.join(TAB, "perf_detail.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- correctness table (values from both engines)
def both(name, key, col, cond=None, kind="float", d=4):
    a = rd(os.path.join(P, f"{name}.csv"), dtype=str, keep_default_na=False)
    b = rd(os.path.join(S, f"{name}.csv"), dtype=str, keep_default_na=False)
    if cond is not None:
        a, b = a[cond(a)], b[cond(b)]
    va, vb = float(a[col].iloc[0]), float(b[col].iloc[0])
    if kind == "int":
        return fmt_int(va), fmt_int(vb), va == vb
    return f"{va:.{d}f}", f"{vb:.{d}f}", abs(va - vb) <= 1e-9 * max(1, abs(va))


rows_eq = []
x = both("eq_1_filter_counts", None, "rows", lambda t: t.step == "usable", "int")
rows_eq.append(("(1) Usable decisioned applications (last filter)", *x))
x = both("eq_2_special_codes", None, "count", lambda t: (t.field == "debt_to_income_ratio") & (t.code == "NA"), "int")
rows_eq.append((r"(2) \fd{debt_to_income_ratio} coded NA, all records", *x))
x = both("eq_3_action_taken", None, "share", lambda t: t.action_taken == "3", d=6)
rows_eq.append(("(3) Share of records denied (action 3)", *x))
x = both("eq_4_amounts_by_loan_type", None, "mean", lambda t: (t["sample"] == "decisioned") & (t.variable == "income") & (t.loan_type == "1"), d=4)
rows_eq.append(("(4) Mean income, conventional decisioned (\\$000)", *x))
x = both("eq_4_amounts_by_loan_type", None, "median_exact", lambda t: (t["sample"] == "decisioned") & (t.variable == "loan_amount") & (t.loan_type == "2"), "int")
rows_eq.append(("(4) Median loan amount, FHA decisioned (\\$)", *x))
x = both("eq_5_denial_by_state", None, "denial_rate", lambda t: t.state_code == "FL", d=6)
rows_eq.append(("(5) Denial rate, Florida", *x))
x = both("eq_6_top50_lenders", None, "oe", lambda t: t["rank"] == "1", d=6)
rows_eq.append(("(6) $O/E$ of the largest lender", *x))
for k, label, dd in [("national_oe", "National $O/E$ (usable sample)", 6), ("tau_lender", r"$\tau$ of lender $O/E$", 6),
                     ("twoway_lender_signal_sd", "Two-way lender effect SD (signal)", 6),
                     ("hhi_spearman_oe", r"Spearman $\rho$, MSA HHI vs $O/E$", 6)]:
    va, vb = kp[k], ks[k]
    rows_eq.append((label, sg(va, dd), sg(vb, dd), abs(va - vb) <= 1e-6 * max(1, abs(va))))
lines = [f"{q} & {a} & {b} & {'yes' if ok else 'NO'} \\\\" for q, a, b, ok in rows_eq]
open(os.path.join(TAB, "eq_values.tex"), "w").write("\n".join(lines) + "\n")
assert all(r[3] for r in rows_eq)

es = rd(os.path.join(EQ, "equivalence_summary.csv"))
lab = {
    "(1) Row counts after each filter": "Row counts after each filter",
    "(2) Special-code counts, 10 fields": "Special-code counts, 10 key fields",
    "(2b) Field profile, all 99 fields (codes and numeric values)": "Field profile, all 99 fields",
    "(3) action_taken counts and shares": r"\fd{action_taken} counts and shares",
    "(4) Mean and median of income, loan amount, rate spread by loan type": "Means and exact medians by loan type",
    "(4b) Rate spread: numeric, NA, Exempt, out of range": "Rate spread code counts by loan type",
    "(5) Denial rate by state (all codes, including NA)": "Denial rate by state, incl.\\ missing state",
    "(6) Aggregates of the 50 largest lenders": "Aggregates of the 50 largest lenders",
    "(4) percentile_approx vs exact median (Spark)": r"Spark \fd{percentile_approx} vs exact median",
    "Key analysis numbers (SQ1-SQ4, two-way, HHI)": "Key analysis numbers (SQ1--SQ4, two-way, HHI)",
}
lines = []
tot = 0
for _, r_ in es.iterrows():
    q = lab.get(r_["Quantity"], r_["Quantity"])
    md = r_["Max relative difference"]
    mds = "0" if md == 0 else (f"{md:.1e}".split("e")[0] + r"\,$\times10^{" + str(int(f"{md:.1e}".split("e")[1])) + "}$")
    note = "report only" if "percentile_approx" in r_["Quantity"] else ("yes" if r_["Agree"] else "NO")
    lines.append(f"{q} & {int(r_['Values compared'])} & {mds} & {note} \\\\")
    if "percentile_approx" not in r_["Quantity"]:
        tot += int(r_["Values compared"])
open(os.path.join(TAB, "eq_summary.tex"), "w").write("\n".join(lines) + "\n")
mac("eqTotal", fmt_int(tot))
mac("eqMaxDiff", f"{es[~es.Quantity.str.contains('percentile_approx')]['Max relative difference'].max():.1e}".split("e")[0])
assert es[~es.Quantity.str.contains("percentile_approx")]["Max relative difference"].max() < 1e-11
mac("approxErr", fmt(100 * es[es.Quantity.str.contains("percentile_approx")]["Max relative difference"].iloc[0], 1))
mac("nKeyNumbers", fmt_int(es[es.Quantity.str.startswith("Key")]["Values compared"].iloc[0]))

# ---------------------------------------------------------------- product table
lines = []
for t, lab_ in [("conventional", "Conventional"), ("FHA", "FHA"), ("VA", "VA"), ("USDA", "USDA")]:
    o = out_t.loc[t]
    lines.append(" & ".join([lab_, fmt_int(o.n_applications), pct(o.originated), pct(o.withdrawn + o.incomplete),
                             fmt_int(dt.loc[t, "n_decisioned"]), pct(dt.loc[t, "denial_rate"]),
                             fmt_int(amt.loc[t, "median"] / 1000), sg(rs_t.loc[t, "mean"]), sg(rs_t.loc[t, "median"])]) + r" \\")
open(os.path.join(TAB, "products.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Kitagawa table
def krow(label, g, comp, rate, ata, atb, hi, n):
    return f"{label} & {pp(g)} & {pp(comp)} & {pp(rate)} & {pp(ata)} & {pp(atb)} & {int(hi)}/{int(n)} \\\\"


lines = [
    krow("FHA vs conventional", f.gap, f.composition, f.rate, f.rate_gap_at_a_mix, f.rate_gap_at_b_mix, f.cells_a_higher, f.n_common_cells),
    krow("Cash-out refi. vs purchase" if STYLE == "acm" else "Cash-out refinance vs purchase", c.gap, c.composition, c.rate, c.rate_gap_at_a_mix, c.rate_gap_at_b_mix, c.cells_a_higher, c.n_common_cells),
    krow("Refinance vs purchase", r.gap, r.composition, r.rate, r.rate_gap_at_a_mix, r.rate_gap_at_b_mix, r.cells_a_higher, r.n_common_cells),
    krow("LMI vs other tracts", la.gap, la.composition, la.rate, la.rate_gap_at_a_mix, la.rate_gap_at_b_mix, la.cells_a_higher, la.n_common_cells),
]
open(os.path.join(TAB, "kitagawa.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- jump test table
lines = []
for scope, lab_ in [("1", "Conventional"), ("2", "FHA"), ("3", "VA"), ("all", "All")]:
    cells = [lab_]
    for jump in ["49 -> 50-60", "42 -> 43", "44 -> 45", "39 -> 40", "46 -> 47"]:
        e, lo, hi = jget(scope, jump, "excess"), jget(scope, jump, "ci_low"), jget(scope, jump, "ci_high")
        sig = "" if (lo <= 0 <= hi) else "$^{*}$"
        cells.append(f"{pp(e, 1 if jump.startswith('49') else 2)}{sig}")
    cells.append(fmt(bset[scope], 2))
    lines.append(" & ".join(cells) + r" \\")
open(os.path.join(TAB, "jump.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- prediction table
lines = []
for m, lab_ in [("p_lr", "Logistic (scikit-learn)" if STYLE == "acm" else "Logistic regression (scikit-learn)"), ("p_gbt", "Gradient boosting" if STYLE == "acm" else "Histogram gradient boosting")]:
    q = pm.loc[m]
    lines.append(f"{lab_} & {q.roc_auc:.3f} & {q.pr_auc:.3f} & {q.brier:.4f} & {q.calibration_in_the_large:+.4f} & {q.calibration_slope:.3f} \\\\")
lines.append(("Logistic (MLlib)" if STYLE == "acm" else "Logistic regression (Spark MLlib)") + f" & {ks['pred_mllib_lr_roc_auc']:.3f} & {ks['pred_mllib_lr_pr_auc']:.3f} & -- & {ks['pred_mllib_lr_calibration_in_the_large']:+.4f} & -- \\\\")
open(os.path.join(TAB, "pred.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- sensitivity table (appendix)
lines = []
SENS_LAB = {
    "cell_min_n=20": "Cell minimum 20 (fallback below)", "cell_min_n=30": "Cell minimum 30 (main)",
    "cell_min_n=50": "Cell minimum 50",
    "n_dec>=250, share>=0.8": r"Lenders $\ge$250 decisioned, $\ge$80\% usable",
    "n_dec>=500, share>=0.8": r"Lenders $\ge$500 decisioned, $\ge$80\% usable (main)",
    "n_dec>=1000, share>=0.8": r"Lenders $\ge$1,000 decisioned, $\ge$80\% usable",
    "n_dec>=500, share>=0.7": r"Lenders $\ge$500 decisioned, $\ge$70\% usable",
    "n_dec>=500, share>=0.9": r"Lenders $\ge$500 decisioned, $\ge$90\% usable",
    "drop lenders with >=10 denials and <50% usable": r"Drop lenders with $\ge$10 denials and $<$50\% usable",
    "drop lenders with >=10 denials and <30% usable": r"Drop lenders with $\ge$10 denials and $<$30\% usable",
    "within conventional": "Within conventional loans", "within FHA": "Within FHA loans"}
seen_ = set()
for _, q in sens.iterrows():
    if q.variant in seen_:
        continue
    seen_.add(q.variant)
    v = SENS_LAB[q.variant]
    lines.append(f"{v} & {int(q.groups)} & {q.median_oe:.2f} & {q.tau:.3f} & {q.share_outside_95 * 100:.0f} \\\\")
open(os.path.join(TAB, "sensitivity.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- data quality table
dq = rd(os.path.join(P, "dq_special_codes.csv"))
prof = rd(os.path.join(P, "dq_field_profile.csv")).set_index("field")
dqs = rd(os.path.join(P, "dq_summary.csv")).set_index("item")
mac("nCLTVout", fmt_int(dqs.loc["decisioned: CLTV numeric but outside (0, 250]", "n"]))
mac("nDTImiss", fmt_int(dqs.loc["decisioned: DTI missing", "n"]))
mac("nCountyBad", fmt_int(dqs.loc["core: county_code not 5 digits", "n"]))
mac("nTractZero", fmt_int(dqs.loc["core: tract ratio 0 (treated as missing)", "n"]))
mac("nIncNA", fmt_int(dqs.loc["decisioned: income NA", "n"]))
mac("nIncNeg", fmt_int(dqs.loc["decisioned: income <= 0", "n"]))
mac("nIncHigh", fmt_int(dqs.loc["decisioned: income > 10,000 (flag only)", "n"]))
mac("exemptN", fmt_int(prof.loc["debt_to_income_ratio", "Exempt"]))
mac("dtiNAall", fmt_int(prof.loc["debt_to_income_ratio", "NA"]))
mac("rsNAall", fmt_int(prof.loc["rate_spread", "NA"]))
mac("incomeNAall", fmt_int(prof.loc["income", "NA"]))
mad = rd(os.path.join(P, "dq_missing_by_action.csv")).set_index("action_taken")
mac("dtiMissWithdrawn", fmt(mad.loc[4, "dti_missing"], 0))
mac("dtiMissOrig", fmt(mad.loc[1, "dti_missing"], 1))
mac("dtiMissDenied", fmt(mad.loc[3, "dti_missing"], 1))
_hm = rd(os.path.join(P, "sq2_heatmap_dti_cltv.csv")).set_index(["dti_band", "cltv_band"])
mac("drHighDTIcltvHundred", fmt_int(100 * _hm.loc[("50-60", "95-100"), "rate"]))
_rc = rd(os.path.join(P, "sq2_risk_cells.csv")).groupby(["dti_band", "cltv_band"])[["n", "fha"]].sum()
mac("fhaHighDTIcltvHundred", fmt_int(100 * _rc.loc[("50-60", "95-100"), "fha"] / _rc.loc[("50-60", "95-100"), "n"]))

# ---- audit additions (10 Nov 2026): numbers that qualify the main claims
_kn = json.load(open(os.path.join(P, "key_numbers.json")))
_dec = int(str(MAC["nDecisioned"]).replace(",", "")); _usa = int(str(MAC["nUsable"]).replace(",", ""))
_den_dec = _kn["denial_rate_core"] * _dec
mac("nExcluded", fmt_int(_dec - _usa))
mac("drExcluded", fmt(100 * (_den_dec - _kn["national_O"]) / (_dec - _usa), 1))
_txp = rd(os.path.join(P, "sq1_denial_type_x_purpose.csv"))
_tot = _txp.groupby("type")[["n_decisioned", "n_denied"]].sum()
_pur = rd(os.path.join(P, "sq1_denial_by_purpose.csv")).set_index("purpose")
mac("gapFHAdecisioned", fmt(100 * (_tot.loc["FHA", "n_denied"] / _tot.loc["FHA", "n_decisioned"]
                                   - _tot.loc["conventional", "n_denied"] / _tot.loc["conventional", "n_decisioned"]), 1))
mac("gapCOdecisioned", fmt(100 * (_pur.loc["cash_out_refinance", "denial_rate"] - _pur.loc["purchase", "denial_rate"]), 1))
_co = _txp[_txp["purpose"] == "cash_out_refinance"].set_index("type")["n_decisioned"]
mac("coShareConv", fmt_int(100 * _co["conventional"] / _tot.loc["conventional", "n_decisioned"]))
mac("coShareFHA", fmt_int(100 * _co["FHA"] / _tot.loc["FHA", "n_decisioned"]))
_lo = rd(os.path.join(P, "sq4_lender_oe.csv"))
mac("topLenderShare", fmt_int(100 * _lo["n"].max() / _usa))
_mo = rd(os.path.join(P, "sq3_msa_oe.csv"))
_mw = np.average(_mo["oe"], weights=_mo["n"])
mac("tauMSAW", fmt(max(np.average((_mo["oe"] - _mw) ** 2, weights=_mo["n"]) - np.average(_mo["sampling_var"], weights=_mo["n"]), 0) ** .5, 2))
mac("twVarRatio", fmt_int((_kn["twoway_lender_signal_sd"] / _kn["twoway_msa_signal_sd"]) ** 2))
mac("twLeTwoShare", fmt(100 * _kn["twoway_share_lenders_le2_msas"], 1))
_jt = rd(os.path.join(P, "sq2_dti_jump_test.csv"))
_jf = _jt[_jt["scope"].astype(str) == "2"].set_index("jump")["excess"]
mac("jumpFHAFortythree", pp(_jf["42 -> 43"], 2)); mac("jumpFHAFortyfive", pp(_jf["44 -> 45"], 2))
# compute-only comparison: Pandas with data already in memory against Spark reading Parquet
mac("warmWOneFull", fmt(perf.loc["100%", "pd_W1"], 2)); mac("spWOneFull", fmt(perf.loc["100%", "sp_W1_nocache"], 2))
mac("warmSpWOneEight", fmt(perf.loc["x8", "pd_W1"] / perf.loc["x8", "sp_W1_nocache"], 1))
mac("warmSpWThreeEight", fmt(perf.loc["x8", "pd_W3"] / perf.loc["x8", "sp_W3_nocache"], 1))
mac("warmWThreeFour", fmt(perf.loc["x4", "pd_W3"], 1)); mac("spWThreeFour", fmt(perf.loc["x4", "sp_W3_nocache"], 1))
mac("loadShareEight", fmt_int(100 * perf.loc["x8", "pd_load"] / perf.loc["x8", "pd_W1_cold"]))

# ---------------------------------------------------------------- write macros
with open(os.path.join(OUT, "numbers.tex"), "w") as fh_:
    fh_.write("% generated by build_assets.py from the Kaggle run outputs; do not edit by hand\n")
    for k, v in MAC.items():
        fh_.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(MAC, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, ensure_ascii=False)

# ================================================================ FIGURES
# F1 denial by purpose x type
fig, ax = plt.subplots(figsize=(6.6, 2.0) if STYLE == "acm" else (6.6, 2.4))
purps = [("purchase", "Purchase"), ("refinance", "Refinance"), ("cash_out_refinance", "Cash-out refi."),
         ("home_improvement", "Home improv."), ("other", "Other")]
types = [("conventional", "Conventional", C_CONV), ("FHA", "FHA", C_FHA), ("VA", "VA", C_VA)]
w = 0.26
for i, (t, lab_, col) in enumerate(types):
    vals = [txp.loc[(t, p_), "denial_rate"] * 100 for p_, _ in purps]
    ns = [txp.loc[(t, p_), "n_decisioned"] for p_, _ in purps]
    xs = np.arange(len(purps)) + (i - 1) * w
    ax.bar(xs, vals, w, color=col, label=lab_)
    for xx, v, n in zip(xs, vals, ns):
        ax.text(xx, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(np.arange(len(purps)))
ax.set_xticklabels([p[1] for p in purps])
ax.set_ylabel("Denial rate (%)")
ax.set_ylim(0, 62)
ax.legend(frameon=False, ncol=3, loc="upper left")
if STYLE != "acm":  # in the ACM layout the caption carries the title
    ax.set_title("Denial rate by loan purpose and loan type, decisioned core applications (2024)")
fig.savefig(os.path.join(FIG, "fig_products.pdf"))
plt.close(fig)

# F2 DTI: (a) all 19 levels; (b) 36..49, 50-60 by type with counts
jtab = rd(os.path.join(P, "sq2_dti_jump_table.csv"))
jtab["scope"] = jtab["scope"].astype(str)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.05, 1]})
labs = list(d19.dti)
a1.plot(range(len(labs)), d19.denial_rate * 100, color=C_ALL, marker="o", ms=2.5, lw=1)
a1.set_xticks(range(len(labs)))
a1.set_xticklabels([{"<20": "<20", "20-<30": "20s", "30-<36": "30-35", "36": "36", "43": "43", "49": "49",
                     "50-60": "50-60", ">60": ">60"}.get(l, "") for l in labs], rotation=90)
a1.set_ylabel("Denial rate (%)")
a1.set_xlabel("Debt-to-income ratio (%)")
a1.set_title("(a) All loan types, 19 DTI values")
a1.set_ylim(0, 92)
a1.axvline(16.5, color="#999999", lw=0.6, ls="--")
a1.annotate(f"{d19.denial_rate.iloc[-2] * 100:.1f}", (17, d19.denial_rate.iloc[-2] * 100), textcoords="offset points", xytext=(-18, 3), fontsize=7)
a1.annotate(f"{d19.denial_rate.iloc[-1] * 100:.1f}", (18, d19.denial_rate.iloc[-1] * 100), textcoords="offset points", xytext=(-16, -2), fontsize=7)
for scope, lab_, col in [("1", "Conventional", C_CONV), ("2", "FHA", C_FHA), ("3", "VA", C_VA)]:
    q = jtab[jtab.scope == scope].sort_values("level")
    a2.plot(q.level, q.rate * 100, color=col, marker="o", ms=2.5, lw=1, label=lab_)
a2.set_xticks(list(range(4, 19)))
a2.set_xticklabels([str(v) if v in (36, 40, 43, 46, 49) else ("50-60" if v == "50-60" else "") for v in
                    list(range(36, 50)) + ["50-60"]], rotation=90)
a2.set_yscale("log")
a2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a2.set_yticks([5, 10, 20, 50, 80])
a2.set_yticklabels(["5", "10", "20", "50", "80"])
a2.set_ylabel("Denial rate (%, log scale)")
a2.set_xlabel("Debt-to-income ratio (%)")
a2.axvline(17.5, color="#999999", lw=0.6, ls="--")
a2.axvline(10.5, color="#bbbbbb", lw=0.6, ls=":")
a2.set_title("(b) DTI 36-49 and 50-60 by loan type")
a2.legend(frameon=False, loc="upper left")
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(FIG, "fig_dti.pdf"))
plt.close(fig)

# F2b counts by DTI value, conventional (bunching) -> small panel in appendix
fig, ax = plt.subplots(figsize=(3.3, 2.1))
q = jtab[(jtab.scope == "1") & (jtab.level <= 17)].sort_values("level")
ax.bar(q.level, q.n / 1000, color=C_CONV)
ax.set_xticks(list(range(4, 18, 2)))
ax.set_xticklabels([str(v) for v in range(36, 50, 2)])
ax.set_ylabel("Applications (000)")
ax.set_xlabel("DTI (%), conventional")
ax.set_title("Conventional applications by DTI value")
fig.savefig(os.path.join(FIG, "fig_bunching.pdf"))
plt.close(fig)

# F3 heatmap DTI x CLTV
hm = rd(os.path.join(P, "sq2_heatmap_dti_cltv.csv"))
db_order = ["<30", "30-<36", "36-42", "43-49", "50-60", ">60"]
cl_order = ["<=80", "80-90", "90-95", "95-100", ">100"]
mat = hm.pivot(index="dti_band", columns="cltv_band", values="rate").loc[db_order, cl_order] * 100
nn = hm.pivot(index="dti_band", columns="cltv_band", values="n").loc[db_order, cl_order]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
im = ax.imshow(np.log10(mat.values), cmap="Greys", aspect="auto", vmin=np.log10(4), vmax=np.log10(120))
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 35 else "black")
ax.set_xticks(range(len(cl_order)))
ax.set_xticklabels(["$\\leq$80", "80-90", "90-95", "95-100", ">100"])
ax.set_yticks(range(len(db_order)))
ax.set_yticklabels(["<30", "30-35", "36-42", "43-49", "50-60", ">60"])
ax.set_xlabel("Combined loan-to-value (%)")
ax.set_ylabel("Debt-to-income (%)")
ax.set_title("Denial rate (%) by DTI and CLTV band")
ax.grid(False)
fig.savefig(os.path.join(FIG, "fig_heatmap.pdf"))
plt.close(fig)

# F4 funnel plots
lo = rd(os.path.join(P, "sq4_lender_oe.csv"))
ms = rd(os.path.join(P, "sq3_msa_oe.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
for ax, d_, title, col in [(axes[0], lo, f"(a) Lenders (n = {len(lo)})", C_CONV), (axes[1], ms, f"(b) MSAs (n = {len(ms)})", C_FHA)]:
    cst = np.median(d_.V / d_.E)
    e = np.logspace(np.log10(d_.E.min() * 0.8), np.log10(d_.E.max() * 1.2), 200)
    for z, ls in [(1.96, "--"), (3.09, ":")]:
        ax.plot(e, 1 + z * np.sqrt(cst / e), color="#555555", lw=0.7, ls=ls)
        ax.plot(e, np.clip(1 - z * np.sqrt(cst / e), 0, None), color="#555555", lw=0.7, ls=ls)
    ax.scatter(d_.E, d_.oe, s=4, color=col, alpha=0.6, linewidths=0)
    ax.axhline(1, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Expected denials $E$ (log scale)")
    ax.set_title(title)
    ax.set_ylim(0, 3.2)
axes[0].set_ylabel("Observed / expected denials ($O/E$)")
axes[1].text(0.98, 0.95, "dashed: 95% limits\ndotted: 99.8% limits", transform=axes[1].transAxes, ha="right", va="top", fontsize=7)
fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(FIG, "fig_funnel.pdf"))
plt.close(fig)
mac_out_lender_above3 = int((lo.oe > 3.2).sum())

# F5 performance
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7))
xr = perf.rows / 1e6
for wl, mk in [("W1", "o"), ("W3", "s")]:
    a1.plot(xr, perf[f"pd_{wl}_cold"], color=C_PD, marker=mk, ms=3, lw=1, label=f"Pandas {wl}")
    a1.plot(xr, perf[f"sp_{wl}_nocache"], color=C_SP, marker=mk, ms=3, lw=1, ls="--", label=f"Spark {wl}")
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("Rows (millions, log scale)")
a1.set_ylabel("Median time (s, log scale)")
a1.set_title("(a) Time per workload, read from Parquet")
a1.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
a1.text(0.98, 0.04, "Pandas out of memory at ×16", transform=a1.transAxes, ha="right", va="bottom", fontsize=7)
pm_ok = perf[perf.pd_status == "ok"]
a2.plot(pm_ok.rows / 1e6, pm_ok.pd_mem, color=C_PD, marker="o", ms=3, lw=1, label="Pandas")
a2.plot([pm_ok.rows.iloc[-1] / 1e6, perf.loc["x16", "rows"] / 1e6], [pm_ok.pd_mem.iloc[-1], perf.loc["x16", "pd_mem"]],
        color=C_PD, lw=1, ls=":")
a2.scatter([perf.loc["x16", "rows"] / 1e6], [perf.loc["x16", "pd_mem"]], marker="x", color=C_PD, s=30, zorder=5)
a2.plot(xr, perf.sp_mem, color=C_SP, marker="o", ms=3, lw=1, ls="--", label="Spark (incl. JVM)")
a2.axhline(meta["env"]["memory_budget_gb"], color="black", lw=0.7)
a2.text(0.6, meta["env"]["memory_budget_gb"] + 0.8, f"budget {meta['env']['memory_budget_gb']:.0f} GB (90% of RAM)", fontsize=7)
a2.set_xscale("log")
a2.set_xlabel("Rows (millions, log scale)")
a2.set_ylabel("Peak memory (GB)")
a2.set_ylim(0, 33)
a2.set_title("(b) Peak memory per process")
a2.legend(frameon=False, fontsize=7, loc="center left")
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(FIG, "fig_perf.pdf"))
plt.close(fig)

# F6 calibration (appendix)
cal = rd(os.path.join(P, "pred_calibration_deciles.csv"))
fig, ax = plt.subplots(figsize=(3.2, 2.5))
for m, lab_, col, mk in [("p_lr", "Logistic regression", C_CONV, "o"), ("p_gbt", "Gradient boosting", C_FHA, "s")]:
    q = cal[cal.model == m]
    ax.plot(q.mean_pred * 100, q.observed * 100, marker=mk, ms=3, lw=1, color=col, label=lab_)
ax.plot([0, 70], [0, 70], color="#888888", lw=0.6, ls="--")
ax.set_xlabel("Mean predicted denial (%)")
ax.set_ylabel("Observed denial (%)")
ax.set_title("Calibration by decile, out-of-fold")
ax.legend(frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG, "fig_calibration.pdf"))
plt.close(fig)

# F7 two-way effects distribution (appendix)
le = rd(os.path.join(P, "sq4_two_way_lender_effects.csv"))
me = rd(os.path.join(P, "sq4_two_way_msa_effects.csv"))
fig, ax = plt.subplots(figsize=(3.3, 2.3))
bins = np.linspace(-0.3, 0.45, 61)
ax.hist(le.effect.clip(-0.3, 0.45) * 100, bins=bins * 100, color=C_CONV, alpha=0.75, label=f"Lender effects ({len(le)})", density=True)
ax.hist(me.effect.clip(-0.3, 0.45) * 100, bins=bins * 100, color=C_FHA, alpha=0.75, label=f"MSA effects ({len(me)})", density=True)
ax.set_xlabel("Effect on denial probability (points)")
ax.set_ylabel("Density")
ax.set_title("Two-way lender and MSA effects")
ax.legend(frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG, "fig_twoway.pdf"))
plt.close(fig)

print(f"{len(MAC)} macros; lenders with O/E > 3.2 not drawn: {mac_out_lender_above3}")
print(perf[["rows", "pd_W1_cold", "sp_W1_nocache", "pd_W3_cold", "sp_W3_nocache", "pd_mem", "sp_mem"]].round(2).to_string())
