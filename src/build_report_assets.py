"""
Build figures, tables and number macros for the HMDA 2024 report.

Every number printed in the report comes from the Kaggle run outputs:
  results/pandas       (Pandas notebook)
  results/spark        (PySpark notebook)
  results/perf         (performance notebook)
  results/equivalence  (equivalence notebook)
  results/external     (FFIEC Data Browser counts, verification only)
  results/independent  (DuckDB re-implementation, verification and dollar magnitudes)
  results/predictive   (calibration of the out-of-fold predictions by subgroup)
  results/sensitivity  (open-end exempt sensitivity run)

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
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "mathtext.fontset": "cm",
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
occ = rd(os.path.join(P, "dq_occupancy_by_type.csv")).set_index("type")
_occ_s = rd(os.path.join(S, "dq_occupancy_by_type.csv")).set_index("type")
assert (occ[["n_before_filter", "n_non_principal"]] == _occ_s.loc[occ.index, ["n_before_filter", "n_non_principal"]]).all().all()
assert occ["n_non_principal"].sum() == eq1["site_built_1_4_unit"] - eq1["principal_residence"]
mac("occConv", pct(occ.loc["conventional", "share_non_principal"]))
mac("occGovMax", pct(occ.loc[["FHA", "VA", "USDA"], "share_non_principal"].max(), 2))
mac("nNonPrincipal", fmt_int(eq1["site_built_1_4_unit"] - eq1["principal_residence"]))
mac("shareNonPrincipal", pct(1 - eq1["principal_residence"] / eq1["site_built_1_4_unit"]))
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
mac("nUSDAdec", fmt_int(dt.loc["USDA", "n_decisioned"]))

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
_ms = mlb[mlb.group == "missing_state"]                      # decisioned applications without a state
mac("nNoStateDec", fmt_int(_ms["n"].sum())); mac("drNoStateDec", pct((_ms["n"] * _ms["denial_rate"]).sum() / _ms["n"].sum()))
mpm = rd(os.path.join(P, "dq_missing_location_purpose_mix.csv")).set_index("group")
mac("noStatePurchase", pct(mpm.loc["missing_state", "purchase"], 0))
st = rd(os.path.join(P, "sq3_state_oe.csv")).sort_values("oe")
mac("stateLowName", st.iloc[0].state_code); mac("stateLowOE", fmt(st.iloc[0].oe, 2))
mac("stateHighName", st.iloc[-1].state_code); mac("stateHighOE", fmt(st.iloc[-1].oe, 2))
mac("fallbackShare", pct(kp["fallback_share"], 2)); mac("nExtCells", fmt_int(kp["n_ext_cells"]))
mac("minCellN", fmt_int(kp["smallest_cell_n"]))
_cv = kp["conv_twoway_last_iter_max_change"]
mac("twLastChange", (f"{_cv:.0e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}") if _cv > 0 else "0")
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
mac("rhoHHI", fmt(kp["hhi_spearman_oe"], 2))   # used inside math mode, so a plain minus sign is correct
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
mac("maxDiffMLlib", fmt(ks["pred_maxdiff_mllib_vs_sklearn_lr"], 2))
mac("meanDiffMLlib", fmt(ks["pred_meanabsdiff_mllib_vs_sklearn_lr"], 4))
mac("shareDiffMLlib", pct(ks["pred_share_absdiff_gt_0p05_mllib_vs_sklearn_lr"], 3))
_bp = rd(os.path.join(S, "pred_mllib_vs_sklearn_by_purpose.csv")).set_index("purpose")
MLLIB_BY_PURPOSE = _bp
mac("nDiffMLlib", fmt_int(_bp["n_abs_diff_gt_0p05"].sum()))
mac("nDiffMLlibNA", fmt_int(_bp.loc["not_applicable", "n_abs_diff_gt_0p05"])); mac("nPurposeNA", fmt_int(_bp.loc["not_applicable", "n"]))
mac("maxDiffMLlibOther", fmt(_bp.drop(index="not_applicable")["max_abs_diff"].max(), 3))
mac("rhoPredLR", fmt(kp["pred_spearman_cells_vs_lr"], 3)); mac("rhoPredGB", fmt(kp["pred_spearman_cells_vs_gbt"], 3))
_cal = rd(os.path.join(P, "pred_calibration_deciles.csv"))
_c0 = _cal[_cal["model"] == "p_lr"].sort_values("mean_pred").iloc[0]      # lowest decile of the logistic model
mac("calLRlowPred", pct(_c0["mean_pred"])); mac("calLRlowObs", pct(_c0["observed"]))
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
SIZES = list(meta["sizes"])                      # 10% ... x8, then x16, x32 added until Pandas stopped
rows = {s: meta["sizes"][s]["rows"] for s in SIZES}
assert meta["equality_100pct"]["all_equal"], "Pandas and Spark must agree at 100% before times are reported"


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
    d["pd_stage"] = m1.stage.iloc[0] if len(m1) else ""
    d["sp_mem"] = m2.peak_rss_gb.iloc[0] if len(m2) else np.nan
    perf.append(d)
perf = pd.DataFrame(perf).set_index("size")
perf.to_csv(os.path.join(OUT, "perf_table_built.csv"))
oom = meta["pandas_oom"][0]
assert oom["size"] == SIZES[-1], "largest size in SIZES must be the one where Pandas stopped"
assert str(oom["stage"]).startswith("load"), "text says Pandas stopped while loading the table"
LAST = SIZES[-2]                                  # largest size both engines completed
assert LAST == "x16" and (perf.drop(index=SIZES[-1]).pd_status == "ok").all()
mac("oomSize", oom["size"].lstrip("x")); mac("oomRows", fmt(oom["rows"] / 1e6, 1)); mac("oomPeak", fmt(oom["peak_rss_gb"], 1))
mac("oomSecs", fmt(oom["seconds"], 0))
mac("lastRows", fmt(rows[LAST] / 1e6, 1))
mac("budget", fmt(meta["env"]["memory_budget_gb"], 0)); mac("ramTotal", fmt(meta["env"]["ram_total_gb"], 1))
mac("pdMemEight", fmt(perf.loc["x8", "pd_mem"], 1)); mac("spMemEight", fmt(perf.loc["x8", "sp_mem"], 1))
mac("pdMemSixteen", fmt(perf.loc["x16", "pd_mem"], 1)); mac("spMemSixteen", fmt(perf.loc["x16", "sp_mem"], 1))
mac("spMemOom", fmt(perf.loc[SIZES[-1], "sp_mem"], 1)); mac("pdMemFull", fmt(perf.loc["100%", "pd_mem"], 1))
mac("spMemTen", fmt(perf.loc["10%", "sp_mem"], 1)); mac("pdMemTen", fmt(perf.loc["10%", "pd_mem"], 2))
w0 = pmem[pmem["size"] == "W0"].set_index("label")
assert w0.loc["pandas_W0_pyarrow", "status"] != "ok" and w0.loc["pandas_W0_c", "status"] != "ok"
assert w0.loc["pandas_W0_arrow", "status"] == "ok"
mac("wzPyarrowPeak", fmt(w0.loc["pandas_W0_pyarrow", "peak_rss_gb"], 1))
mac("wzCPeak", fmt(w0.loc["pandas_W0_c", "peak_rss_gb"], 1))
mac("wzArrowPeak", fmt(w0.loc["pandas_W0_arrow", "peak_rss_gb"], 1))
mac("wzArrowTime", fmt(med("pandas", "W0", "W0_arrow"), 1))
mac("wzSparkPeak", fmt(w0.loc["spark_W0", "peak_rss_gb"], 1))
mac("wzSparkTime", fmt(med("spark", "W0", "W0_spark"), 1))
mac("wzSparkInit", fmt(med("spark", "W0", "init"), 1))
mac("wzCsvGb", fmt(meta["env"]["w0_csv_gb"], 2))
mac("wzMemRatio", fmt(w0.loc["pandas_W0_arrow", "peak_rss_gb"] / w0.loc["spark_W0", "peak_rss_gb"], 0))
mac("initMin", fmt(perf.sp_init.min(), 1)); mac("initMax", fmt(perf.sp_init.max(), 1))
mac("cbFull", fmt(perf.loc["100%", "sp_cache_build"], 1)); mac("cbSixteen", fmt(perf.loc["x16", "sp_cache_build"], 1))
mac("cbOom", fmt(perf.loc[SIZES[-1], "sp_cache_build"], 1))
mac("pdPerMillion", fmt((perf.loc["x16", "pd_mem"] - perf.loc["100%", "pd_mem"]) / ((rows["x16"] - rows["100%"]) / 1e6), 2))
eqp = meta["equality_100pct"]
mac("perfEqThree", f"{max(eqp['W3_oe_sum_rel_diff'], eqp['W3_lender_sd_rel_diff']):.1e}".replace("e-", r"\times10^{-") + "}")

# ratios T_pandas / T_spark at 10% and at the largest common size
r_ = lambda s, a, b: perf.loc[s, a] / perf.loc[s, b]
mac("memWOneTen", fmt(r_("10%", "pd_W1", "sp_W1_cache"), 2)); mac("memWOneSixteen", fmt(r_("x16", "pd_W1", "sp_W1_cache"), 1))
mac("memWThreeTen", fmt(r_("10%", "pd_W3", "sp_W3_cache"), 2)); mac("memWThreeSixteen", fmt(r_("x16", "pd_W3", "sp_W3_cache"), 2))
mac("memWThreeEightInv", fmt(r_("x8", "sp_W3_cache", "pd_W3"), 1))
mac("spWOneSixteen", fmt(r_("x16", "pd_W1_cold", "sp_W1_nocache"), 1))
mac("spWThreeSixteen", fmt(r_("x16", "pd_W3_cold", "sp_W3_nocache"), 1))
mac("loadShareSixteen", fmt_int(100 * perf.loc["x16", "pd_load"] / perf.loc["x16", "pd_W1_cold"]))
mac("memWOneFullPd", fmt(perf.loc["100%", "pd_W1"], 2)); mac("memWOneFullSp", fmt(perf.loc["100%", "sp_W1_cache"], 2))
mac("memWThreeSixPd", fmt(perf.loc["x16", "pd_W3"], 1)); mac("memWThreeSixSp", fmt(perf.loc["x16", "sp_W3_cache"], 1))

# the crossover statements in the text, checked against the data (sizes where Pandas is faster)
both = [s for s in SIZES if perf.loc[s, "pd_status"] == "ok"]
fast = lambda a, b: [s for s in both if perf.loc[s, a] < perf.loc[s, b]]
assert fast("pd_W1", "sp_W1_cache") == ["10%", "25%", "50%", "x2"], fast("pd_W1", "sp_W1_cache")
assert abs(r_("100%", "pd_W1", "sp_W1_cache") - 1) < 0.1                       # 100%: tie
assert fast("pd_W3", "sp_W3_cache") == ["10%", "25%", "50%", "100%", "x2", "x4", "x8"], fast("pd_W3", "sp_W3_cache")
assert abs(r_("x16", "pd_W3", "sp_W3_cache") - 1) < 0.1                        # 16x: tie
assert fast("pd_W1_cold", "sp_W1_nocache") == ["10%", "25%"], fast("pd_W1_cold", "sp_W1_nocache")
assert fast("pd_W3_cold", "sp_W3_nocache") == ["10%", "25%", "50%", "100%", "x2"], fast("pd_W3_cold", "sp_W3_nocache")
big = [s for s in SIZES if rows[s] >= rows["x4"]]
assert all(perf.loc[s, f"sp_{w}_cache"] > perf.loc[s, f"sp_{w}_nocache"] for s in big for w in ["W1", "W2", "W3"])

_pr = pd.read_csv(os.path.join(PF, "perf_runs.csv"))
_sp = _pr.groupby(["engine", "size", "measure"]).seconds.agg(["min", "median", "max", "count"])
_sp = _sp[_sp["count"] >= 5]; _sp["spread"] = (_sp["max"] - _sp["min"]) / _sp["median"]
_sm = _sp.groupby(level=[0, 2]).spread.median()
mac("spreadPandas", fmt_int(100 * max(_sm[("pandas", w_)] for w_ in ["W1", "W2", "W3"])))
mac("spreadSparkLo", fmt_int(100 * min(_sm[("spark", f"{w_}_nocache")] for w_ in ["W1", "W2", "W3"])))
mac("spreadSparkHi", fmt_int(100 * max(_sm[("spark", f"{w_}_nocache")] for w_ in ["W1", "W2", "W3"])))


def _lab(s):
    return s.replace("x", r"$\times$") if s.startswith("x") else s.replace("%", r"\%")


# performance table: W1 and W3, each compared in memory (Pandas warm vs Spark cached) and from disk
# (Pandas cold vs Spark without cache); speedup = T_pandas / T_spark
def _cmp(a, b, stopped):
    if np.isnan(a):
        return (["stopped", f"{b:.2f}", "Spark only", "--"] if stopped else ["n.r.", f"{b:.2f}", "--", "--"])
    return [f"{a:.2f}", f"{b:.2f}", "Pandas" if a < b else "Spark", f"{a / b:.2f}"]


lines = []
for w, title in [("W1", r"W1: denial rate and mean loan amount by loan type $\times$ purpose"),
                 ("W3", r"W3: join, lender $O/E$, ranking, ten two-way iterations")]:
    lines.append(r"\multicolumn{12}{l}{\emph{" + title + r"}} \\")
    for s_ in SIZES:
        d = perf.loc[s_]
        stopped = d.pd_status == "oom_budget"
        cells = [_lab(s_), f"{d.rows / 1e6:.1f}"] + _cmp(d[f"pd_{w}"], d[f"sp_{w}_cache"], stopped) \
            + _cmp(d[f"pd_{w}_cold"], d[f"sp_{w}_nocache"], stopped)
        if w == "W1":
            cells += [f"{d.pd_mem:.1f}" + (r"$^{a}$" if stopped else ""), f"{d.sp_mem:.1f}"]
        else:
            cells += ["", ""]
        lines.append(" & ".join(cells) + r" \\")
    if w == "W1":
        lines.append(r"\midrule")
open(os.path.join(TAB, "perf_main.tex"), "w").write("\n".join(lines) + "\n")

# warm / cache table
lines = []
for s in SIZES:
    d = perf.loc[s]
    miss = "stopped" if d.pd_status == "oom_budget" else "n.r."
    def g(v):
        return miss if np.isnan(v) else f"{v:.2f}"
    cells = [_lab(s), g(d.pd_load), g(d.pd_W1), g(d.pd_W2), g(d.pd_W3), f"{d.sp_init:.1f}", f"{d.sp_cache_build:.1f}",
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
for k, label, dd in [("tau_lender", r"$\tau$ of lender $O/E$ (SQ4)", 6)]:
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
_mx = es[~es.Quantity.str.contains("percentile_approx")]["Max relative difference"].max()
assert _mx < 1e-9
mac("eqMaxDiff", f"{_mx:.1e}".split("e")[0]); mac("eqMaxExp", str(int(f"{_mx:.1e}".split("e")[1])))
mac("nKeyNumbers", fmt_int(es[es.Quantity.str.startswith("Key")]["Values compared"].iloc[0]))
# where the largest difference sits, and how large the two-way differences are (the text names both)
_ed = rd(os.path.join(EQ, "equivalence_details.csv"))
_worst = _ed.sort_values("rel_diff", ascending=False).iloc[0]
assert _worst["output"] == "key_numbers" and _worst["key"] in ("dti5_national_oe", "national_E", "national_oe", "E_nonfallback"), _worst
_tw = _ed[(_ed["output"] == "key_numbers") & _ed["key"].str.startswith("twoway_")]["rel_diff"].max()
assert _tw < 1e-15
# key numbers that Spark derives from the Pandas out-of-fold predictions (not independent computations)
_dep = [k for k in ("pred_spearman_cells_vs_lr", "pred_spearman_cells_vs_gbt") if k in set(_ed.loc[_ed["output"] == "key_numbers", "key"])]
mac("nKeyDependent", {0: "none", 1: "one", 2: "two"}[len(_dep)])
# percentile_approx against the exact median, from the Spark output: true relative and absolute difference
_e4 = rd(os.path.join(S, "eq_4_amounts_by_loan_type.csv")).dropna(subset=["median_exact", "median_approx"])
_abs = (_e4["median_exact"] - _e4["median_approx"]).abs()
mac("approxErr", fmt(100 * (_abs / _e4["median_exact"].abs()).max(), 2))
_wa = _e4.loc[_abs.idxmax()]
assert _wa["variable"] == "rate_spread"
mac("approxAbs", fmt(_abs.max(), 3))

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
    for jump in ["49 -> 50-60", "44 -> 45", "39 -> 40", "46 -> 47"]:
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
# field profile excerpt (appendix): all cells from dq_field_profile.csv
_pl = []
for f_ in ["debt_to_income_ratio", "rate_spread", "income", "combined_loan_to_value_ratio", "property_value", "loan_term"]:
    q_ = prof.loc[f_]; oth = int(q_["1111"] + q_["8888"] + q_["9999"])
    other = "binned" if f_ == "debt_to_income_ratio" else (f"{oth:,}$^{{a}}$" if oth else "--")
    _pl.append(" & ".join([r"\fd{" + f_ + "}", fmt_int(q_["NA"]), fmt_int(q_["Exempt"]), fmt_int(q_["numeric"]), other]) + r" \\")
open(os.path.join(TAB, "profile.tex"), "w").write("\n".join(_pl) + "\n")
_ex = prof.loc[["debt_to_income_ratio", "rate_spread", "combined_loan_to_value_ratio", "property_value", "loan_term"], "Exempt"]
assert 286_500 <= _ex.min() and _ex.max() <= 289_000      # the text says "about 287,000 records per field"
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
mac("jumpFHAFortyfive", pp(_jf["44 -> 45"], 2))

# ---- checks added after the first submission review (16 Nov 2026)
def sci(x):                 # mantissa and exponent for $m\times10^{e}$
    m_, e_ = f"{x:.1e}".split("e")
    return m_, str(int(e_))


IND, EXT = os.path.join(RES, "independent"), os.path.join(RES, "external")
# (a) third implementation in DuckDB SQL with every rule hard-coded, compared with Pandas
isum = rd(os.path.join(IND, "independent_vs_pandas_summary.csv"))
assert isum["agree"].all(), isum[~isum["agree"]]
mac("indValues", fmt_int(isum["values"].sum())); mac("indQuantities", fmt_int(len(isum)))
_m, _e = sci(isum["max_rel_diff"].max()); mac("indMaxDiff", _m); mac("indMaxExp", _e)
# (b) slices of the whole file against the FFIEC Data Browser: record counts and summed loan amounts
dbc = rd(os.path.join(EXT, "data_browser_comparison.csv"), keep_default_na=False)
assert dbc["agree"].astype(str).eq("True").all()
mac("dbSlices", fmt_int(len(dbc))); mac("dbDims", fmt_int(dbc["dimension"].nunique()))
mac("dbStates", fmt_int(dbc.loc[dbc["dimension"] == "states", "value"].nunique()))
mac("dbLoanTrillion", fmt(dbc.loc[dbc["dimension"] == "loan_types", "sum_db"].astype(float).sum() / 1e12, 2))
# (c) automated tests
_tests = {f: open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", f)).read()
          for f in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")) if f.startswith("test_")}
mac("nTests", fmt_int(sum(t.count("\ndef test_") for t in _tests.values())))
mac("nTestsSlow", fmt_int(sum(t.count("@pytest.mark.slow") for t in _tests.values())))
# (d) open-end exempt sensitivity (planned in the proposal): add back records whose open-end flag is 1111
oe_ = rd(os.path.join(RES, "sensitivity", "open_end_exempt.csv")).set_index("key")
of_ = rd(os.path.join(RES, "sensitivity", "open_end_exempt_funnel.csv")).set_index("step")
_add = of_["sensitivity"] - of_["main"]
assert int(_add["closed_end"]) == int(MAC["nOpenEndExempt"].replace(",", ""))
assert int(of_.loc["principal_residence", "main"]) == int(oe_.loc["core", "main"])
mac("oeAddBusiness", fmt_int(_add["non_business"])); mac("oeAddCore", fmt_int(_add["principal_residence"]))
mac("oeAddDecisioned", fmt_int(_add["decisioned"])); mac("oeAddUsable", fmt_int(_add["usable"]))
_stat = oe_.drop(index=["core", "decisioned", "usable", "lenders_in_both", "spearman_lender_oe_main_vs_sens"])
_m, _e = sci(_stat["change"].abs().max()); mac("oeMaxAbs", _m); mac("oeMaxAbsExp", _e)
mac("oeSpearman", fmt(oe_.loc["spearman_lender_oe_main_vs_sens", "sensitivity"], 4))
_m, _e = sci(1 - oe_.loc["spearman_lender_oe_main_vs_sens", "sensitivity"]); mac("oeRhoGap", _m); mac("oeRhoGapExp", _e)
mac("oeNStats", fmt_int(len(_stat)))
# (e) why caching did not pay off: storage status of the cached table and W1 times
cdg = rd(os.path.join(PF, "perf_cache_diagnosis.csv")).set_index("size")
for v in ("full", "w1cols"):
    assert (cdg[f"cache_{v}_disk_gb"] == 0).all()
    assert (cdg[f"cache_{v}_partitions_cached"] == cdg[f"cache_{v}_partitions"]).all()
big = [x for x in cdg.index if x != "100%"]
assert (cdg.loc[big, "w1_cache_full_s"] > cdg.loc[big, "w1_nocache_s"]).all()
assert (cdg.loc[big, "w1_cache_w1cols_s"] > cdg.loc[big, "w1_nocache_s"]).all()
assert cdg.loc["100%", "w1_cache_full_s"] < cdg.loc["100%", "w1_nocache_s"]
mac("cdStorageMax", fmt(cdg["storage_max_gb"].iloc[0], 1))
mac("cdMemSixteen", fmt(cdg.loc["x16", "cache_full_mem_gb"], 1)); mac("cdColsMemSixteen", fmt(cdg.loc["x16", "cache_w1cols_mem_gb"], 1))
mac("cdParquetSixteen", fmt(cdg.loc["x16", "parquet_gb"], 1))
mac("cdNoCacheSixteen", fmt(cdg.loc["x16", "w1_nocache_s"], 1)); mac("cdFullSixteen", fmt(cdg.loc["x16", "w1_cache_full_s"], 1))
mac("cdColsSixteen", fmt(cdg.loc["x16", "w1_cache_w1cols_s"], 1))
mac("cdNoCacheFull", fmt(cdg.loc["100%", "w1_nocache_s"], 2)); mac("cdFullFull", fmt(cdg.loc["100%", "w1_cache_full_s"], 2))
mac("cdColsBuildSixteen", fmt(cdg.loc["x16", "cache_w1cols_build_s"], 0))
# (f) calibration of the out-of-fold predictions within subgroups
psg = rd(os.path.join(RES, "predictive", "pred_subgroups.csv")).set_index(["dimension", "group"])
_lmi, _unk = psg.loc[("tract", "LMI tract")], psg.loc[("tract", "tract unknown")]
mac("predLMIobs", pct(_lmi["observed"])); mac("predLMIgb", pct(_lmi["gbt_predicted"])); mac("predLMIlr", pct(_lmi["lr_predicted"]))
mac("predLMIcitlGB", pp(_lmi["gbt_citl"])); mac("predLMIcitlLR", pp(_lmi["lr_citl"]))
_oth = psg.loc[("tract", "other tract")]
mac("predOthcitlGB", pp(_oth["gbt_citl"]))
mac("predUnkN", fmt_int(_unk["n"])); mac("predUnkObs", pct(_unk["observed"])); mac("predUnkGB", pct(_unk["gbt_predicted"]))
_reg = psg[psg.index.get_level_values(0).isin(["loan_type", "purpose", "dti_band"])].drop(index=("purpose", "not applicable"))
mac("predMaxCITL", fmt(100 * _reg[["lr_citl", "gbt_citl"]].abs().max().max(), 1))
mac("predAUCminGB", fmt(_reg["gbt_auc"].min(), 2)); mac("predAUCmaxGB", fmt(_reg["gbt_auc"].max(), 2))
# (g) dollar magnitudes for the financial interpretation
mg_ = json.load(open(os.path.join(IND, "ind_magnitudes.json")))
mac("loanCOconv", fmt_int(mg_["median_loan_conv_cashout"] / 1000)); mac("loanPurchConv", fmt_int(mg_["median_loan_conv_purchase"] / 1000))
mac("dollarCOconv", fmt_int(round(cv.gap / 100 * mg_["median_loan_conv_cashout"], -1)))
mac("incDTIfortynine", fmt_int(mg_["median_income_conv_dti49"]))
mac("dtiPointMonthly", fmt_int(mg_["median_income_conv_dti49"] * 1000 * 0.01 / 12))
mac("nConvDTIfortynine", fmt_int(mg_["n_conv_dti49"]))

# verification table (Section 7): three implementations and the external source
_eqs = rd(os.path.join(EQ, "equivalence_summary.csv"))
_eqv = int(MAC["eqTotal"].replace(",", ""))
lines = [
    rf"Pandas vs.\ PySpark, one rules file & {MAC['eqTotal']} & ${MAC['eqMaxDiff']}\times10^{{{MAC['eqMaxExp']}}}$ & yes \\",
    rf"DuckDB SQL, rules hard-coded, vs.\ Pandas & {MAC['indValues']} & ${MAC['indMaxDiff']}\times10^{{{MAC['indMaxExp']}}}$ & yes \\",
    rf"FFIEC Data Browser: counts and loan-amount sums & {MAC['dbSlices']} slices & 0 & yes \\",
]
open(os.path.join(TAB, "verify.tex"), "w").write("\n".join(lines) + "\n")

# cache diagnosis table (appendix)
lines = []
for sz, d in cdg.iterrows():
    lines.append(f"{_lab(sz)} & {d.rows / 1e6:.1f} & {d.cache_full_build_s:.1f} & {d.cache_w1cols_build_s:.1f} & "
                 f"{d.cache_full_mem_gb:.2f} & {d.cache_w1cols_mem_gb:.2f} & {d.cache_full_disk_gb:.0f} & "
                 f"{int(d.cache_full_partitions_cached)}/{int(d.cache_full_partitions)} & "
                 f"{d.w1_nocache_s:.2f} & {d.w1_cache_full_s:.2f} & {d.w1_cache_w1cols_s:.2f} \\\\")
open(os.path.join(TAB, "cache.tex"), "w").write("\n".join(lines) + "\n")

# clean core table kept as a notebook output, compared row by row with the Pandas table
_cs = os.path.join(RES, "clean", "clean_core_summary.json")
if os.path.exists(_cs):
    cc_ = json.load(open(_cs))
    assert cc_["all_agree"] and cc_["rows"] == int(MAC["nCore"].replace(",", ""))
    mac("cleanRows", fmt_int(cc_["rows"])); mac("cleanCols", fmt_int(cc_["columns"]))
    mac("cleanParts", fmt_int(cc_["partitions"])); mac("cleanMB", fmt_int(cc_["size_mb"]))
    mac("cleanCmp", fmt_int(cc_["columns_compared_with_pandas"]))
    _cd = rd(os.path.join(RES, "clean", "clean_core_columns.csv"))
    _raw = int((_cd["source"].str.startswith("raw") & (_cd["column"] != "row_id")).sum())
    lines = [f"Rows & {fmt_int(cc_['rows'])} \\\\",
             f"Columns: row id, raw text, derived & {cc_['columns']} (1, {_raw}, {cc_['columns'] - _raw - 1}) \\\\",
             f"State folders / files & {cc_['partitions']} / {cc_['files']} \\\\",
             f"Size on disk (MB) & {fmt(cc_['size_mb'], 0)} \\\\",
             f"Columns compared with Pandas, row by row & {cc_['columns_compared_with_pandas']} \\\\",
             f"Cells that differ & 0 \\\\",
             f"Rules version & {cc_['rules_version']} \\\\"]
    open(os.path.join(TAB, "clean.tex"), "w").write("\n".join(lines) + "\n")

# open-end sensitivity table (appendix)
OE_LAB = [("decisioned", "Decisioned applications", "int"), ("usable", "Usable applications", "int"),
          ("kitagawa_loan_type_2v1_gap", "FHA$-$conventional denial gap", "pp"),
          ("kitagawa_loan_type_2v1_composition", "\\quad composition part", "pp"),
          ("kitagawa_loan_purpose_32v1_rate", "Cash-out$-$purchase gap, rate part", "pp"),
          ("jump_1_49_to_5060_excess", "Conventional DTI jump 49$\\to$50--60", "pp"),
          ("pricing_conventional_32v1_gap", "Cash-out spread premium (points)", "f2"),
          ("tau_msa", "$\\tau$, MSAs", "f3"), ("tau_lender", "$\\tau$, lenders", "f3"),
          ("twoway_signal_sd_ratio", "Lender/MSA signal SD ratio", "f2")]
def _f(v, k):
    return {"int": fmt_int(v), "pp": pp(v, 2), "f2": fmt(v, 2), "f3": fmt(v, 3)}[k]
lines = [f"{lab} & {_f(oe_.loc[k, 'main'], kind)} & {_f(oe_.loc[k, 'sensitivity'], kind)} \\\\" for k, lab, kind in OE_LAB]
lines.append(f"Spearman $\\rho$ of lender $O/E$ & & {MAC['oeSpearman']} \\\\")
open(os.path.join(TAB, "open_end.tex"), "w").write("\n".join(lines) + "\n")

# ---- state map (Figure 4a): extremes of the risk-standardized ratio
ST_NAME = {"HI": "Hawaii", "FL": "Florida", "LA": "Louisiana", "ID": "Idaho", "UT": "Utah", "MN": "Minnesota",
           "WV": "West Virginia", "DC": "the District of Columbia", "MS": "Mississippi", "SD": "South Dakota"}
_so = rd(os.path.join(P, "sq3_state_oe.csv"), keep_default_na=False).sort_values("oe")
mac("stMinName", ST_NAME[_so.state_code.iloc[0]]); mac("stMinOE", fmt(_so.oe.iloc[0], 2))
mac("stMaxName", ST_NAME[_so.state_code.iloc[-1]]); mac("stMaxOE", fmt(_so.oe.iloc[-1], 2))
mac("stAboveOne", fmt_int((_so.oe > 1).sum()))

# ---- distributions of the financial variables, computed in Pandas and PySpark (results/descriptive)
DS = os.path.join(RES, "descriptive")
dsum = rd(os.path.join(DS, "summary_stats.csv")).set_index(["variable", "group"])
dband = rd(os.path.join(DS, "loan_by_income_band.csv"))
drho = rd(os.path.join(DS, "loan_income_correlation.csv")).set_index("group")
deq = rd(os.path.join(DS, "pandas_vs_spark.csv"))
dmeta = json.load(open(os.path.join(DS, "meta.json")))
assert deq["agree"].all()
# the loan-amount medians must equal those of Table 2 (same sample of originations)
_t2 = rd(os.path.join(P, "sq1_loan_amount_originated_by_type.csv")).set_index("type")
for _t in ["conventional", "FHA", "VA", "USDA"]:
    assert abs(dsum.loc[("loan_amount", _t), "p50"] * 1000 - _t2.loc[_t, "median"]) < 1e-6
    assert int(dsum.loc[("loan_amount", _t), "n"]) == int(_t2.loc[_t, "n"])
mac("dEqValues", fmt_int(deq["values"].sum())); _m, _e = sci(max(deq["max_rel_diff"].max(), 1e-17))
mac("dEqMax", _m); mac("dEqMaxExp", _e)
def _d(v, g, c, d=0, scale=1):
    return fmt(dsum.loc[(v, g), c] * scale, d)
mac("dLoanMed", _d("loan_amount", "all", "p50")); mac("dLoanMean", _d("loan_amount", "all", "mean"))
mac("dLoanPten", _d("loan_amount", "all", "p10")); mac("dLoanPninety", _d("loan_amount", "all", "p90"))
mac("dIncMed", _d("income", "all", "p50")); mac("dIncMean", _d("income", "all", "mean"))
mac("dIncPten", _d("income", "all", "p10")); mac("dIncPninety", _d("income", "all", "p90"))
mac("dIrMed", _d("interest_rate", "all", "p50", 2)); mac("dIrPtwentyfive", _d("interest_rate", "all", "p25", 2))
mac("dIrPseventyfive", _d("interest_rate", "all", "p75", 2))
for _t, _k in [("conventional", "Conv"), ("FHA", "FHA"), ("VA", "VA")]:
    mac(f"dIrMed{_k}", _d("interest_rate", _t, "p50", 3)); mac(f"dPtiMed{_k}", _d("pti", _t, "p50", 1))
    mac(f"dCostMed{_k}", _d("cost_share", _t, "p50", 1))
mac("dPtiMed", _d("pti", "all", "p50", 1)); mac("dCostMed", _d("cost_share", "all", "p50", 1))
mac("dPtiN", fmt_int(dsum.loc[("pti", "all"), "n"])); mac("dIrN", fmt_int(dsum.loc[("interest_rate", "all"), "n"]))
mac("dIrOut", fmt_int(dmeta["interest_rate_orig_numeric_out_of_range"]))
mac("dIrNotNum", fmt_int(dmeta["interest_rate_orig_not_numeric"]))
_ba = dband[dband["group"] == "all"].set_index("band_idx")
mac("dBandLoanLow", fmt_int(_ba.loc[0, "median_loan"])); mac("dBandLoanHigh", fmt_int(_ba.loc[_ba.index.max(), "median_loan"]))
mac("dBandLtiLow", fmt(_ba.loc[0, "median_lti"], 1)); mac("dBandLtiHigh", fmt(_ba.loc[_ba.index.max(), "median_lti"], 1))
mac("dRho", fmt(drho.loc["all", "spearman"], 2))
_rt = drho.loc[["conventional", "FHA", "VA"], "spearman"]
mac("dRhoMin", fmt(_rt.min(), 2)); mac("dRhoMax", fmt(_rt.max(), 2))

D_LAB = [("loan_amount", "Loan amount (\\$000), originated", 0), ("income", "Income (\\$000), decisioned", 0),
         ("interest_rate", "Interest rate (\\%), originated", 3), ("rate_spread", "Rate spread (points), pricing", 2),
         ("cltv", "CLTV (\\%), decisioned", 1), ("lti", "Loan-to-income, decisioned", 2),
         ("pti", "Payment-to-income (\\%), pricing", 1), ("cost_share", "Loan costs (\\% of loan), pricing", 2)]
lines = []
for v, lab, d in D_LAB:
    a = dsum.loc[(v, "all")]
    fm = lambda x: fmt(x, d).replace("-", "$-$")
    cells = [lab, fmt_int(a["n"])] + [fm(a[c]) for c in ["mean", "sd", "p10", "p25", "p50", "p75", "p90"]]
    cells += [fm(dsum.loc[(v, t), "p50"]) for t in ["conventional", "FHA", "VA", "USDA"]]
    lines.append(" & ".join(cells) + r" \\")
open(os.path.join(TAB, "dist_summary.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- write macros
with open(os.path.join(OUT, "numbers.tex"), "w") as fh_:
    fh_.write("% generated by build_report_assets.py from the Kaggle run outputs; do not edit by hand\n")
    for k, v in MAC.items():
        fh_.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(MAC, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, ensure_ascii=False)

# ================================================================ FIGURES
# F1 products and distributions: (a) denial by purpose x type, (b) loan amount box plots,
#    (c) interest-rate histogram, (d) loan amount by income band (b-d computed in Pandas and PySpark)
dhist = rd(os.path.join(DS, "histograms.csv"))
TYP3 = [("conventional", "Conv.", C_CONV), ("FHA", "FHA", C_FHA), ("VA", "VA", C_VA), ("USDA", "USDA", "#7f6000")]
fig, (ax, b0, b1, b2) = plt.subplots(1, 4, figsize=(7.1, 2.6), gridspec_kw={"width_ratios": [1.85, 0.95, 1.1, 1.1]})
purps = [("purchase", "Purchase"), ("refinance", "Refinance"), ("cash_out_refinance", "Cash-out"),
         ("home_improvement", "Home impr."), ("other", "Other")]
types = [("conventional", "Conv.", C_CONV), ("FHA", "FHA", C_FHA), ("VA", "VA", C_VA)]
w = 0.27
for i, (t, lab_, col) in enumerate(types):
    vals = [txp.loc[(t, p_), "denial_rate"] * 100 for p_, _ in purps]
    xs = np.arange(len(purps)) + (i - 1) * w
    ax.bar(xs, vals, w, color=col, label=lab_)
    for xx, v in zip(xs, vals):
        ax.text(xx, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=6.2)
ax.set_xticks(np.arange(len(purps)))
ax.set_xticklabels([p[1] for p in purps], fontsize=7.2, rotation=25, ha="right", rotation_mode="anchor")
ax.set_ylabel("Denial rate (%)", fontsize=8)
ax.set_ylim(0, 64)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7.2, handlelength=1.2, columnspacing=1.0)
ax.set_title("(a) Denial rate by purpose", fontsize=8.5)
stats = []
for t, lab_, col in TYP3:
    q = dsum.loc[("loan_amount", t)]
    stats.append({"label": lab_, "med": q.p50, "q1": q.p25, "q3": q.p75, "whislo": q.p05, "whishi": q.p95, "mean": q["mean"]})
bp = b0.bxp(stats, showfliers=False, showmeans=True, widths=0.55, patch_artist=True,
            meanprops={"marker": "D", "markersize": 2.5, "markerfacecolor": "white", "markeredgecolor": "black"},
            medianprops={"color": "black", "lw": 1})
for patch, (_, _, col) in zip(bp["boxes"], TYP3):
    patch.set_facecolor(col); patch.set_alpha(0.55)
b0.set_ylabel("Loan amount (\\$000)", fontsize=8)
b0.set_title("(b) Loan amount", fontsize=8.5)
b0.tick_params(axis="x", labelsize=7.2, rotation=35)
for t, lab_, col in TYP3[:3]:
    h = dhist[(dhist.variable == "interest_rate") & (dhist.group == t) & (dhist.bin >= 0) & (dhist.bin < 64)]
    h = h.assign(b2=h.bin // 2).groupby("b2")["n"].sum()          # bins of 1/4 point
    x = 2.0 + 0.25 * h.index.to_numpy()
    b1.step(x, 100 * h.to_numpy() / dsum.loc[("interest_rate", t), "n"], where="post", color=col, lw=0.9, label=lab_)
b1.set_xlim(4.0, 8.75)
b1.set_xlabel("Interest rate (%)", fontsize=8)
b1.set_ylabel("Share of loans (%)", fontsize=8)
b1.set_title("(c) Interest rate", fontsize=8.5)
b1.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.2)
for t, lab_, col in TYP3[:3]:
    q = dband[dband.group == t].sort_values("band_idx")
    b2.plot(q.band_idx, q.median_loan, marker="o", ms=2.2, lw=1, color=col, label=f"{lab_} {drho.loc[t, 'spearman']:.2f}")
labs = dband.drop_duplicates("band_idx").sort_values("band_idx").band.tolist()
b2.set_xticks(np.arange(len(labs))); b2.set_xticklabels(labs, rotation=45, fontsize=6.8)
b2.set_xlabel("Income (\\$000)", fontsize=8, labelpad=1)
b2.set_ylabel("Median loan (\\$000)", fontsize=8)
b2.set_title("(d) Loan by income", fontsize=8.5)
b2.legend(frameon=False, fontsize=6.6, loc="upper left", handlelength=1.0, borderaxespad=0.2,
          title="Spearman $\\rho$", title_fontsize=6.6, alignment="left")
for a_ in (ax, b0, b1, b2):
    a_.tick_params(axis="y", labelsize=7.2)
b1.tick_params(axis="x", labelsize=7.2)
fig.tight_layout(w_pad=0.6)
fig.savefig(os.path.join(FIG, "fig_products.pdf"))
plt.close(fig)

# F2 DTI: (a) all 19 levels; (b) 36..49, 50-60 by type with counts
jtab = rd(os.path.join(P, "sq2_dti_jump_table.csv"))
jtab["scope"] = jtab["scope"].astype(str)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.15), gridspec_kw={"width_ratios": [1.05, 1]})
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
TILE = {"AK": (0, 0), "ME": (0, 11), "VT": (1, 10), "NH": (1, 11),
        "WA": (2, 1), "ID": (2, 2), "MT": (2, 3), "ND": (2, 4), "MN": (2, 5), "IL": (2, 6), "WI": (2, 7), "MI": (2, 8),
        "NY": (2, 9), "RI": (2, 10), "MA": (2, 11), "OR": (3, 1), "NV": (3, 2), "WY": (3, 3), "SD": (3, 4), "IA": (3, 5),
        "IN": (3, 6), "OH": (3, 7), "PA": (3, 8), "NJ": (3, 9), "CT": (3, 10), "CA": (4, 1), "UT": (4, 2), "CO": (4, 3),
        "NE": (4, 4), "MO": (4, 5), "KY": (4, 6), "WV": (4, 7), "VA": (4, 8), "MD": (4, 9), "DE": (4, 10), "AZ": (5, 2),
        "NM": (5, 3), "KS": (5, 4), "AR": (5, 5), "TN": (5, 6), "NC": (5, 7), "SC": (5, 8), "DC": (5, 9), "OK": (6, 4),
        "LA": (6, 5), "MS": (6, 6), "AL": (6, 7), "GA": (6, 8), "HI": (7, 0), "TX": (7, 4), "FL": (7, 9)}
so = rd(os.path.join(P, "sq3_state_oe.csv"), keep_default_na=False).set_index("state_code")
assert set(so.index) == set(TILE), set(so.index) ^ set(TILE)
fig = plt.figure(figsize=(7.0, 2.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.26, left=0.01, right=0.99, top=0.9, bottom=0.2)
am = fig.add_subplot(gs[0])
axes = [fig.add_subplot(gs[1]), None]
axes[1] = fig.add_subplot(gs[2], sharey=axes[0])
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=0.75, vcenter=1.0, vmax=1.25)
cmap = plt.get_cmap("RdBu_r")
for st, (r_, c_) in TILE.items():
    v = so.loc[st, "oe"]
    am.add_patch(plt.Rectangle((c_, -r_), 0.92, 0.92, color=cmap(norm(v)), lw=0))
    am.text(c_ + 0.46, -r_ + 0.46, st, ha="center", va="center", fontsize=5.2,
            color="white" if abs(v - 1) > 0.15 else "black")
am.set_xlim(-0.2, 12.1); am.set_ylim(-7.2, 1.1); am.set_aspect("equal"); am.axis("off")
am.set_title("(a) States: risk-standardized $O/E$", fontsize=8)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cax = am.inset_axes([0.08, -0.07, 0.84, 0.045])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=6); cb.set_ticks([0.8, 0.9, 1.0, 1.1, 1.2])
for ax, d_, title, col in [(axes[0], lo, f"(b) Lenders (n = {len(lo)})", C_CONV), (axes[1], ms, f"(c) MSAs (n = {len(ms)})", C_FHA)]:
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
axes[1].text(0.98, 0.95, "dashed: 95% limits\ndotted: 99.8% limits", transform=axes[1].transAxes, ha="right", va="top", fontsize=6.5)
plt.setp(axes[1].get_yticklabels(), visible=False)
for ax in axes:
    ax.title.set_fontsize(8); ax.xaxis.label.set_fontsize(7.5); ax.tick_params(labelsize=7)
axes[0].yaxis.label.set_fontsize(7.5)
fig.savefig(os.path.join(FIG, "fig_funnel.pdf"))
plt.close(fig)
mac_out_lender_above3 = int((lo.oe > 3.2).sum())

# F5 performance: (a) in memory, (b) from disk, (c) peak memory
fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(7.0, 2.5))
xr = perf.rows / 1e6
for ax, pcol, scol, ttl in [(a0, "pd_{}", "sp_{}_cache", "(a) In memory"), (a1, "pd_{}_cold", "sp_{}_nocache", "(b) Read from Parquet")]:
    for wl, mk in [("W1", "o"), ("W3", "s")]:
        ax.plot(xr, perf[pcol.format(wl)], color=C_PD, marker=mk, ms=3, lw=1, label=f"Pandas {wl}")
        ax.plot(xr, perf[scol.format(wl)], color=C_SP, marker=mk, ms=3, lw=1, ls="--", label=f"Spark {wl}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.025, 80)
    ax.set_xlabel("Rows (millions, log scale)"); ax.set_title(ttl)
a0.set_ylabel("Median time (s, log scale)")
a0.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left", columnspacing=0.8, handlelength=1.6)
pm_ok = perf[perf.pd_status == "ok"]
a2.plot(pm_ok.rows / 1e6, pm_ok.pd_mem, color=C_PD, marker="o", ms=3, lw=1, label="Pandas")
a2.plot([pm_ok.rows.iloc[-1] / 1e6, perf.loc[SIZES[-1], "rows"] / 1e6], [pm_ok.pd_mem.iloc[-1], perf.loc[SIZES[-1], "pd_mem"]],
        color=C_PD, lw=1, ls=":")
a2.scatter([perf.loc[SIZES[-1], "rows"] / 1e6], [perf.loc[SIZES[-1], "pd_mem"]], marker="x", color=C_PD, s=30, zorder=5)
a2.plot(xr, perf.sp_mem, color=C_SP, marker="o", ms=3, lw=1, ls="--", label="Spark (incl. JVM)")
a2.axhline(meta["env"]["memory_budget_gb"], color="black", lw=0.7)
a2.text(0.6, meta["env"]["memory_budget_gb"] + 0.8, f"budget {meta['env']['memory_budget_gb']:.0f} GB", fontsize=7)
a2.set_xscale("log")
a2.set_xlabel("Rows (millions, log scale)")
a2.set_ylabel("Peak memory (GB)")
a2.set_ylim(0, 33)
a2.set_title("(c) Peak memory per process")
a2.legend(frameon=False, fontsize=6.5, loc="center left")
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(FIG, "fig_perf.pdf"))
plt.close(fig)

# F8 Kitagawa decomposition (main text): (a) composition + rate = gap; (b) rate part at each group's own mix
KG = [("FHA vs conventional", f), ("Cash-out vs purchase", c), ("Refinance vs purchase", r), ("LMI vs other tracts", la)]
fig, axes = plt.subplots(2, 1, figsize=(3.3, 2.55), sharex=True, gridspec_kw={"hspace": 0.5})
ys = np.arange(len(KG))[::-1]
ax = axes[0]
for y, (lab_, k) in zip(ys, KG):
    ax.barh(y, 100 * k.composition, color="#a6a6a6", height=0.55, label="Composition" if y == ys[0] else None)
    ax.barh(y, 100 * k.rate, left=100 * k.composition, color=C_CONV, height=0.55, label="Rate (same risk)" if y == ys[0] else None)
    ax.plot([100 * k.gap] * 2, [y - 0.36, y + 0.36], color="black", lw=1.1)
    ax.text(max(100 * k.gap, 100 * k.composition) + 0.8, y, f"{100 * k.gap:.1f}", va="center", fontsize=6.5)
ax.set_yticks(ys); ax.set_yticklabels([k_[0] for k_ in KG], fontsize=7)
ax.axvline(0, color="#555555", lw=0.6)
ax.set_title("(a) Gap = composition + rate (averaged weights)", fontsize=7.5, loc="left")
ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2)
ax = axes[1]
for y, (lab_, k) in zip(ys, KG):
    lo_, hi_ = sorted([100 * k.rate_gap_at_a_mix, 100 * k.rate_gap_at_b_mix])
    ax.plot([lo_, hi_], [y, y], color="#999999", lw=1.4, zorder=1)
    ax.scatter(100 * k.rate_gap_at_a_mix, y, marker="^", s=18, color=C_FHA, zorder=2, label="at A's risk mix" if y == ys[0] else None)
    ax.scatter(100 * k.rate_gap_at_b_mix, y, marker="o", s=14, color=C_CONV, zorder=2, label="at B's risk mix" if y == ys[0] else None)
ax.set_yticks(ys); ax.set_yticklabels([k_[0] for k_ in KG], fontsize=7)
ax.axvline(0, color="#555555", lw=0.6)
ax.set_xlabel("Denial gap, A minus B (percentage points)", fontsize=7.5)
ax.set_title("(b) Rate part depends on whose risk mix is used", fontsize=7.5, loc="left")
ax.legend(frameon=False, fontsize=6.5, loc="lower left", handletextpad=0.3)
for ax in axes:
    ax.tick_params(axis="x", labelsize=7)
fig.savefig(os.path.join(FIG, "fig_kitagawa.pdf"))
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
