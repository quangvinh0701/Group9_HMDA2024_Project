"""Tính lại các con số của báo cáo từ các bảng trung gian đã lưu, độc lập với code của pipeline."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import RES

P = lambda n: pd.read_csv(os.path.join(RES, "pandas", f"{n}.csv"))
K = json.load(open(os.path.join(RES, "pandas", "key_numbers.json")))


def test_kitagawa_parts_add_up():
    k = P("sq2_kitagawa_denial")
    assert np.allclose(k["composition"] + k["rate"], k["gap"], atol=1e-12)
    assert np.allclose(k["rate"], (k["rate_gap_at_a_mix"] + k["rate_gap_at_b_mix"]) / 2, atol=1e-12)
    assert (k["cells_a_higher"] + k["cells_a_lower"] <= k["n_common_cells"]).all()


def test_tau_matches_oe_table():
    for table, key in [("sq4_lender_oe", "tau_lender"), ("sq3_state_oe", "tau_state")]:
        t = P(table)
        tau2 = t["oe"].var(ddof=1) - t["sampling_var"].mean()
        assert np.sqrt(tau2) == pytest.approx(K[key], rel=1e-12)


def test_oe_is_observed_over_expected():
    t = P("sq4_lender_oe")
    assert np.allclose(t["O"] / t["E"], t["oe"], rtol=1e-12)


def test_dti_jump_recomputed_from_rates():
    d = P("sq2_denial_by_dti_19")
    r = d.set_index(d["dti_ordinal"].astype(int))["denial_rate"]
    excess = (r[18] - r[17]) - (r[17] - r[4]) / 13
    assert excess == pytest.approx(K["jump_all_49_to_5060_excess"], rel=1e-12)


def test_sample_funnel_is_monotone():
    f = P("eq_1_filter_counts").set_index("step")["rows"]
    order = ["all_records", "applications", "first_lien", "closed_end", "non_business", "non_reverse",
             "site_built_1_4_unit", "principal_residence"]
    assert all(f[a] >= f[b] for a, b in zip(order, order[1:]))
    assert f["principal_residence"] >= f["decisioned"] >= f["usable"]


def test_equivalence_summary_all_agree():
    s = pd.read_csv(os.path.join(RES, "equivalence", "equivalence_summary.csv"))
    assert s["Agree"].all()
    assert s.loc[~s["Quantity"].str.contains("percentile_approx"), "Values compared"].sum() == 1684


def test_independent_implementation_agrees():
    s = pd.read_csv(os.path.join(RES, "independent", "independent_vs_pandas_summary.csv"))
    assert s["agree"].all() and s["values"].sum() > 2000


def test_data_browser_slices_agree():
    c = pd.read_csv(os.path.join(RES, "external", "data_browser_comparison.csv"))
    assert c["agree"].all()
    a = json.load(open(os.path.join(RES, "external", "data_browser_action_taken_2024.json")))
    db = {x["actions_taken"]: x["count"] for x in a["aggregations"]}
    ours = P("eq_3_action_taken").set_index("action_taken")["n"]
    assert all(int(ours[int(k)]) == v for k, v in db.items())


def test_perf_engines_agree_at_100pct():
    m = json.load(open(os.path.join(RES, "perf", "perf_meta.json")))
    assert m["equality_100pct"]["all_equal"]


def test_clean_core_table_matches_pandas():
    s = json.load(open(os.path.join(RES, "clean", "clean_core_summary.json")))
    m = pd.read_csv(os.path.join(RES, "clean", "clean_core_manifest.csv"), keep_default_na=False)
    c = pd.read_csv(os.path.join(RES, "clean", "clean_core_pandas_vs_spark.csv"))
    core = P("eq_1_filter_counts").set_index("step")["rows"]["principal_residence"]
    assert s["all_agree"] and s["rows"] == core == m["rows"].sum() and s["partitions"] == len(m)
    both = c[c["in_pandas"] & c["in_spark"]]
    assert len(both) == s["columns_compared_with_pandas"] and both["agree"].all() and (both["n_diff"] == 0).all()


def test_open_end_sensitivity_changes_little():
    t = pd.read_csv(os.path.join(RES, "sensitivity", "open_end_exempt.csv")).set_index("key")
    f = pd.read_csv(os.path.join(RES, "sensitivity", "open_end_exempt_funnel.csv")).set_index("step")
    core = P("eq_1_filter_counts").set_index("step")["rows"]
    assert f.loc["closed_end", "main"] == core["closed_end"] and f.loc["usable", "main"] == core["usable"]
    stats = t.drop(index=["core", "decisioned", "usable", "lenders_in_both", "spearman_lender_oe_main_vs_sens"])
    assert stats["change"].abs().max() < 1e-5
    assert t.loc["spearman_lender_oe_main_vs_sens", "sensitivity"] > 0.9999


def test_cache_fitted_in_memory():
    d = pd.read_csv(os.path.join(RES, "perf", "perf_cache_diagnosis.csv")).set_index("size")
    assert (d[["cache_full_disk_gb", "cache_w1cols_disk_gb"]] == 0).all().all()
    assert (d["cache_full_mem_gb"] < d["storage_max_gb"]).all()
    assert (d["cache_full_partitions_cached"] == d["cache_full_partitions"]).all()


def test_independent_lender_oe_matches_pandas():
    a = pd.read_csv(os.path.join(RES, "independent", "ind_oe_lender.csv"))
    b = P("sq4_lender_oe")[["lei", "n", "O", "E"]]
    m = b.merge(a, on="lei", suffixes=("_pd", "_db"))
    assert len(m) == len(b)
    assert (m["n_pd"] == m["n_db"]).all() and (m["O_pd"] == m["O_db"]).all()
    assert np.allclose(m["E_pd"], m["E_db"], rtol=1e-9, atol=0)
    d = pd.read_csv(os.path.join(RES, "independent", "independent_vs_pandas.csv"))
    assert d["agree"].all()
    s = pd.read_csv(os.path.join(RES, "independent", "independent_vs_pandas_summary.csv"))
    assert len(d) == s["values"].sum()


def test_descriptive_distributions_agree_and_match_table2():
    e = pd.read_csv(os.path.join(RES, "descriptive", "pandas_vs_spark.csv"))
    assert e["agree"].all() and (e["max_rel_diff"] <= 1e-9).all()
    d = pd.read_csv(os.path.join(RES, "descriptive", "summary_stats.csv")).set_index(["variable", "group"])
    t2 = P("sq1_loan_amount_originated_by_type").set_index("type")
    for t in ["conventional", "FHA", "VA", "USDA"]:
        assert d.loc[("loan_amount", t), "p50"] * 1000 == pytest.approx(t2.loc[t, "median"])
        assert d.loc[("loan_amount", t), "n"] == t2.loc[t, "n"]
    rs = P("sq2_rate_spread_by_type").set_index("type")
    for t in ["conventional", "FHA", "VA", "USDA"]:
        assert d.loc[("rate_spread", t), "mean"] == pytest.approx(rs.loc[t, "mean"], rel=1e-9)
    h = pd.read_csv(os.path.join(RES, "descriptive", "histograms.csv"))
    for v, var in [("loan_amount", "loan_amount"), ("interest_rate", "interest_rate")]:
        tot = h[(h.variable == v) & (h.group != "all")]["n"].sum()
        assert tot == d.loc[(var, "all"), "n"]
