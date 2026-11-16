"""Kiểm tra file luật: các băng liền nhau, thang DTI đủ 19 mức, bộ lọc dùng đúng tên cột của HMDA."""
import math
import os

import yaml

from conftest import ROOT, SRC

R = yaml.safe_load(open(os.path.join(ROOT, "config/rules.yaml"), encoding="utf-8"))
COLS = {c.strip() for c in open(os.path.join(SRC, "columns.txt")).read().replace(",", " ").split() if c.strip()}


def _resolve(v, cuts=None):
    return cuts[v] if isinstance(v, str) else float(v)


def check_contiguous(bands, cuts=None):
    for a, b in zip(bands, bands[1:]):
        assert _resolve(a["hi"], cuts) == _resolve(b["lo"], cuts), (a, b)
    assert len({b["label"] for b in bands}) == len(bands)


def test_core_filters_use_real_columns():
    for f in R["samples"]["core_filters"]:
        assert f["column"] in COLS, f
        assert ("in" in f) != ("equals" in f), f


def test_dti_ordinal_has_19_levels_in_order():
    m = R["variables"]["debt_to_income_ratio"]["ordinal_map"]
    assert sorted(m.values()) == list(range(1, 20))
    assert [m[str(v)] for v in range(36, 50)] == list(range(4, 18))


def test_dti_bands_cover_the_ordinal_scale():
    bands = R["variables"]["debt_to_income_ratio"]["bands"]
    check_contiguous(bands)
    assert bands[0]["lo"] == 0 and bands[-1]["hi"] == 19 and len(bands) == 6


def test_cltv_and_lti_bands_are_contiguous():
    check_contiguous(R["variables"]["combined_loan_to_value_ratio"]["bands"])
    lti = R["variables"]["loan_to_income"]
    check_contiguous(lti["bands"], lti["cut_exact"])
    assert lti["cut_exact"]["q1"] < lti["cut_exact"]["q2"]
    assert [round(lti["cut_exact"][k], 2) for k in ("q1", "q2")] == lti["cut_reported"]


def test_risk_cells_are_6_by_5_by_3():
    V = R["variables"]
    n = len(V["debt_to_income_ratio"]["bands"]) * len(V["combined_loan_to_value_ratio"]["bands"]) * len(V["loan_to_income"]["bands"])
    assert n == 90


def test_valid_ranges_are_sane():
    rs = R["variables"]["rate_spread"]["valid_range"]
    assert rs["ge"] < 0 < rs["le"]
    cl = R["variables"]["combined_loan_to_value_ratio"]["valid_range"]
    assert cl["gt"] == 0 and 100 < cl["le"] < math.inf
