"""Kiểm tra chính script so khớp: nó phải báo khớp khi hai bên giống nhau và bắt được sai lệch khi có."""
import os
import shutil

import pandas as pd
import pytest

from conftest import RES, SRC, run

FILES = ["eq_1_filter_counts", "eq_2_special_codes", "eq_2b_field_profile", "eq_3_action_taken",
         "eq_4_amounts_by_loan_type", "eq_4b_rate_spread_counts", "eq_5_denial_by_state", "eq_6_top50_lenders"]


@pytest.fixture
def dirs(tmp_path):
    for side in ("pandas", "spark"):
        d = tmp_path / side
        d.mkdir()
        for f in FILES:
            shutil.copy(os.path.join(RES, side, f + ".csv"), d)
        shutil.copy(os.path.join(RES, side, "key_numbers.json"), d)
    return tmp_path


def check(dirs):
    code, out = run([os.path.join(SRC, "check_equivalence.py"), str(dirs / "pandas"), str(dirs / "spark"), str(dirs / "out")])
    assert code == 0, out
    return pd.read_csv(dirs / "out" / "equivalence_summary.csv")


def test_identical_outputs_agree(dirs):
    assert check(dirs)["Agree"].all()


def test_one_count_off_by_one_is_caught(dirs):
    p = dirs / "spark" / "eq_5_denial_by_state.csv"
    t = pd.read_csv(p, dtype=str, keep_default_na=False)
    t.loc[0, "n_denied"] = str(int(t.loc[0, "n_denied"]) + 1)
    t.to_csv(p, index=False)
    s = check(dirs)
    assert not s.loc[s["Quantity"].str.startswith("(5)"), "Agree"].iloc[0]


def test_mean_differing_beyond_tolerance_is_caught(dirs):
    p = dirs / "spark" / "eq_4_amounts_by_loan_type.csv"
    t = pd.read_csv(p)
    t.loc[0, "mean"] *= 1 + 1e-7
    t.to_csv(p, index=False)
    s = check(dirs)
    assert not s.loc[s["Quantity"].str.startswith("(4) Mean"), "Agree"].iloc[0]


def test_floating_point_noise_is_tolerated(dirs):
    p = dirs / "spark" / "eq_4_amounts_by_loan_type.csv"
    t = pd.read_csv(p)
    t.loc[0, "mean"] *= 1 + 1e-13
    t.to_csv(p, index=False, float_format="%.17g")
    assert check(dirs)["Agree"].all()
