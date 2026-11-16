"""Chạy pipeline trên file giả có đúng schema HMDA (gen_synth.py) và kiểm tra các cài đặt khớp nhau.

Chạy: pytest -m slow   (400.000 dòng giả; Pandas + DuckDB vài phút, PySpark cần Java)
"""
import os
import shutil

import pandas as pd
import pytest

from conftest import ROOT, SRC, run


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    inp = d / "in"
    inp.mkdir()
    code, out = run([os.path.join(SRC, "gen_synth.py"), str(inp), "400000"])
    assert code == 0, out
    # chỉ chép rules.yaml: expected.yaml chứa số của dữ liệu thật nên không áp cho dữ liệu giả
    shutil.copy(os.path.join(ROOT, "config", "rules.yaml"), inp)
    env = {"HMDA_INPUT": str(inp), "HMDA_OUT": str(d / "out"), "HMDA_WORK": str(d / "work")}
    code, out = run([os.path.join(SRC, "pandas_pipeline.py")], env)
    assert code == 0, out[-3000:]
    return d, env


@pytest.mark.slow
def test_duckdb_reimplementation_matches_pandas(synth):
    d, env = synth
    code, out = run([os.path.join(SRC, "independent_check.py"), str(d / "in" / "hmda_2024_raw.parquet"),
                     str(d / "out" / "results" / "pandas"), str(d / "ind")], env)
    assert code == 0, out[-3000:]
    s = pd.read_csv(d / "ind" / "independent_vs_pandas_summary.csv")
    assert s["agree"].all(), s


@pytest.mark.slow
def test_spark_matches_pandas(synth):
    pytest.importorskip("pyspark")
    if shutil.which("java") is None:
        pytest.skip("cần Java cho PySpark")
    d, env = synth
    code, out = run([os.path.join(SRC, "spark_pipeline.py")], env)
    assert code == 0, out[-3000:]
    res = d / "out" / "results"
    code, out = run([os.path.join(SRC, "check_equivalence.py"), str(res / "pandas"), str(res / "spark"), str(d / "eq")], env)
    assert code == 0, out[-3000:]
    assert pd.read_csv(d / "eq" / "equivalence_summary.csv")["Agree"].all()
