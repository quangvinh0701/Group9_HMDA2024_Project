import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
RES = os.path.join(ROOT, "results")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: chạy pipeline trên dữ liệu giả (vài phút); chọn bằng -m slow")


@pytest.fixture
def root():
    return ROOT


def run(args, env=None, cwd=None):
    """Chạy một script Python, trả về (mã thoát, stdout+stderr)."""
    import subprocess
    e = {**os.environ, **(env or {})}
    p = subprocess.run([sys.executable] + args, capture_output=True, text=True, env=e, cwd=cwd or ROOT)
    return p.returncode, p.stdout + p.stderr
