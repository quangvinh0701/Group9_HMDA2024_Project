"""Chuyển file .py có dấu '# %%' / '# %% [markdown]' thành .ipynb (không xuất kết quả)."""
import json
import sys


def convert(src, dst):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return
        if kind == "markdown":
            text = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [t + "\n" for t in text[:-1]] + [text[-1]]})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": [t + "\n" for t in lines[:-1]] + [lines[-1]]})

    for ln in open(src, encoding="utf-8").read().splitlines():
        if ln.startswith("# %% [markdown]"):
            flush(); kind, buf = "markdown", []
        elif ln.startswith("# %%"):
            flush(); kind, buf = "code", []
        else:
            buf.append(ln)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(dst, len(cells), "cells")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
