"""Read-only parsers for outputs/ files and README sections."""
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
README = os.path.join(ROOT, "README.md")


def _coerce(v):
    if v in ("yes", "no"):
        return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def read_results(name):
    """Return list of (key, value) from outputs/<name>."""
    pairs = []
    with open(os.path.join(OUT_DIR, name), newline="") as f:
        for line in f.read().splitlines():
            if " = " in line:
                k, v = line.split(" = ", 1)
                pairs.append((k, _coerce(v)))
    return pairs


def results_dict(name):
    return dict(read_results(name))


def read_table(name):
    """Return {'header': [...], 'rows': [[...]]} from outputs/<name>."""
    with open(os.path.join(OUT_DIR, name), newline="") as f:
        rows = list(csv.reader(f))
    return {"header": rows[0], "rows": [[_coerce(c) for c in r] for r in rows[1:]]}


def _readme_text():
    with open(README, encoding="utf-8") as f:
        return f.read()


def readme_section(heading):
    """Text under '## <heading>' up to the next '## ' heading (empty string if absent)."""
    pattern = r"^## " + re.escape(heading) + r"[ \t]*\n(.*?)(?=^## |\Z)"
    m = re.search(pattern, _readme_text(), re.S | re.M)
    return m.group(1).strip() if m else ""
