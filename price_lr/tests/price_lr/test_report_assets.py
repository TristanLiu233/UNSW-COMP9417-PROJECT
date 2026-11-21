"""Report assets: every table number traces back to its source file. Run: pytest tests/price_lr -q"""
import glob
import json
import os

import pandas as pd

from src.price_lr.env import RESULTS
from src.price_lr.report_assets import num

REP = os.path.join(RESULTS, "report")
MAN = json.load(open(os.path.join(REP, "manifest.json")))
TEX = {os.path.basename(f)[len("tab_p5_"):-len(".tex")]: open(f).read()
       for f in glob.glob(os.path.join(REP, "tables", "tab_p5_*.tex"))}


def source_value(cell):
    df = pd.read_csv(os.path.join(RESULTS, cell["source"]), float_precision="round_trip")
    if "year" in df:
        row = df.set_index("year").loc[int(cell["key"])]
    elif {"a", "b"} <= set(df):
        a, b = cell["key"].split("-")
        row = df.set_index(["a", "b"]).loc[(a, b)]
    else:
        row = df.set_index("strategy").loc[cell["key"]]
    if "-" in cell["column"]:
        x, y = cell["column"].split("-")
        return row[x] - row[y]
    return row[cell["column"]]


def test_every_cell_matches_source_and_appears_in_table():
    cells = [c for c in MAN["cells"] if "text" in c]
    assert len(cells) == 98 and set(TEX) == {"s0_alarms", "strategies", "stage3", "strategies_stage3"}
    for c in cells:
        assert num(float(source_value(c)), c["dp"], c["sign"]) == c["text"], c
        assert c["text"] in TEX[c["table"]], c


def test_counts_match_daily_files():
    al = next(c for c in MAN["cells"] if c.get("kind") == "alarm_counts")["values"]
    mon = pd.concat([pd.read_csv(os.path.join(RESULTS, "s0_monitor_cusum.csv")),
                     pd.read_csv(os.path.join(RESULTS, "stage3", "s0_monitor_cusum_2025_2026.csv"))])
    by_year = mon[mon["alarm"]].groupby(pd.to_datetime(mon[mon["alarm"]]["day"]).dt.year).size()
    assert {int(y): n for y, n in al.items()} == {y: int(by_year.get(y, 0)) for y in range(2015, 2027)}
    for y, n in al.items():
        assert f"{y}{'$^{*}$' if y == '2026' else ''} & {n} &" in TEX["s0_alarms"]
    rts = [c["values"] for c in MAN["cells"] if c.get("kind") == "retrains"]
    assert len(rts) == 2 and rts[0] == rts[1]
    rt = rts[0]
    log = pd.read_csv(os.path.join(RESULTS, "models_log.csv"))
    eff = pd.to_datetime(log["effective_from"])
    for s, n in rt.items():
        assert n == int(((log["strategy"] == s) & (log["model_id"] != "LR_frozen") & (eff <= "2024-12-31")).sum())
    ev = pd.read_csv(os.path.join(RESULTS, "s3_events.csv"))       # M7 file already holds the 2025-2026 events
    late = pd.read_csv(os.path.join(RESULTS, "stage3", "s3_events_2025_2026.csv"))
    assert list(ev[ev["alarm_day"] >= "2025-01-01"]["alarm_day"]) == list(late["alarm_day"])
    assert MAN["s3_markers"] == {"suppressed": int((~ev["retrained"]).sum()), "retrained": int(ev["retrained"].sum())}


def test_captions_empty_and_no_prose_uploaded():
    for name, tex in TEX.items():
        n_labels = 2 if name == "strategies_stage3" else 1      # merged table keeps both old labels
        assert "\\caption{} % P5: caption" in tex and tex.count("\\label{tab:p5-") == n_labels, name
    figs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(REP, "figures", "*.pdf")))
    assert figs == ["fig_p5_monthly_gap.pdf", "fig_p5_s0_monitor.pdf", "fig_p5_s3_cusum.pdf", "fig_p5_yearly_ba.pdf"]
    assert all(w == MAN["figure_width_in"] for w, _ in MAN["figure_sizes_in"].values())


def test_facts_numbers_from_files():
    f = open(os.path.join(REP, "P5_facts.md")).read()
    veto = json.load(open(os.path.join(RESULTS, "stage3", "veto.json")))
    assert f"BA(P) = {veto['BalAcc_P']:.2f}" in f and f"最终 = {veto['final_strategy']}" in f
    assert "不上传 Overleaf" in f and "不是**未见过的留出集" in f


def test_merged_table_matches_separate_tables():
    """11-21 merge: every number in the merged table is the same text as in tab_p5_strategies / tab_p5_stage3."""
    merged = [c for c in MAN["cells"] if c.get("table") == "strategies_stage3" and "text" in c]
    old = {(c["source"], c["key"], c["column"]): c["text"] for c in MAN["cells"]
           if c.get("table") in ("strategies", "stage3") and "text" in c}
    assert len(merged) == 31
    for c in merged:
        assert old[(c["source"], c["key"], c["column"])] == c["text"], c


def test_overleaf_bundle_matches_generated_tables():
    from src.price_lr.report_assets import SUBSET
    ov = os.path.join(REP, "overleaf")
    body = open(os.path.join(ov, "sections", "p5_drift.tex")).read() + open(os.path.join(ov, "sections", "p5_adaptation.tex")).read()
    for t in SUBSET["tables"]:
        env = TEX[t[len("tab_p5_"):-len(".tex")]].split("\\begin{table}", 1)[1]
        assert "\\begin{table}" + env in body, t
    for f in SUBSET["figures"]:
        assert open(os.path.join(ov, "figures", "p5", f), "rb").read() == open(os.path.join(REP, "figures", f), "rb").read()
        assert f"figures/p5/{f}" in body
    prose = [l for l in body.splitlines() if l.strip() and not l.lstrip().startswith(("%", "\\", "&")) and "&" not in l]
    assert prose == [], prose       # no sentences: every line is a comment, a command or a table row
