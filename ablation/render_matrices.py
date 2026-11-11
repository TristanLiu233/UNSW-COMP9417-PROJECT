"""Build the standalone HTML view of the saved ablation results; no model fitting."""

import argparse
import csv
import json
from pathlib import Path

from .matrix_data import collect_results


HERE = Path(__file__).resolve().parent
MATRICES = HERE / "outputs" / "2014-ablation-matrices"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=MATRICES)
    parser.add_argument("--rolling", type=Path, default=MATRICES / "跨年天气对照.csv")
    args = parser.parse_args()
    data = collect_results(args.results)
    with args.rolling.open(encoding="utf-8-sig", newline="") as source:
        rolling = list(csv.DictReader(source))
    if {(int(row['year']), row['model']) for row in rolling} != {
        (year, model) for year in (2012, 2013, 2014) for model in ("ridge", "boost_demand", "logistic", "boost_price")
    } or len(rolling) != 12:
        raise ValueError("The rolling weather supplement needs all four models for each of the three years")
    data["rolling"] = [{**r, **{key: float(r[key]) for key in ("core", "core_weather", "gain_pp")},
                       "year": int(r['year'])} for r in rolling]
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = (HERE / "matrix_template.html").read_text(encoding="utf-8")
    assert template.count("__ABLATION_JSON__") == 1
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "消融矩阵.html"
    output.write_text(template.replace("__ABLATION_JSON__", payload), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
