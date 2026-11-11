"""Verify P3 with the official runner using yesterday-price ONLY as a smoke-test placeholder."""
import argparse
from pathlib import Path
from shutil import copy2
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

MODEL = """from p3_hgbr.demand import DemandForecaster

class Forecaster:
    def __init__(self, artifact_dir):
        self.demand = DemandForecaster(artifact_dir)

    def predict(self, history, weather_next):
        output = self.demand.predict(history, weather_next)
        yesterday = history['nsw_price'].iloc[-48:]
        # Smoke-test placeholder ONLY; replace with the selected team price model.
        output['nsw_price_up'] = yesterday.gt(yesterday.mean()).astype(int).to_numpy()
        return output
"""

TRAIN = """import json
from pathlib import Path
import sys
from p3_hgbr.train_demand import train

root = Path(__file__).resolve().parent
train(sys.argv[1], json.loads((root / 'demand_config.json').read_text()), root / 'artifacts')
"""


def run(kit, data, stage2, work):
    work.mkdir(parents=True, exist_ok=False)
    package = work / "p3_smoke_only"
    (package / "p3_hgbr").mkdir(parents=True)
    (package / "artifacts").mkdir()
    for module in ("__init__.py", "demand.py", "train_demand.py"):
        copy2(Path(__file__).with_name(module), package / "p3_hgbr" / module)
    copy2(stage2 / "final_artifacts" / "demand_hgbr.joblib", package / "artifacts")
    copy2(stage2 / "p3_final_demand_config.json", package / "demand_config.json")
    (package / "model.py").write_text(MODEL)
    (package / "train.py").write_text(TRAIN)
    archive = work / "P3-SMOKE-ONLY-NOT-FOR-SUBMISSION.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for file in sorted(package.rglob("*")):
            if file.is_file():
                z.write(file, file.relative_to(package))
    print("P3 demand smoke test; price uses yesterday baseline placeholder.", flush=True)
    print("Do not submit this test ZIP. Last 14 days are in-sample for final demand training.", flush=True)
    subprocess.run([sys.executable, str(kit / "check_submission.py"), str(archive),
                    "--data", str(data), "--days", "14"], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--stage2", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    run(a.kit, a.data, a.stage2, a.work)
