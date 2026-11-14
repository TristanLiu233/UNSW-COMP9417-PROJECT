"""M0 acceptance: execute the kit's starter notebook with the .venv interpreter.

The kit notebook is read, not modified: one probe cell is appended to the in-memory copy
to record which interpreter / library versions the kernel actually used, and the executed
copy is written to results/price_hgbc/selfcheck/. The kernel's interpreter is recorded
by the probe, so a fallback to another Python on PATH would be detected.
Usage: python -m src.price_hgbc.m0_run_notebook
"""
import json
import os
import sys

import nbformat
from nbclient import NotebookClient

from .kit import KIT, RESULTS

PROBE = ("import sys, json, numpy, pandas, sklearn\n"
         "print('PROBE ' + json.dumps({'executable': sys.executable, 'python': sys.version.split()[0],"
         " 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__}))")


def run(out_dir=os.path.join(RESULTS, "selfcheck")):
    nb = nbformat.read(os.path.join(KIT, "starter_notebook.ipynb"), as_version=4)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    nb.cells.append(nbformat.v4.new_code_cell(PROBE))
    # The .venv kernelspec launches plain "python"; put this interpreter's bin/ first on PATH
    # so that it resolves to .venv/bin/python (the probe cell below verifies it).
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": KIT}})
    client.execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    probe = None
    for o in nb.cells[-1].outputs:
        for line in o.get("text", "").splitlines():
            if line.startswith("PROBE "):
                probe = json.loads(line[6:])
    os.makedirs(out_dir, exist_ok=True)
    nbformat.write(nb, os.path.join(out_dir, "m0_starter_notebook_executed.ipynb"))
    summary = {"n_code_cells_kit": n_code, "n_error_outputs": len(errors), "probe": probe}
    with open(os.path.join(out_dir, "m0_starter_notebook_probe.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
