import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KIT = os.path.join(ROOT, "comp9417_project_kit")
for p in (ROOT, os.path.join(KIT, "harness")):
    if p not in sys.path:
        sys.path.insert(0, p)
