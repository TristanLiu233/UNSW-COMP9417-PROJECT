# COMP9417 26T3 Project Kit

Everything you need for the project, apart from the specification on Moodle.

| Item | What it is |
| --- | --- |
| `data/nem_weather_halfhourly_2010_2026.csv` | The fixed dataset (see `data/README.md`) |
| `starter_notebook.ipynb` | Loads the data, builds a day-ahead table, computes both baselines |
| `template/model.py` | A blank `Forecaster` to start your model package |
| `example_package/` | A complete working package (a 7-day moving average): `model.py`, `train.py`, `artifacts/` |
| `check_submission.py` | Runs a package exactly as it will be run in marking |
| `harness/` | Code used by `check_submission.py` and in marking |
| `requirements.txt` | The exact library versions used in marking |

All commands below are run in a terminal (macOS/Linux) or in Command Prompt or PowerShell
(Windows), from inside the `comp9417_project_kit` folder.

## 1. Create a new environment (required)

Your models are saved with one set of library versions and loaded in marking with another
unless you match them exactly. Do **not** install into the Python you already use for other
courses: create a new environment with **Python 3.11** for this project.

**Option A: venv** (needs Python 3.11 installed, e.g. from https://www.python.org/downloads/)

macOS / Linux:
```
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows (Command Prompt):
```
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

Windows (PowerShell):
```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt.

**Option B: conda** (any operating system)
```
conda create -n comp9417 python=3.11
conda activate comp9417
```

Then, in the activated environment (on every system):
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**PyTorch (only if you use it).** Install version 2.15.0 in the same environment:
```
python -m pip install torch==2.15.0
```
On Windows or Linux without an NVIDIA GPU, the smaller CPU build is enough:
`python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.
PyTorch 2.15 is not available for Intel-based Macs; on those machines use a CSE lab
computer, Google Colab or another computer for PyTorch models.

Check the environment (it should print the versions in `requirements.txt`):
```
python -c "import sklearn, pandas, numpy; print(sklearn.__version__, pandas.__version__, numpy.__version__)"
```

Every time you open a new terminal, activate the environment again before working.

## 2. Days and timestamps

Each timestamp marks the **end** of a half-hour interval, in market time (AEST, UTC+10, no
daylight saving). Day D consists of the 48 intervals stamped D 00:30 to D 23:30 plus
(D+1) 00:00. When predicting day D+1, `history` ends with the interval stamped (D+1) 00:00.

## 3. Explore the data

```
python -m notebook starter_notebook.ipynb
```

## 4. Build your package

```
my_package/
├── model.py     defines class Forecaster
├── train.py     rebuilds artifacts/ from the released CSV only
└── artifacts/   your fitted models (at most 100 MB in total)
```

Start from `template/model.py`, or copy `example_package/`. The example is a 7-day moving
average: it shows the interface and how `artifacts/` is written and read, but it learns
nothing and is not a good forecaster. Rebuild its `artifacts/` with:
```
python example_package/train.py data/nem_weather_halfhourly_2010_2026.csv
```

Tips:
* Share one feature function between `train.py` and `model.py` (for example, import it in
  `train.py` from `model.py`), so that training and prediction compute features in the same way.
* `__init__` must only load files. `predict` receives the full history; use only what you
  need (the example uses the last 7 days) to stay well inside the 20-second limit.
* Build file paths with `os.path.join(artifact_dir, "...")`, never with `\` or `/` typed by
  hand, so that your code runs on any operating system.
* PyTorch: save `model.state_dict()` with `torch.save`, rebuild the network in `__init__` and
  load the weights with `torch.load(path, weights_only=True, map_location="cpu")`.

## 5. Check it before you submit

```
python check_submission.py my_package
python check_submission.py submission.zip
```

This runs your package one day at a time for the last 14 days of the dataset, with the same
wrapper, time limits and output checks used in marking, and compares it with both baselines.
These 14 days are part of the released data, which your final models will normally be trained
on, so the scores are in-sample: they show that the package runs and produces sensible output,
not how it will perform in the live week.
If any day fails, the reason is printed. **In marking, one failed day costs 2 marks, and a package
that fails on more than one day of the live week receives no live-evaluation marks.**

In marking, packages run on Linux with no network access, where only `/tmp` is writable,
with 4 GB of memory and 2 CPUs. Keep your package within these limits.

## 6. Make the zip

The zip must contain `model.py`, `train.py` and `artifacts/`, either at the top level or
inside a single folder.

* macOS: right-click `my_package` in Finder and choose **Compress**.
* Windows: right-click `my_package` in File Explorer and choose
  **Send to > Compressed (zipped) folder** (on Windows 11: **Compress to ZIP file**).
* Linux or macOS terminal: `zip -r submission.zip my_package`

Rename the result to `submission.zip`, then run `python check_submission.py submission.zip`
on the zip itself before uploading it.

## Rules in brief

* Train only on the released CSV. No external data.
* No network access and no writing outside `/tmp` during prediction.
* The package should run on every day of the live week (one failed day costs 2 marks; more than one
  means no live-evaluation marks).

Weather data: Open-Meteo (https://open-meteo.com), CC BY 4.0. Market data: AEMO.

## 小组模型与共享特征 v1

项目预测 NSW 次日 48 个半小时的需求和电价方向。需求回归使用 Ridge 和梯度提升回归，电价方向分类使用逻辑回归和梯度提升分类。[四个模型的字段与训练说明](ablation/消融说明.md) · [交互消融矩阵](ablation/outputs/2014-ablation-matrices/消融矩阵.html)

### 训练、回测与单日预测

在仓库根目录的课程 Python 3.11 环境中运行。将 `--data` 后的路径替换为本机课程 CSV 的位置。

```shell
python -m baseline.train --data "data/nem_weather_halfhourly_2010_2026.csv"
python -m baseline.replay --data "data/nem_weather_halfhourly_2010_2026.csv"
```

训练命令保存两版权重。验证版用截至 2013 年底的数据拟合，在 2014 年验证。冻结版用截至 2014 年底的数据重新拟合，随后固定权重，回测 2015 年 1 月至 2026 年 10 月。[模型权重与 2014 验证分数](baseline/outputs/) · [逐月、逐年及全期回测](baseline/replay_results/metrics.csv)

`build_features(df)` 为训练和回测生成批量特征。`build_features_for_day(history, weather_next)` 返回单日 48 行特征。两者以相同顺序输出 36 列，每个模型按自己的有序字段清单取列。单日调用时，传入与批量数据同一起点、截至目标日 00:00 的完整市场历史，以及随后 48 档天气。原 35 个字段与旧版逐位一致，第 36 列为 NSW 公共假期标记。后续修改接口、列序或计算公式时，通过 PR 提交并运行一致性测试。[共享模块](src/common/features.py) · [单日预测](baseline/predict.py) · [一致性测试](tests/test_common_features.py)

### NSW 公共假期消融

需求提升树增加假期列后，2014 年 MAPE 从 2.4355% 降至 2.3318%，长期冻结 MAPE 从 4.4230% 降至 4.3452%。Ridge 的长期冻结 MAPE 从 5.0100% 升至 5.0398%。仓库中的基线权重使用原配方，假期实验结果单独保存。[实验结果、分组比较与复现命令](baseline/holiday_results/README.md)
