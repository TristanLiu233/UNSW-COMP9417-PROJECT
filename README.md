# COMP9417 26T3 · NSW 日前预测

`baseline` 分支保存小组的三轮消融实验。任务是预测 NSW 次日 48 个半小时的需求（MW），并判断各档电价是否高于前一市场日的 NSW 均价。四种算法用相同的 2010–2013 年训练样本、2014 年验证日期，以及昨日和上周同档基线进行比较。

## 阅读实验结果

| 入口 | 用途 |
| --- | --- |
| [原版交互消融矩阵（HTML）](ablation/outputs/2014-ablation-matrices/消融矩阵.html) | 克隆仓库后用浏览器打开这个文件，可以筛选模型、点选单元格查看对照及置信区间，并查看 340 个配置 × 53 个字段的开关矩阵。 |
| [消融说明与四个模型的训练交接](ablation/消融说明.md) | 按回归组、分类组给出四份有序输入清单、训练参数和独立复现代码。字段字典逐列说明 41 个派生特征的公式、用途和入选模型。 |
| [完整 Excel 矩阵](ablation/outputs/2014-ablation-matrices/消融实验矩阵.xlsx) · [维度热图](ablation/outputs/2014-ablation-matrices/维度消融热图.png) | Excel 支持筛选完整结果；热图展示 20 个维度 × 3 种对照方式 × 4 个模型的指标差。 |
| [2014 验证结果摘要](ablation/RESULTS.md) · [全部配置与配对结果](ablation/results/) | 340 个配置、368 个配对对照、两套区间，另保存每个模型的逐档预测。 |

| 任务 | 模型与字段数 | 2014 年主指标 |
| --- | --- | ---: |
| 需求回归 | Ridge，25 列 | MAPE 3.309% |
| 需求回归 | **梯度提升回归，29 列** | **MAPE 2.436%** |
| 价格方向分类 | 逻辑回归，22 列 | 平衡准确率 75.793% |
| 价格方向分类 | **梯度提升分类，18 列** | **平衡准确率 79.300%** |

粗体为按预设规则入选的模型。2014 年用于选型；2015–2026 年冻结回放属于下一阶段。

## 复现实验

1. 在仓库根目录按下方说明创建 Python 3.11 环境并安装 `requirements.txt`。
2. 从课程发布的套件取得 `nem_weather_halfhourly_2010_2026.csv`。将它放在本地 `data/` 目录，或在 `--data` 后填写已有文件的路径。
3. 从仓库根目录运行以下命令；各轮的特征、指标和输出路径见[消融运行说明](ablation/README.md)。

```shell
python -m ablation.run --data "data/nem_weather_halfhourly_2010_2026.csv"
```

## Official project kit

| Item | What it is |
| --- | --- |
| [`specs/project_spec.pdf`](specs/project_spec.pdf) | The project specification |
| `data/nem_weather_halfhourly_2010_2026.csv` | Local path for the fixed dataset supplied by the course |
| `starter_notebook.ipynb` | Loads the data, builds a day-ahead table, computes both baselines |
| `template/model.py` | A blank `Forecaster` to start your model package |
| `example_package/` | A complete working package (a 7-day moving average): `model.py`, `train.py`, `artifacts/` |
| `check_submission.py` | Runs a package exactly as it will be run in marking |
| `harness/` | Code used by `check_submission.py` and in marking |
| `requirements.txt` | The exact library versions used in marking |

Run the following commands from the repository root in a terminal (macOS/Linux), Command Prompt or PowerShell (Windows).

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
