# price_hgbc — 电价方向 HGBC、CUSUM 漂移检测与重训策略

本文件夹 `price_hgbc/` 包含小组项目中 **NSW 次日电价方向（`nsw_price_up`）** 这条工作线的全部内容：代码在 `src/price_hgbc/`，测试在 `tests/price_hgbc/`，结果在 `results/price_hgbc/`。
目标定义与官方一致：`nsw_price_up = 1[P(D+1, t) > P̄(D)]`，日界、标签、基线和评分全部调用 `harness/dayahead.py`。

工作范围：
- HistGradientBoostingClassifier（正式模型为 P1 的 `boost_price`）；
- 基于电价误差的漂移检测（CUSUM）；
- 自适应重训策略 S0–S3；
- 类别比例、预测表现与统计分析；
- 冻结模型与逐日预测接口。

> 本工作线只在仓库中新增了 `price_hgbc/` 这一个顶层文件夹，没有改动其他组员的文件。下文所有路径和命令都相对于 `price_hgbc/`。

---

## 1. 阶段进度（M0–M6 已完成；下一步 M7）

| 阶段 | 内容 | 状态与关键结果 | 自检报告 |
|---|---|---|---|
| M0 | 环境、kit、基线、公共工具 | 2014 年 b1 BalAcc 73.986%，b7 BalAcc 74.282% | `results/price_hgbc/selfcheck/M0.md` |
| M1 | 自研 HGBC 骨架与 τ 搜索 | 已归档为非正式敏感性分析 | `selfcheck/M1.md`（输出在 `_archive_m1m2/`） |
| M2 | 扩展特征与 48 组调参 | 结论为保留 M1；已归档 | `selfcheck/M2.md` |
| M2.5 | 确立正式参照 | **P1 `boost_price`**：18 列，15 叶，τ = 0.45，random_state = 17。本环境 2014 BalAcc **79.187%**（文档记载 79.300%，差 −0.113，已作为历史差异记录） | `selfcheck/M2_5.md` |
| M3 | 解释性分析（消融、permutation、电价尖峰） | “昨日均价与价差”组贡献最大（消融 −2.72 pp） | `selfcheck/M3.md` |
| M4 | 误差型 CUSUM 标定（只用 2014） | 日均 log loss；k = 0.5，**h = 6**，ARL₀ 中位数 912.5 天；方案 **A**（重训后参考值不变） | `selfcheck/M4.md` |
| M5 | 冻结 Stage 1 模型；逐日预测接口 | 训练 2010-01-29 至 2014-12-31（86,304 档）；2015–2026 批量预测与逐日预测**逐位一致** | `selfcheck/M5.md` |
| M6 | 策略 S0–S3 的因果回放 | 2015–2024 BalAcc：S0 70.08、**S1 75.38**、S2 73.80、S3 74.71（b1 72.67） | `selfcheck/M6.md` |
| M6 敏感性 | 预注册的 S3 变体 | 方案 B +0.55 pp、k = 0.75 +0.49 pp，均优于 S3 对照，但都没有超过 S1，因此不采用 | `selfcheck/M6_sensitivity.md` |

**2025–2026 是留出期**（留给 M8）：只做因果预测与策略模拟，**没有计算任何评分**，也没有用于任何选择。

---

## 2. 正式模型与依赖版本（重要）

- 正式模型：P1 `boost_price`。所有 HGBC 参数从 `results/price_hgbc/official/boost_price_ref.json`（REF）读取。
- 特征代码：直接 import P1 的 `ablation/features.py`（`load_study`、`build_features`、`_construct`、`CORE`）和 `ablation/models.py`，**没有复制或修改**。
- **所有结果都基于本仓库 `baseline` 分支的 commit `116e296`。** 之后 `main` 上的 `ablation/features.py` 和 `ablation/models.py` 有改动，`main` 也新增了 `src/common/features.py`（含节假日的共享特征表）。
  - 本工作线**尚未**换用这两处新代码。
  - 测试固定了 `ablation/` 文件的 SHA-256：如果在新代码上运行，测试会直接失败，不会悄悄得出不同的数字。
  - 是否迁移到 `src/common/features.py`，需要和 P1 讨论后作为新阶段处理。

---

## 3. 环境准备（Setup）

先进入本文件夹，再执行：

```bash
cd price_hgbc
```

```bash
git worktree add comp9417_project_kit baseline
```

```bash
echo "price_hgbc/comp9417_project_kit/" >> "$(git rev-parse --git-common-dir)/info/exclude"
```

```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

```bash
python -m pip install -r comp9417_project_kit/requirements.txt -r tests/price_hgbc/requirements-dev.txt
```

再把课程数据 `nem_weather_halfhourly_2010_2026.csv` 放到 `comp9417_project_kit/data/`（数据不入库）。

说明：
- `git worktree` 一行把 `baseline` 分支检出到 `price_hgbc/comp9417_project_kit/`，代码固定从这里读取 kit。
- `echo` 一行只在本地生效，让主仓库忽略这个工作树，不会改动共享的 `.gitignore`。
- 部分编辑器（例如 VS Code 的源代码管理面板）会把这个工作树显示为另一个仓库，并把其中的 `data/` 列为未提交更改。这属于正常现象，它不会进入任何提交。
- 如果找不到 `comp9417_project_kit/harness/dayahead.py`，`src/price_hgbc/kit.py` 会直接报错并给出上面的命令。
- 环境：Python 3.11，依赖版本以 kit 的 `requirements.txt` 为准；pytest 版本见 `tests/price_hgbc/requirements-dev.txt`。

---

## 4. 运行测试

快速回归：

```bash
pytest tests/price_hgbc -q -m "not slow and not archive and not historical_repro"
```

完整验收（包括 slow）：

```bash
pytest tests/price_hgbc -q -m "not archive and not historical_repro"
```

历史复现诊断（预期 3 项失败，均已获授权）：

```bash
pytest tests/price_hgbc -q -m "historical_repro"
```

| 标记 | 含义 |
|---|---|
| `slow` | 耗时超过 30 秒 |
| `historical_repro` | 只有 3 项经授权的历史逐位复现检查（组内 79.29988% 与跨平台浮点差异）。失败已记录为已接受的历史差异，不阻塞后续阶段 |
| `archive` | M1/M2 的非正式结果；不带 `-m` 运行 pytest 时默认排除 |

本地最近一次完整验收：**201 passed，2 skipped（需手动开启的 Docker 测试），0 failed**。

---

## 5. 复现各阶段

在 `price_hgbc/` 下执行：

| 阶段 | 命令 | 主要输出 |
|---|---|---|
| M2.5 正式参照 | `python -m src.price_hgbc.official_ref` | `results/price_hgbc/official/` |
| M3 解释性分析 | `python -m src.price_hgbc.ablate`；`python -m src.price_hgbc.m3_permutation`；`python -m src.price_hgbc.m3_spike`；`python -m src.price_hgbc.m3_handoff` | `results/price_hgbc/m3/` |
| M4 CUSUM 标定 | `python -m src.price_hgbc.m4` | `results/price_hgbc/m4/cusum_config.json` 等 |
| M5 冻结模型与预测 | `python -m src.price_hgbc.m5`；`python -m src.price_hgbc.m5_readme` | `results/price_hgbc/m5/` |
| M6 策略回放与评估 | `python -m src.price_hgbc.m6 run`；`python -m src.price_hgbc.m6 s3-daily`；`python -m src.price_hgbc.m6_eval` | `results/price_hgbc/m6/` |
| M6 敏感性 | `python -m src.price_hgbc.m6_sensitivity` | `results/price_hgbc/m6/sensitivity/` |

所有步骤都是确定性的：同一环境下重复运行，结果逐位相同，相应的检查写在测试里。

---

## 6. 给其他组员的接口

- **逐日预测**：`src/price_hgbc/m5.py` 中的 `predict_price(model, tau, history, weather_next)`。
  - 输入与 kit 的 `Forecaster.predict` 相同。
  - 返回值以 `weather_next.index` 为索引，包含 `p_up` 和 `nsw_price_up` 两列。
- **冻结预测**（2015–2026）：`results/price_hgbc/m5/predictions.parquet`。索引为时间戳，列为 `day`、`period`、`p_up`、`pred`，不含标签。
- **策略预测**：`results/price_hgbc/m6/s{0..3}_predictions.parquet`，在上面的列之外再加 `strategy` 和 `model_id`。模型日志在 `results/price_hgbc/m6/models_log.csv`。
- **CUSUM 配置**：`results/price_hgbc/m4/cusum_config.json`。参数必须从文件读取完整精度值，不要使用文档里的舍入值。

---

## 7. 待确认 / 下一步

1. 与 P1、P3 确认预测文件和策略输出的字段格式（目前仓库里没有 P3 的 `run_strategy` 接口）。
2. 是否把电价特征迁移到 `main` 上新的 `src/common/features.py` 和 `ablation/` 代码；迁移后需要重新冻结。
3. M7：Stage 3 策略选择。依据 M6，S1 是最强候选；如果必须用检测触发的策略，S3-B（方案 B）是事后选出的备选，需要如实注明。
4. M8：在 2025–2026 留出期上做最终比较。

---

## 8. 目录

```
price_hgbc/
  README.md              本文件
  src/price_hgbc/        代码（kit.py 定位 kit；official_ref、ablate、m3_*、cusum、m4、m5、m6、m6_eval、m6_sensitivity 等）
  tests/price_hgbc/      自动化测试（conftest 中注册了 slow / historical_repro / archive 三个标记）；requirements-dev.txt
  results/price_hgbc/
    official/            M2.5 正式参照（REF、2014 预测、模型）
    m3/ m4/ m5/ m6/      各阶段输出（m6/sensitivity/ 为预注册的敏感性分析）
    selfcheck/           各阶段自检报告与测试日志（M0.md … M6_sensitivity.md）
    _archive_m1m2/       M1/M2 归档（MANIFEST.csv 记录搬移前后的哈希）
  comp9417_project_kit/  （本地自建，不入库）baseline 分支的工作树和课程数据
```

注：部分自检日志和环境记录中保留了生成结果时本机的绝对路径，只用作证据，不影响复现。
