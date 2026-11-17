# price_lr：小组选定电价模型 LR 的 CUSUM 标定与 S0–S3 重训策略（M7）

本文件夹是 P5 的电价工作线第 7 阶段。它面向小组在 11-15 选定的电价模型：P4 的 Logistic Regression `log_interactions31`（C=1，class_weight=None，冻结 τ₀=0.42）。代码在 `src/price_lr/`，测试在 `tests/price_lr/`，结果在 `results/price_lr/`。

> 本次只新增 `price_lr/` 这一个顶层文件夹，没有改动其他组员的文件。P4 的 `logistic_price/` 只通过 import 使用，没有复制或修改。

## 预注册设计

[`results/price_lr/strategy_design.md`](results/price_lr/strategy_design.md)，第 2 版，2026-11-17 在任何回放之前确认后冻结。要点如下：

* **重训方法：** S1、S3 每次重训都直接调用 P4 的 `logistic_price.final_training.fit_final(data, D)`，与 Stage 3 的最终训练是同一份代码。S2 使用 `src/price_lr/fit_sliding.py`，逻辑逐行照搬 `fit_final`，只把训练窗口改为最近三个日历年；start = 2010-01-29 时与 `fit_final` 逐位一致。
* **CUSUM：** 与 HGBC 的 M4 规则相同，参考误差换成 LR 在 2014 年的样本外预测（P4 validation 列）。
* **策略：** S0 冻结；S1 扩展窗口，在 2016、2018、2020、2022、2024 年末重训；S2 三年滑动窗口，重训日期相同；S3 由 CUSUM 报警触发，冷却期 180 天，最多重训 6 次。
* **评估：** 只用 2015–2024。2025–2026 只做因果模拟，不打分。
* **Stage 3：** 决策规则见设计第 11 节，G 的历史类比见第 12 节。

## 结果（2015-01-01 … 2024-12-31，3,653 天，BalAcc %）

| 策略 | BalAcc | − b1 | − S0（95% 配对区间） | 生效重训 |
|---|---|---|---|---|
| S0 冻结 | 75.09 | +2.42 | — | 0 |
| S1 扩展窗口 | 76.17 | +3.51 | +1.09 [0.85, 1.32] | 4 |
| S2 三年滑动窗口 | 76.30 | +3.64 | +1.22 [0.95, 1.49] | 4 |
| S3 CUSUM 触发 | 75.97 | +3.30 | +0.88 [0.68, 1.05] | 6 |
| b1 / b7 | 72.67 / 69.89 | — | — | — |

* S2 − S1 = +0.13 [−0.05, +0.30]，区间包含 0；S3 − S1 = −0.21 [−0.38, −0.05]。
* S3 在 2019-09-17 之前用完了 6 次重训预算。
* **LR CUSUM：** q99 1.122，μ₀ 0.455，σ₀ 0.204，k 0.5，h 6（ARL₀ 中位数 3650 天）。2014 年内没有报警。
* **Stage 3 暂定（未使用 2025–2026）：** P = S1，备选 R = S2。11-18 用 2025–2026 做否决检查后才最终确定。
* **G 历史类比**（每年训练到 10-31，预测 12-07 … 12-13，2015–2024）：中位数 +4.8 个百分点，中间 80% 为 [−2.9, +11.2]。这只是描述性的类比，不用于选策略。

逐项自检见 [`results/price_lr/selfcheck/M7.md`](results/price_lr/selfcheck/M7.md)，其中记录了两处数值说明：refit 与 P4 文件相差约 1e-14；一次预测的行数不同时，概率相差约 1e-16。给组员的交接包：

* `results/price_lr/handoff/p4/`：§4.2 和 §5 的材料，包括比较表、models_log 和 G 类比；
* `results/price_lr/handoff/p3_price_cusum/`：§4.3 的材料，包括 CUSUM 配置、报警表，以及 LR 与 HGBC 重训日期的对照。

## 运行环境

与 `price_hgbc/` 相同，代码假定在一个工作区根目录下运行。根目录的结构如下：

```text
<workspace>/
  src/price_hgbc/  src/price_lr/        # 从 price_hgbc/src 和 price_lr/src 复制
  tests/price_lr/                       # 从 price_lr/tests 复制
  results/price_hgbc/  results/price_lr/
  comp9417_project_kit/                 # 本仓库 baseline 分支（116e296）的 worktree；data/ 下放课程 CSV
  _group_main/                          # 本仓库 main 的 worktree，固定在 c23c959（只读）
```

```sh
git worktree add comp9417_project_kit baseline
git worktree add --detach _group_main c23c959
python -m src.price_lr.cusum_cal      # CUSUM 标定（只读到 2015-01-01 00:00）
python -m src.price_lr.replay run     # S0–S3（约 30 秒）
python -m src.price_lr.replay s3-daily
python -m src.price_lr.evaluate       # 只评 2015–2024；写出 stage3_provisional.json
python -m src.price_lr.g_analog
python -m src.price_lr.consistency
python -m src.price_lr.handoff
pytest tests/price_lr -q              # 12 个测试
```

`src/price_lr/env.py` 每次运行都会断言 `_group_main` 位于 `c23c959`，并核对 10 个 P4 文件和课程 CSV 的 SHA-256。main 之后的 `f845c28`（PR #8）只新增了 P4 的说明文档，不影响这些文件。

环境：Python 3.11.8，scikit-learn 1.9.1，numpy 2.4.6，pandas 3.1.0。
