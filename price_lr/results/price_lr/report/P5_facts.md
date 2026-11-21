# P5 事实清单：Results 4.3 / 4.4（LR 电价）

由 `python -m src.price_lr.report_assets` 生成，只给自己写作时参考，**不上传 Overleaf**。只列事实和出处，不是报告文字。数值保留两位小数，和 `stage3/decision.md` 一致。路径相对于 `results/price_lr/`。

## 4.3 退化与漂移检测（价格部分）

### 检测器设置
* CUSUM 作用在每日 log loss 上：k = 0.5，h = 6，参考期为 2014 年，μ0 = 0.4549，σ0 = 0.2043，截断上限 q99 = 1.1218。出处：`cusum/cusum_config.json` → `schemes.main`。
* 已有的标定图：`cusum/fig_cusum_path_2014.png`、`fig_arl0_by_h.png`、`fig_delay_by_shift.png`。这部分可能属于 Methodology，归属待定。

### 冻结 LR（S0）：报警和退化
* 2015–2024 全期：S0 的 BA 为 75.09，b1 为 72.67，差 +2.42 [+1.98, +2.86]（95% CI）。出处：`strategy_comparison_dev.csv`、`paired_bootstrap_dev.csv`。
* S0 低于 b1 的年份：2023, 2024（-1.88, -3.06）。出处：`yearly_dev.csv`。
* 2025–2026 S0 的 BA 为 71.01，b1 为 76.97（差 -5.97）。出处：`stage3/holdout_2025_2026_comparison.csv`。
* 只监测时每年的报警数：2015 0，2016 15，2017 14，2018 8，2019 15，2020 6，2021 23，2022 51，2023 39，2024 47，2025 64，2026 26。没有报警的年份：2015；最多的是 2025 年。出处：`stage3/s0_monitor_alarms_by_year_2015_2026.csv`，已和逐日文件 `s0_monitor_cusum.csv` 及 `stage3/s0_monitor_cusum_2025_2026.csv` 重数核对。
* 报警多的年份 2022（51）、2024（47）、2025（64），对应 S0 − b1 分别为 +0.45、-3.06、-5.97（2025–2026 合计）。2022 年 S0 高于 b1。
* 差值用未取整的 BA 计算。例如 2022 年为 +0.45；用 P4 `tab:p4years` 里取整后的数相减是 +0.44。

### 必须讲清楚的口径（4.3）
* S0 的监测不重训。报警数是检测器的输出，不是模型更新次数。
* 检测器针对的是模型误差（log loss），不是输入分布。课程问的"输入变化是否总会让预测变差"，价格这边没有做输入漂移检测器，归属待定（P3 做了需求的 `input_detector`）。
* "报警年份和退化年份相符"只是描述，不能当作因果。

## 4.4 模型自适应（价格部分）

### 2015–2024（3,653 天，Stage 2 回放）
* S0：BA 为 75.09，Acc 为 75.64，比 b1 高 +2.42，重训生效 0 次。出处：`strategy_comparison_dev.csv`。
* S1：BA 为 76.17，Acc 为 76.12，比 b1 高 +3.51，重训生效 4 次，比 S0 高 +1.09 [+0.85, +1.32]。出处：`strategy_comparison_dev.csv`。
* S2：BA 为 76.30，Acc 为 76.05，比 b1 高 +3.64，重训生效 4 次，比 S0 高 +1.22 [+0.95, +1.49]。出处：`strategy_comparison_dev.csv`。
* S3：BA 为 75.97，Acc 为 75.84，比 b1 高 +3.30，重训生效 6 次，比 S0 高 +0.88 [+0.68, +1.05]。出处：`strategy_comparison_dev.csv`。
* S2 − S1 = +0.13 [-0.05, +0.30]，区间包含 0；S3 − S1 = -0.21 [-0.38, -0.05]。出处：`paired_bootstrap_dev.csv`。
* S3：2015–2024 报警 70 次，其中重训 6 次，因冷却期抑制 23 次，因预算抑制 41 次。最后一次重训在 2019-09-17，之后预算用完。出处：`s3_events.csv`、`models_log.csv`。
* τ 每次重训都重新选：S1 依次为 0.42 → 0.27 → 0.34 → 0.40 → 0.45 → 0.49（`models_log.csv` → `tau`）。
* 已有的图：`fig_m7_yearly_balacc.png`、`fig_m7_tradeoff.png`、`fig_m7_s3_cusum.png`。报告用 `report/figures/` 里重画的版本。

### Stage 3（2025-01-01 … 2026-10-31，669 天）
* P = S1，R = S2。BA(P) = 79.90，BA(R) = 80.71，BA(b1) = 76.97。否决没有触发，最终 = S1。出处：`stage3/veto.json`。
* S3 在这两年报警 17 次，全部因预算被抑制，没有重训。出处：`stage3/s3_events_2025_2026.csv`。
* 描述性的 S2 − S1 = +0.81 [+0.36, +1.27]，不参与决策。出处：`stage3/holdout_2025_2026_paired_bootstrap_DESCRIPTIVE.csv`。
* 最终模型：P4 的 `fit_final(data, 2026-10-31)`，τ = 0.49，293,760 行。G（Live-Week 已用）：中位数 +4.60。出处：`stage3/decision.md` 第 6 节、`g_analog/`。

### 必须讲清楚的口径（4.4）
* 2025–2026 **不是**未见过的留出集。S0 在这两年的成绩组内已经看过；P4 用 2025-11 … 2026-10 选过 τ。只能说"按预注册规则第一次给 S1–S3 打分"（`stage3/decision.md` 第 5 节）。
* 2025–2026 的 bootstrap 只做描述。选择依据是 11-17 冻结的规则（`strategy_design.md` 第 11 节）。
* S2 在两段时间里的 BA 都略高于 S1。选 S1 的依据是规则：区间包含 0，重训次数相同，扩展窗口优先。不是因为 S1 更准。
* 本机最终工件的 SHA（`b44d5df3…`）和 P4 的（`7e898b2a…`）不同。11-21 已核对：`training.json` 相同，系数最大差 1.5e-14，672 档类别完全相同（`stage3/final_artifact_check.json`）。提交用 P4 的工件。
* 2026 只有 1–10 月。
* S0 的 2015–2024 逐年 BA 已在 P4 的 `tab:p4years`（4.2 节）里，不要重复列出。
