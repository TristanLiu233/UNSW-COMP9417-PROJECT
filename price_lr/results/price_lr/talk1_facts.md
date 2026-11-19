# 讲解会 #1（11-17）P5 事实清单

只列事实和出处，不是讲稿。除特别注明外，范围都是 2015-01-01 … 2024-12-31（3,653 天），BalAcc 单位为 %。

1. **两个电价模型在 2014 年验证集上接近：** LR（`log_interactions31`，τ = 0.42）79.12；HGBC（P1 `boost_price`，τ = 0.45）79.19（本机复现值）。
   出处：`_group_main/logistic_price/results/frozen/validation_2014.csv`；`results/price_hgbc/official/`。
2. **冻结后，树模型退化得更快：** 冻结 LR 75.09，冻结 HGBC 70.08，b1 72.67。冻结 LR 到 2023–2024 年才落到 b1 以下，冻结 HGBC 从 2017 年起就低于 b1。2015–2026 全期（含 2025–2026，组内已看过）：LR 74.62，HGBC 70.34。
   出处：`results/price_lr/strategy_comparison_dev.csv`、`yearly_dev.csv`；`results/price_hgbc/m6/`；P4 `handoff/main_comparison.csv`。
3. **LR 的 CUSUM 标定：** 误差是 2014 年样本外的日均 log loss；q99 = 1.122，μ₀ = 0.455，σ₀ = 0.204，k = 0.5，h = 6（ARL₀ 中位数 3650 天）；2014 年内 0 次报警。用的规则和 HGBC 的 M4 完全一样，HGBC 那边 h 也是 6。
   出处：`results/price_lr/cusum/cusum_config.json`。
4. **LR 的 S0–S3：** S0 75.09，S1 76.17，S2 76.30，S3 75.97。三种重训策略都显著优于 S0：+1.09 [0.85, 1.32]，+1.22 [0.95, 1.49]，+0.88 [0.68, 1.05]。S2 − S1 = +0.13 [−0.05, +0.30]，不显著；S3 − S1 = −0.21 [−0.38, −0.05]。
   出处：`paired_bootstrap_dev.csv`。
5. **报警后重训没有比定期重训更好。** S3 在 2016-03 到 2019-09 之间用完 6 次预算，之后 70 次报警中有 41 次因预算被抑制、23 次因冷却期被抑制。HGBC 的情况相同：它在 2019-03 用完预算，S3 − S1 = −0.67。
   出处：`s3_events.csv`；`results/price_hgbc/m6/`。
6. **冻结 LR 只监测不重训时，2022–2024 年报警最多**（51、39、47 次）。误差型检测器只能说明误差变大了，不能单独区分是 P(X) 变了还是 P(Y|X) 变了。
   出处：`handoff/p3_price_cusum/s0_monitor_alarms_by_year.csv`。
7. **HGBC 的敏感性分析结果是事后的：** 方案 B 比方案 A 高 0.55 个百分点，但仍不优于 S1，而且是在看过回放结果之后得到的。因此 LR 用的是 HGBC 官方的方案 A，没有改。
   出处：`results/price_hgbc/m6/sensitivity/`。
8. **Stage 3 的暂定选择**（规则在看到结果之前写好，见设计第 11 节）：S2 第一，但和 S1 的差异不显著；两者都重训 4 次，规则规定扩展窗口优先，所以 **P = S1，R = S2**。11-18 上午用 2025–2026 做否决检查：只有 S1 低于 b1、而 S2 不低于 b1 时才改用 S2。
   出处：`stage3_provisional.json`。
9. **最终训练和 Stage 2 用的是同一份代码：** S1、S3 的每次重训都直接调用 P4 的 `fit_final`。最终模型就是 `fit_final(data, 2026-10-31)`，P4 已经跑出 τ = 0.49，明天重跑核对。
   出处：设计第 2、11 节；P4 `notes/final_training.md`。
10. **live 周 G 的历史类比：** 每年训练到 10-31，预测 12-07 … 12-13，共 10 年。中位数 +4.8 个百分点，中间 80% 为 [−2.9, +11.2]，10 周中有 7 周 G > 0。这是描述性的类比，不是预测区间。
    出处：`g_analog/g_analog_summary.json`。
