# Stage 3 自检：否决检查、最终模型、G 补上 2025（2026-11-18）

设计：`results/price_lr/strategy_design.md`（第 2 版，11-17 冻结），今天没有改任何条款或参数。代码：`src/price_lr/stage3.py`、`final_check.py`、`g_analog.py add2025`、`handoff.py stage3-p4 / stage3-p1-p3`。环境 `.venv`（Python 3.11.8，sklearn 1.9.1，numpy 2.4.6，pandas 3.1.0）。小组代码 `_group_main/`（`c23c959`，只读）。

## 1. 逐项核对

| 设计要求 | 结果 | 证据 |
|---|---|---|
| 运行前：输入 SHA、M7 测试、`stage3_provisional.json` 未变 | PASS：`check_inputs()` 通过；`pytest tests/price_lr` 12 passed；provisional SHA `94bbcde1…` 和 11-17 交给 P4 的副本一致，修改时间 11-17 23:04 | `stage3.py` 开头的断言 |
| 打开前先登记 | PASS：`data_access_log.json` 的 `events` 在打分前写入，范围 2025-01-01 … 2026-10-31 | `data_access_log.json` |
| §11.4 计算 S0–S3、b1、b7 的 Acc、BalAcc（669 天） | PASS：全期、年度、月度表；类别 1 15,705 档，类别 0 16,407 档 | `stage3/holdout_2025_2026_*.csv` |
| §11.4 S3 按因果顺序继续，预算从 2015-01 起累计 | PASS：重新跑 S3，预测逐位一致，报警日和重训日与 M7 相同；2025–2026 共 17 次报警，全部因预算被抑制，0 次重训 | `stage3/stage3_meta.json`、`s3_events_2025_2026.csv` |
| §11.4 否决规则（未取整的数值，严格小于） | PASS：P = S1 79.9015 ≥ b1 76.9747 → 保留 P；R = S2 80.7102 | `stage3/veto.json`；`test_veto_follows_rule_mechanically` |
| §11.4 所有策略完整报告，并写明起作用的规则 | PASS | `stage3/decision.md` 第 2、3 节 |
| §11.5 扩展窗口 → P4 `fit_final(data, 2026-10-31)`，复现 τ = 0.49、293,760 行 | PASS：τ、行数、折边界、36 个候选、recipe、C、种子，以及 τ 处的选择分数（6 位小数）都和 P4 一致 | `stage3/final_artifact_check.json`；`test_final_artifact_required_checks` |
| §11.5 live 周模型冻结，CUSUM 状态不进 package | 写进 `handoff/final_config/README.md` | — |
| §11.6 记录：P、R、比较值、否决判断、工件 SHA、`training.json` | PASS | `stage3/decision.md`、`stage3/final_artifact/` |
| §12.4 Y = 2025，n = 11，n = 10 的版本分开保留 | PASS：2015–2024 的 20 行和 11-17 文件逐位一致；11-17 的文件没动；2025 年的两次逐日接口抽查都逐位一致 | `g_analog/g_analog_*_2015_2025.*`；`test_g_analog_2025_added_apart` |
| §12.5 补充窗口只用 2015–2024 | PASS：仍是 n = 15 | `g_analog_summary_2015_2025.json` |
| 2025–2026 的成绩只出现在 `stage3/`、`g_analog/*_2015_2025.*` 和交接副本里 | PASS | `test_holdout_scores_only_under_stage3`；`test_no_holdout_scores_in_summaries` |
| 例行测试 | 见第 3 节 | — |

## 2. 偏离和待办

1. **最终工件的 SHA 和 P4 记录的不同。** 本机是 `b44d5df3…`，P4 是 `7e898b2a…`。
   * **必须复现的项全部一致。** 本机重跑两次 SHA 相同；CLI 结果和进程内 `fit_final` 逐位一致；672 档 `PriceClassifier.predict` 正常。
   * **P4 的工件没有提交到任何分支，**所以系数级的比较还没做。
   * **可能原因：** 和 M7 偏离第 1 条同类，即 lbfgs 在不同机器上的浮点差异。两边环境不同：本机 Python 3.11.8、macOS 26.6.2；P4 是 Python 3.11.18、macOS 26.3.1。这一点还没有证实。
   * **待办：** 拿到 P4 的文件后运行 `python -m src.price_lr.final_check --reference <目录>`。提交用 P4 的工件。
2. **G 交给 P4 晚了。** 原定 14:00 前交，`g_analog_summary_2015_2025.json` 到 11-18 15:51 才生成，发送时间以实际为准。
3. **没有改变任何预注册参数。** 2025–2026 上的描述性配对 bootstrap（S2 − S1 = +0.81 [+0.36, +1.27]）没有参与决策。

## 3. 测试

* `pytest tests/price_lr -q`：18 passed（M7 的 12 个，加今天新增的 6 个）。
* `pytest tests/price_hgbc -q -m "not archive and not historical_repro"`：201 passed，2 skipped（11-18 复跑，和基准相同）。

## 4. 记下的事实（不用于任何选择）

* 2025 年 S1 的 BalAcc 是 81.97，b1 是 79.42；2026 年 1–10 月 S1 是 77.48，b1 是 73.87。
* 冻结 LR（S0）2025–2026 的 BalAcc 是 71.01，比 b1 低 5.97；S0 只监测时，2025 年报警 64 次，是 2015 年以来最多的一年。
* S3 自 2019-09-17 起一直用同一个模型（τ = 0.37），2025–2026 的 BalAcc 是 74.72，低于 b1。
* G 的 Y = 2025 那一周，b1 的 BalAcc 达到 88.53，是 11 年里最高的；扩展窗口的 G 是 +1.85。
