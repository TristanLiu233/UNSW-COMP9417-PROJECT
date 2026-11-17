"""M7 handoff packs (tables and facts only, no report prose). 2015-2024 content only.
Usage: python -m src.price_lr.handoff
"""
import json
import os
import shutil

import pandas as pd

from .env import RESULTS

H = os.path.join(RESULTS, "handoff")
DEV_END = pd.Timestamp("2024-12-31")


def cp(src, dst_dir, name=None):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(os.path.join(RESULTS, src), os.path.join(dst_dir, name or os.path.basename(src)))


def main():
    p4, p3 = os.path.join(H, "p4"), os.path.join(H, "p3_price_cusum")
    for f in ("strategy_comparison_dev.csv", "yearly_dev.csv", "monthly_dev.csv", "paired_bootstrap_dev.csv",
              "tradeoff_dev.csv", "stage3_provisional.json", "fig_m7_yearly_balacc.png", "fig_m7_tradeoff.png",
              "g_analog/g_analog_weeks.csv", "g_analog/g_analog_summary.json", "g_analog/g_age37_43_deployed.csv"):
        cp(f, p4)
    log = pd.read_csv(os.path.join(RESULTS, "models_log.csv"))
    cols = ["strategy", "model_id", "trigger", "alarm_day", "cutoff", "train_start", "train_end", "train_rows", "tau",
            "val_start", "val_end", "val_BalAcc_at_tau", "effective_from", "effective_to", "trainer", "model_path", "model_sha256"]
    log[cols].to_csv(os.path.join(p4, "models_log.csv"), index=False)
    for f in ("cusum/cusum_config.json", "cusum/reference_stats_2014.csv", "cusum/arl0_by_h.csv", "cusum/delay_by_shift.csv",
              "cusum/daily_err_2014.csv", "cusum/fig_cusum_path_2014.png", "cusum/fig_arl0_by_h.png",
              "s0_monitor_cusum.csv", "fig_m7_s3_cusum.png"):
        cp(f, p3)
    ev = pd.read_csv(os.path.join(RESULTS, "s3_events.csv"), parse_dates=["alarm_day"])
    ev[ev["alarm_day"] <= DEV_END].to_csv(os.path.join(p3, "s3_events_2015_2024.csv"), index=False)
    dly = pd.read_csv(os.path.join(RESULTS, "s3_daily_cusum.csv"), parse_dates=["day"])
    dly[dly["day"] <= DEV_END].to_csv(os.path.join(p3, "s3_daily_cusum_2015_2024.csv"), index=False, float_format="%.17g")
    # LR vs HGBC detector timing (both option A, k 0.5, h 6, cooldown 180, budget 6)
    hg = pd.read_csv(os.path.join(RESULTS, "..", "price_hgbc", "m6", "s3_events.csv"), parse_dates=["alarm_day"])
    hg = hg[hg["alarm_day"] <= DEV_END]
    lr = ev[ev["alarm_day"] <= DEV_END]
    rows = []
    for name, e in (("LR", lr), ("HGBC", hg)):
        for _, r in e[e["retrained"]].iterrows():
            rows.append({"model": name, "retrain_index": int(r["retrain_index"]), "alarm_day": r["alarm_day"].date()})
    t = pd.DataFrame(rows).pivot(index="retrain_index", columns="model", values="alarm_day").reset_index()
    t["days_LR_minus_HGBC"] = (pd.to_datetime(t["LR"]) - pd.to_datetime(t["HGBC"])).dt.days
    t.to_csv(os.path.join(p3, "s3_retrain_days_lr_vs_hgbc.csv"), index=False)
    mon = pd.read_csv(os.path.join(RESULTS, "s0_monitor_cusum.csv"), parse_dates=["day"])
    by_year = mon.groupby(mon["day"].dt.year)["alarm"].sum().astype(int).rename("frozen_LR_monitor_alarms").reset_index()
    by_year.columns = ["year", "frozen_LR_monitor_alarms"]
    by_year.to_csv(os.path.join(p3, "s0_monitor_alarms_by_year.csv"), index=False)
    cfg = json.load(open(os.path.join(RESULTS, "cusum", "cusum_config.json")))["schemes"]["main"]
    st = json.load(open(os.path.join(RESULTS, "stage3_provisional.json")))
    g = json.load(open(os.path.join(RESULTS, "g_analog", "g_analog_summary.json")))
    cmp_ = pd.read_csv(os.path.join(RESULTS, "strategy_comparison_dev.csv")).set_index("strategy")
    boot = pd.read_csv(os.path.join(RESULTS, "paired_bootstrap_dev.csv"))
    def b(a, c):
        r = boot[(boot["a"] == a) & (boot["b"] == c)].iloc[0]
        return f"{r['diff_pp']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]"
    ge = g["main"]["expanding"]
    open(os.path.join(p4, "README.md"), "w").write(f"""# P5 → P4：LR S0–S3、Stage 3 暂定选择、G 历史类比（2026-11-17）

只有表格和事实，不是报告正文。设计：`results/price_lr/strategy_design.md`（11-17 确认后冻结）。全部只评 2015-01-01 … 2024-12-31（3,653 天）；2025–2026 未打分。

| 策略 | BalAcc % | − b1 | − S0（95% 配对区间） | 2015–2024 内生效重训 |
|---|---|---|---|---|
| S0 冻结 | {cmp_.loc['S0','BalAcc']:.2f} | {cmp_.loc['S0','BalAcc_minus_b1']:+.2f} | — | 0 |
| S1 扩展，年末 ×5 | {cmp_.loc['S1','BalAcc']:.2f} | {cmp_.loc['S1','BalAcc_minus_b1']:+.2f} | {b('S1','S0')} | {int(cmp_.loc['S1','retrains_effective_in_dev'])} |
| S2 3 年滑动，年末 ×5 | {cmp_.loc['S2','BalAcc']:.2f} | {cmp_.loc['S2','BalAcc_minus_b1']:+.2f} | {b('S2','S0')} | {int(cmp_.loc['S2','retrains_effective_in_dev'])} |
| S3 CUSUM 触发 | {cmp_.loc['S3','BalAcc']:.2f} | {cmp_.loc['S3','BalAcc_minus_b1']:+.2f} | {b('S3','S0')} | {int(cmp_.loc['S3','retrains_effective_in_dev'])} |
| b1 | {cmp_.loc['b1','BalAcc']:.2f} | — | — | — |
| b7 | {cmp_.loc['b7','BalAcc']:.2f} | — | — | — |

* S2 − S1 = {b('S2','S1')}；S3 − S1 = {b('S3','S1')}。
* Stage 3 暂定（第 11 节第 1–2 条，未用 2025–2026）：第一名 {st['F']}；候选集 {st['candidate_set']}；按"重训次数 → 扩展窗口 → BalAcc"得 **P = {st['P_provisional']}**，备选 R = {st['R_backup']}。11-18 上午用 2025–2026 做否决检查后才确定。
* 如果最终是扩展窗口（S1/S3/S0），最终模型 = 你的 `fit_final(data, 2026-10-31)`，需复现 τ = 0.49；若改为 S2，则用 `fit_sliding(…, start=2023-11-01)` 替换你的 10-31 工件。
* `models_log.csv`：每个部署模型的截止日、生效日、被替换日、τ，供你的 37–43 天框架使用。

## G 的历史类比（只用于 §5，不用于选策略）

训练到 Y-10-31，预测 Y-12-07 … 12-13（模型年龄 37–43 天），Y = 2015 … 2024，每天单独 predict 48 档。

| 训练方法 | n | 中位数 | 中间 80% | 最小 / 最大 | G > 0 的周 |
|---|---|---|---|---|---|
| 扩展窗口 | {ge['n']} | {ge['median']:+.2f} | [{ge['q10']:+.2f}, {ge['q90']:+.2f}] | {ge['min']:+.2f} / {ge['max']:+.2f} | {ge['weeks_G_gt_0']} |
| 3 年滑动 | {g['main']['sliding']['n']} | {g['main']['sliding']['median']:+.2f} | [{g['main']['sliding']['q10']:+.2f}, {g['main']['sliding']['q90']:+.2f}] | {g['main']['sliding']['min']:+.2f} / {g['main']['sliding']['max']:+.2f} | {g['main']['sliding']['weeks_G_gt_0']} |

补充（实际部署模型的第 37–43 天窗口，n = {g['supplement_37_43']['n']}，描述性）：中位数 {g['supplement_37_43']['median']:+.2f}，中间 80% [{g['supplement_37_43']['q10']:+.2f}, {g['supplement_37_43']['q90']:+.2f}]。Y = 2025 那一周在 11-18 打开 2025–2026 后补上。

局限：其他年份的 12-07 … 12-13 不是周一到周日；n = 10 且年份不独立；历史用再分析天气；类比模型沿用最终训练方法但不是提交的模型；这是描述性的历史类比，不是预测区间或 live 成绩。
""")
    open(os.path.join(p3, "README.md"), "w").write(f"""# P5 → P3：电价 LR 的 CUSUM（§4.3 材料，2026-11-17）

只有表格和事实。全部只到 2024-12-31。

* 统计量：日均 log loss（概率裁剪到 [1e-6, 1−1e-6]），q99 截断；参考期 = 2014 年样本外（LR 训练 2010-01-29 … 2013-12-31，即 P4 validation 列）。
* q99 = {cfg['q99']:.4f}，μ₀ = {cfg['mu0']:.4f}，σ₀ = {cfg['sigma0']:.4f}，k = 0.5，h = {cfg['h']}（ARL₀ 中位数 {cfg['arl0_median']} 天，5% 分位 {cfg['arl0_q05']:.0f} 天；周分块 bootstrap，51 周，1000 次，seed 0）。2014 年内 0 次报警。方案 A：参考统计量重训后不更新。
* `s0_monitor_cusum.csv`：冻结 LR 只监测不重训时的每日 S 和报警；按年计数见 `s0_monitor_alarms_by_year.csv`。
* `s3_events_2015_2024.csv`：S3 的报警、抑制原因（cooldown / budget）、重训日。冷却 180 天，预算 6 次，2019-09-17 已用完。
* `s3_retrain_days_lr_vs_hgbc.csv`：同一套 CUSUM 规则下，LR 和 HGBC 的 6 次重训日期对照。
* 和 HGBC 的 M4 一样：误差型检测器只看到误差变大，不能单独区分是 P(X) 变了还是 P(Y|X) 变了。
""")
    print("written", p4, p3)


if __name__ == "__main__":
    main()
