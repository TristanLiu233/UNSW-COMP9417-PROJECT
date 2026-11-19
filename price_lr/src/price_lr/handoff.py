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


MARK0, MARK1 = "<!-- 11-18 stage3 begin -->", "<!-- 11-18 stage3 end -->"


def _prepend(readme, block):
    old = open(readme).read() if os.path.exists(readme) else ""
    if MARK0 in old:                                         # idempotent: replace an earlier 11-18 block
        old = old.split(MARK0)[0] + old.split(MARK1, 1)[1].lstrip("\n")
    open(readme, "w").write(f"{MARK0}\n{block.strip()}\n\n---\n{MARK1}\n\n{old}")


def stage3_p4():
    """11-18 additions for P4: Stage 3 decision, 2025-2026 tables, G with Y = 2025 (n = 11)."""
    p4 = os.path.join(H, "p4")
    for f in ("stage3/decision.md", "stage3/veto.json", "stage3/holdout_2025_2026_comparison.csv",
              "stage3/holdout_2025_2026_yearly.csv", "stage3/holdout_2025_2026_monthly.csv",
              "stage3/holdout_2025_2026_paired_bootstrap_DESCRIPTIVE.csv",
              "g_analog/g_analog_weeks_2015_2025.csv", "g_analog/g_analog_summary_2015_2025.json"):
        cp(f, p4)
    v = json.load(open(os.path.join(RESULTS, "stage3", "veto.json")))
    c = pd.read_csv(os.path.join(RESULTS, "stage3", "holdout_2025_2026_comparison.csv")).set_index("strategy")
    g = json.load(open(os.path.join(RESULTS, "g_analog", "g_analog_summary_2015_2025.json")))
    fc = json.load(open(os.path.join(RESULTS, "stage3", "final_artifact_check.json")))
    sel = g["selected_method"]
    y25 = {r["method"]: r for r in g["Y2025"]}
    f = lambda x: f"{x:+.2f}"
    def row(m, k):
        s_ = g["both_methods"][m][k]
        return (f"| {'扩展窗口' if m == 'expanding' else '3 年滑动'} | {s_['n']} | {f(s_['median'])} | [{f(s_['q10'])}, {f(s_['q90'])}] "
                f"| {f(s_['min'])} / {f(s_['max'])} | {s_['weeks_G_gt_0']} |")
    trows = "\n".join(f"| {s} | {c.loc[s, 'Acc']:.2f} | {c.loc[s, 'BalAcc']:.2f} | {c.loc[s, 'BalAcc_minus_b1']:+.2f} |"
                      for s in ("S0", "S1", "S2", "S3", "b1", "b7"))
    block = f"""# 11-18 更新：Stage 3 决定和 n = 11 的 G

只有表格和事实，不是报告正文。规则：设计第 11.4、12.4 节。

## Stage 3 决定：**{v['final_strategy']}（{'扩展窗口' if v['final_window'] == 'expanding' else '3 年滑动窗口'}）**

第 11.4 条否决检查（2025-01-01 … 2026-10-31，669 天）：P = {v['P']} 的 BalAcc {v['BalAcc_P']:.2f}，b1 {v['BalAcc_b1']:.2f}，P 不低于 b1，所以保留 P，没有改用 R = {v['R']}（{v['BalAcc_R']:.2f}）。
最终模型就是你的 `fit_final(data, 2026-10-31)`，你的 10-31 工件不用换。

| 策略 | Acc | BalAcc | − b1 |
|---|---|---|---|
{trows}

* S2 − S1 在这两年 = +0.81 [+0.36, +1.27]（配对 bootstrap，**只做描述，不参与决策**）。第 11.4 条写明"R 更高但 P 不低于 b1"时保留 P。
* 这两年不是未见过的留出集：S0 的成绩组内已看过，2025-11 … 2026-10 你在 11-16 用来选过 τ。
* 完整记录：`decision.md`、`veto.json`、`holdout_2025_2026_*.csv`。

## 最终工件核对

本机重跑你的 CLI：τ = 0.49、最终 293,760 行、内部 276,240 / 验证 17,520 行、36 个候选、τ 处选择分数都和你的记录一致。
但 `logistic.joblib` 的 SHA 是 `{fc['logistic_joblib_sha256'][:16]}…`，你记录的是 `{fc['P4_recorded_sha256'][:16]}…`。本机重跑两次结果相同，672 档 `PriceClassifier.predict` 正常。
**请把你的 `logistic.joblib` 和 `training.json` 发给我**，我跑 `python -m src.price_lr.final_check --reference <目录>` 比较系数（要求差 ≤ 1e-12、672 档类别相同）。提交用你的工件。

## G 的历史类比（补上 Y = 2025，用于 §5）

最终方法：{'扩展窗口' if sel == 'expanding' else '3 年滑动'}。Y = 2025：训练到 2025-10-31（{y25[sel]['train_rows']:,} 行，τ = {y25[sel]['tau']:.2f}），2025-12-07 … 12-13 BA_m {y25[sel]['BA_m']:.2f}、BA_b1 {y25[sel]['BA_b1']:.2f}，**G = {f(y25[sel]['G_pp'])}**。

| 训练方法 | n | 中位数 | 中间 80% | 最小 / 最大 | G > 0 的周 |
|---|---|---|---|---|---|
{row(sel, 'n11_2015_2025')}
{row(sel, 'n10_2015_2024')}
{row('sliding' if sel == 'expanding' else 'expanding', 'n11_2015_2025')}

* 2015–2024 的 10 行和 11-17 的文件逐位一致；n = 10 和 n = 11 分开记录（`g_analog_summary_2015_2025.json`）。
* 补充窗口（实际部署模型的第 37–43 天）不变：n = 15，中位数 +4.64，只用 2015–2024（设计第 12.5 节）。
* 局限：其他年份的 12-07 … 12-13 不是周一到周日（2025 年那周从周日开始）；n = 11 且年份不独立；历史用再分析天气；类比模型沿用最终训练方法但不是提交的模型；**Y = 2025 那一周落在你选 τ 的验证期里**。这是描述性的历史类比，不是预测区间，也不是 live 成绩。
"""
    _prepend(os.path.join(p4, "README.md"), block)
    print("written", p4)


def stage3_p1_p3():
    """11-18: final training config for P1 / P4, and 2025-2026 CUSUM records for P3."""
    st = os.path.join(RESULTS, "stage3")
    fc = json.load(open(os.path.join(st, "final_artifact_check.json")))
    t = json.load(open(os.path.join(st, "final_artifact", "training.json")))
    v = json.load(open(os.path.join(st, "veto.json")))
    fd = os.path.join(H, "final_config")
    os.makedirs(fd, exist_ok=True)
    cp("stage3/final_artifact/training.json", fd, "training_local_P5.json")
    cp("stage3/final_artifact_check.json", fd)
    f0 = t["folds"][0]
    env = fc["environment"]
    open(os.path.join(fd, "README.md"), "w").write(f"""# P5 → P1、P4：电价最终训练配置（2026-11-18）

只有配置和事实，不是报告正文。决定过程见 `results/price_lr/stage3/decision.md`。

## 配置

| 项 | 值 |
|---|---|
| 最终策略 | {v['final_strategy']}（{v['final_window']}，Stage 3 第 11.4 条否决未触发） |
| 训练函数 | P4 `logistic_price.final_training.fit_final`（main `c23c959`），不改代码 |
| 截止日 | {t['train_end']}（CSV 只读到 2026-11-01 00:00） |
| 训练窗口 | {t['training_policy']}，最终拟合 {t['train_rows']:,} 行 |
| τ 选择 | 内部训练 {f0['train_start']} … {f0['train_end']}（{f0['train_rows']:,} 行），验证 {f0['validation_start']} … {f0['validation_end']}（{f0['validation_rows']:,} 行），网格 0.25–0.60 步长 0.01 共 36 个，pooled BalAcc 最高 |
| τ | **{t['threshold']}** |
| recipe | `{t['recipe']}`（31 列） |
| C / class_weight / 种子 | {t['C']:g} / {t['class_weight']} / {t['random_state']} |
| 拟合设置 | `threadpool_limits(1)`；ConvergenceWarning 视为错误 |
| live 周 | 模型冻结；`predict` 内部不训练；S3 的 CUSUM 状态不进 package |

## 复现命令（在小组仓库根目录）

```sh
python -m logistic_price.final_training \\
  --data /absolute/path/nem_weather_halfhourly_2010_2026.csv \\
  --cutoff 2026-10-31 \\
  --output /absolute/path/NEW_DIRECTORY
```

输出 `logistic.joblib`（供 `PriceClassifier(output)` 直接加载）和 `training.json`。本机单次墙钟 2.74 秒（含解释器启动）；P4 记录 3.35 秒。整个 `train.py` 的两小时要求由 P1 在最终组合包里测。

## 工件核对

| 项 | 结果 |
|---|---|
| τ、行数、折边界、候选数、recipe、C、种子 | 全部和 P4 记录一致 |
| τ 处内部选择分数 | 78.896947 / 78.835616，和 P4 笔记一致（6 位小数；不是测试成绩） |
| 本机 `logistic.joblib` SHA-256 | `{fc['logistic_joblib_sha256']}` |
| P4 记录的 SHA-256 | `{fc['P4_recorded_sha256']}` |
| SHA 相同？ | **否**。本机重跑两次 SHA 相同；CLI 结果和进程内 `fit_final` 逐位一致；2026-10-18 … 10-31 共 672 档 `PriceClassifier.predict` 正常（一致性检查，不是成绩） |
| 待办 | P4 发来自己的 `logistic.joblib`、`training.json` 后，P5 运行 `python -m src.price_lr.final_check --reference <目录>`：要求 `training.json` 相同、系数差 ≤ 1e-12、672 档类别相同 |
| 提交用哪个工件 | **P4 的工件**；本机工件只用于核对 |

本机环境：Python {env['python']}，{env['platform']}，scikit-learn {env['sklearn']}，numpy {env['numpy']}，pandas {env['pandas']}，scipy {env['scipy']}，joblib {env['joblib']}，threadpoolctl {env['threadpoolctl']}。P4：Python 3.11.18，macOS 26.3.1，其余包版本相同。

`training_local_P5.json` 是本机生成的 `training.json`；`final_artifact_check.json` 是逐项核对记录。
""")
    p3 = os.path.join(H, "p3_price_cusum")
    for f, name in (("stage3/s0_monitor_cusum_2025_2026.csv", "OPENED_1118_s0_monitor_cusum_2025_2026.csv"),
                    ("stage3/s0_monitor_alarms_by_year_2015_2026.csv", "OPENED_1118_s0_monitor_alarms_by_year_2015_2026.csv"),
                    ("stage3/s3_daily_cusum_2025_2026.csv", "OPENED_1118_s3_daily_cusum_2025_2026.csv"),
                    ("stage3/s3_events_2025_2026.csv", "OPENED_1118_s3_events_2025_2026.csv")):
        cp(f, p3, name)
    yr = pd.read_csv(os.path.join(st, "s0_monitor_alarms_by_year_2015_2026.csv")).set_index("year")["alarms"]
    ev = pd.read_csv(os.path.join(st, "s3_events_2025_2026.csv"))
    block = f"""# 11-18 更新：2025–2026 的 CUSUM 记录（11-18 打开的数据）

文件名以 `OPENED_1118_` 开头的都是 11-18 打开 2025–2026 后才算的；下面 11-17 的部分仍然只到 2024-12-31。

* **S0 只监测：** 状态从 2015-01-01 连续计算，2015–2024 部分和 11-17 的 `s0_monitor_cusum.csv` 逐位一致。按年报警数：2022 {yr[2022]}、2023 {yr[2023]}、2024 {yr[2024]}、**2025 {yr[2025]}**、**2026（1–10 月）{yr[2026]}**。
* **S3：** 预算 2019-09-17 已用完。2025–2026 共 {len(ev)} 次报警，全部因预算被抑制（suppressed_reason = {', '.join(sorted(set(ev['suppressed_reason'])))}），没有重训。
* 最终选的是 S1（定期扩展窗口），不是 S3；S3 的 CUSUM 状态不进 package。
* 解读边界不变：误差型检测器只看到误差变大，不能单独区分 P(X) 还是 P(Y|X) 变了；2025–2026 的报警增多只是事实，原因要另找证据。
"""
    _prepend(os.path.join(p3, "README.md"), block)
    print("written", fd, p3)


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    stage3_p4() if a == ["stage3-p4"] else stage3_p1_p3() if a == ["stage3-p1-p3"] else main()

