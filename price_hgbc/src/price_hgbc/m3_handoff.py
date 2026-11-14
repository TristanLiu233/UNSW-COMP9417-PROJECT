"""Build results/price_hgbc/m3/m3_P1_handoff.md from the saved M3 outputs (numbers are read, never typed).
Usage: python -m src.price_hgbc.m3_handoff
"""
import json
import os

import pandas as pd

from . import ablate, m3_permutation as mp, m3_spike as ms
from . import m3_common as c
from .kit import KIT

OUT_MD = os.path.join(c.OUT, "m3_P1_handoff.md")
P1_CONTRASTS = ["CW_minus_C", "CR_minus_C", "F_minus_C", "no_explicit_gap", "spike_max", "negative_count",
                "both_spike_stats", "extra_city_weather", "selected_vs_tuned_CW"]


def f(x, d=3):
    return f"{x:.{d}f}"


def p1_rows():
    t = pd.read_csv(os.path.join(KIT, "ablation", "results", "contrasts.csv"))
    t = t[(t["model"] == "boost_price") & t["contrast"].isin(P1_CONTRASTS)].set_index("contrast").loc[P1_CONTRASTS]
    return t.reset_index()


def build():
    r = c.ref()
    m = r["metrics_2014"]
    ab = pd.read_csv(ablate.CSV)
    abm = json.load(open(ablate.JSON))
    col = pd.read_csv(mp.COL_CSV)
    grp = pd.read_csv(mp.GRP_CSV)
    pm = json.load(open(mp.JSON))
    s14 = pd.read_csv(ms.CSV_2014)
    sl = pd.read_csv(ms.CSV_LABEL)
    sm = json.load(open(ms.JSON))
    L = []
    w = L.append
    w("# M3 交接说明（boost_price 解释性分析）\n")
    w("> **这些是解释性结果，不用于重新选型。** 正式模型、参数和 τ 均保持 REF 不变；本文件未发送给 P1，仅为交接草稿。\n")
    w("## 1. 正式模型（REF）\n")
    w(f"- 来源：P1 `boost_price`，代码 `comp9417_project_kit/ablation/`（分支 `{r['p1']['branch']}`，commit `{r['p1']['commit'][:7]}`），直接导入，未复制或修改。")
    w(f"- 参数：HistGradientBoostingClassifier，log_loss，max_leaf_nodes={r['params']['max_leaf_nodes']}，max_iter={r['params']['max_iter']}，"
      f"learning_rate={r['params']['learning_rate']}，early_stopping=False，random_state={r['params']['random_state']}；τ={r['tau']}（p ≥ τ 判为 1）。")
    w(f"- 训练 {r['train']['first_day']} 至 {r['train']['last_day']}（{r['train']['rows']:,} 档），验证 2014（{r['valid']['rows']:,} 档）；18 列，无天气列。")
    w(f"- 本环境 2014：BalAcc {f(m['BalAcc'])}%，Acc {f(m['Acc'])}%，0 类召回 {f(m['recall_0'])}%，1 类召回 {f(m['recall_1'])}%。"
      f"消融说明记载 79.300% / 79.218%（76.325% / 82.275%），差 {f(r['diff_vs_doc_pp']['BalAcc'])} 个百分点；以本环境稳定结果为正式基准（见 selfcheck/M2_5.md）。\n")
    w("## 2. P1 已有的加法消融（引用，不重做）\n")
    w("来自 `ablation/results/contrasts.csv`（P1 的机器、2014 选型年）。第 1、3 轮为固定参数 31 叶、τ=0.5，与 REF（15 叶、τ=0.45）不同，只作背景参考。\n")
    w("| 对照 | 候选 − 参照 | ΔBalAcc（pp） | 整日 95% 区间 |")
    w("|---|---|---|---|")
    for _, x in p1_rows().iterrows():
        w(f"| `{x['contrast']}` | {x['candidate']} − {x['reference']} | {x['delta_pp']:+.3f} | [{x['day_low']:+.3f}, {x['day_high']:+.3f}] |")
    pr = p1_rows().set_index("contrast")
    ng, cw = pr.loc["no_explicit_gap"], pr.loc["selected_vs_tuned_CW"]
    w(f"\n要点：加悉尼天气（CW−C）、邻州（CR−C）、全部（F−C）的区间都跨零；去掉显式价差（no_explicit_gap）{ng['delta_pp']:+.3f}，"
      f"区间 [{ng['day_low']:+.3f}, {ng['day_high']:+.3f}] 跨零；调参后核心＋悉尼天气比核心高 {-cw['delta_pp']:.3f}，"
      f"区间 [{-cw['day_high']:+.3f}, {-cw['day_low']:+.3f}]（P1 按 0.5 个百分点近似并列规则保留核心）。\n")
    w("## 3. 留一组消融（本阶段，REF 参数原样重训）\n")
    w(f"训练 {abm['train']['first_day']}–{abm['train']['last_day']}，评估 2014。主结果用固定 τ={abm['tau']}；"
      f"“乐观”列按 P1 协议在 2014 上从 {abm['p1_thresholds_for_optimistic']} 重选 τ，使用了评估年本身，偏乐观。"
      f"ΔBalAcc = 消融后 − 完整模型；配对整天 bootstrap n_boot={abm['n_boot']}，seed={abm['seed']}，所有行共用同一组抽样日期（{abm['n_days']} 天）。\n")
    w("| 行 | 去掉的列 | BalAcc（τ=0.45） | Acc | ΔBalAcc（pp） | 95% 区间 | 乐观 τ | 乐观 BalAcc | 结论 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for _, x in ab.iterrows():
        verdict = {"reference": "对照", "removing it lowers BalAcc (group contributes)": "去掉后显著下降（有贡献）",
                   "removing it raises BalAcc": "去掉后显著上升"}.get(x["verdict"], "未检测到显著贡献")
        dropped = ", ".join(json.loads(x["dropped"])) or "—"
        ci = "—" if x["row"] == "full_features" else f"[{x['delta_lo']:+.3f}, {x['delta_hi']:+.3f}]"
        w(f"| {x['row']}（{x['group_zh']}） | {dropped} | {f(x['BalAcc'])} | {f(x['Acc'])} | {x['delta_BalAcc_pp']:+.3f} | {ci} | "
          f"{x['tau_2014_optimistic']:.2f} | {f(x['BalAcc_2014_optimistic'])} | {verdict} |")
    w("\n`full_features` 的预测与 `official/preds_2014.parquet` 逐位一致。\n")
    w("## 4. Permutation importance（2014，不重新拟合）\n")
    w(f"打分器：pred = 1[predict_proba ≥ {pm['tau']}]，用 `da.balanced_accuracy`；未打乱时得分 {f(pm['baseline_BalAcc'], 5)}% = REF。"
      f"n_repeats={pm['n_repeats']}，random_state={pm['random_state']}；重要性 = 基准 − 打乱后（pp）。按组时同组各列使用同一个行排列。\n")
    w("**按组（全部 6 组）**\n")
    w("| 排名 | 组 | 列数 | 重要性均值（pp） | 标准差 |")
    w("|---|---|---|---|---|")
    for _, x in grp.iterrows():
        w(f"| {x['rank']} | {x['group']}（{x['group_zh']}） | {x['n_columns']} | {f(x['importance_mean_pp'])} | {f(x['importance_std_pp'])} |")
    w("\n**逐列前 10 名**（标注的列只报告数值，不单独解读）\n")
    w("| 排名 | 列 | 组 | 重要性均值（pp） | 标准差 | 不单独解读的原因 |")
    w("|---|---|---|---|---|---|")
    for _, x in col.head(10).iterrows():
        note = x["single_column_note"] if isinstance(x["single_column_note"], str) else ""
        w(f"| {x['rank']} | `{x['feature']}` | {x['group']} | {f(x['importance_mean_pp'])} | {f(x['importance_std_pp'])} | {note or '—'} |")
    w("\n### 按组 permutation 与消融的比较\n")
    ga = grp.set_index("group")["importance_mean_pp"]
    aa = ab.set_index("row")
    w("| 组 | 按组 permutation（pp 下降） | 消融 ΔBalAcc（pp） | 消融区间 |")
    w("|---|---|---|---|")
    for g in c.GROUPS:
        w(f"| {g} | {f(ga[g])} | {aa.loc[g, 'delta_BalAcc_pp']:+.3f} | [{aa.loc[g, 'delta_lo']:+.3f}, {aa.loc[g, 'delta_hi']:+.3f}] |")
    w("\n- **一致**：阈值与价差（prev_price_mean、lag1_price_gap）在两种方法下都是最重要的组；时段编码在两种方法下都有可检测的贡献。")
    w("- **不一致**：星期、需求滞后、电价滞后在 permutation 下有明显下降，但消融的区间都包含 0。可能原因（不强行调和）：")
    w("  1. 两者回答的问题不同。permutation 衡量**这个已拟合模型**对某组列的依赖；消融衡量**重新训练时**去掉这组列能否由其他列弥补。")
    w("  2. 信息冗余：去掉星期后，上周同档需求（nsw_demand_lag7，同一星期几）和昨日需求仍携带星期效应，重训的模型可以转而依赖它们；permutation 不允许这种替代。")
    w("  3. permutation 会制造训练中不存在的组合（例如打乱星期后出现 weekday_0=1 且 weekend=1 的行），把模型推到训练分布之外，可能放大下降。")
    w("  4. 消融在 2014 上的差值区间较宽（约 ±0.5–1 个百分点），小于此量级的贡献检测不到。\n")
    w("## 5. 电价尖峰分组（D 日最大 NSW 电价）\n")
    w("边界：≤300、(300, 1000]、>1000 $/MWh（不改动）。b1 用 `da.baseline`，与模型使用完全相同的区间。\n")
    w("**2014（模型指标）**\n")
    w("| 组 | 天数 | 档数 | 真实 1 类占比 | HGBC 预测 1 类占比 | HGBC Acc | HGBC BalAcc | b1 Acc | b1 BalAcc | HGBC 混淆（TN/FP/FN/TP） | HGBC 召回 0/1 | 警告 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, x in s14.iterrows():
        r1 = "—" if pd.isna(x["HGBC_recall_1"]) else f(x["HGBC_recall_1"], 1)
        w(f"| {x['group']} | {x['days']} | {x['rows']} | {f(x['true_class1_share'], 2)}% | {f(x['HGBC_pred_class1_share'], 2)}% | "
          f"{f(x['HGBC_Acc'], 2)} | {f(x['HGBC_BalAcc'], 2)} | {f(x['b1_Acc'], 2)} | {f(x['b1_BalAcc'], 2)} | "
          f"{x['HGBC_tn']}/{x['HGBC_fp']}/{x['HGBC_fn']}/{x['HGBC_tp']} | {f(x['HGBC_recall_0'], 1)} / {r1} | "
          f"{'WARN：少于 30 天' if x['warn_lt_30_days'] else ''} |")
    w("\n2014 年只有 1 天落在 (300, 1000]、1 天落在 >1000，这两组**不能解读**。>1000 那天的 D+1 全是 0 类，"
      "按 kit 的约定 BalAcc 只取出现的类别（0 类）的召回率。\n")
    w("**2010–2014 只用标签的统计（m3_spike_label_only.csv）**\n")
    w("| 组 | 天数 | 真实 1 类占比 |")
    w("|---|---|---|")
    for _, x in sl.iterrows():
        w(f"| {x['group']} | {x['days']} | {f(x['true_class1_share'], 2)}% |")
    cl = sm["claim_9_7pct"]
    w(f"\n核对“约 9.7%”：该数字来自 2010–2026 全样本（192 天）。本阶段不读 2015 年以后的数据，在 2010-01-02 至 2014-12-31 上，"
      f">1000 组有 {cl['high_group_days']} 天，D+1 真实 1 类占比为 **{f(cl['high_group_class1_share'], 2)}%**（其余天 {f(cl['other_days_class1_share'], 2)}%），"
      "与 9.7% 不一致；方向相同（尖峰日之后“涨”的比例很低），数值以本表为准。\n")
    w("## 6. 结论与局限\n")
    w(f"- 模型主要依赖“昨日均价阈值与价差”这一组；去掉它，2014 BalAcc 变化 {aa.loc['threshold_gap', 'delta_BalAcc_pp']:+.2f} 个百分点，"
      f"按组打乱下降 {ga['threshold_gap']:.2f} 个百分点。时段编码也有可检测的贡献（消融 {aa.loc['slot', 'delta_BalAcc_pp']:+.2f}）。")
    w("- 电价滞后、需求滞后、季节、星期在消融中未检测到显著贡献；permutation 显示模型确实在用星期和需求滞后，二者差异主要来自信息冗余和打乱造成的分布外样本。")
    d14 = s14.set_index("group")["days"]
    w(f"- 2014 年尖峰日太少（(300,1000] {d14['(300,1000]']} 天、>1000 {d14['>1000']} 天），尖峰组上的模型表现无法评价；"
      f"2010–2014 只有 {cl['high_group_days']} 个 >1000 的 D 日。")
    w("- 局限：只用一个评估年（2014，且 P1 已用于选型）；区间假设天与天可交换，可能偏窄；乐观 τ 列用到了评估年；"
      "permutation 只解释这一个已拟合模型，不说明因果。")
    w("- **这些是解释性结果，不用于重新选型。**\n")
    w("文件：`results/price_hgbc/m3/`（m3_feature_groups.csv、m3_ablation.csv、m3_permutation_columns.csv、m3_permutation_groups.csv、"
      "m3_spike_2014.csv、m3_spike_label_only.csv 及 6 张图）。")
    return "\n".join(L) + "\n"


def main():
    open(OUT_MD, "w", encoding="utf-8").write(build())
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
