# 给 P2 的冻结需求逐日误差

[frozen_demand_daily.csv](frozen_demand_daily.csv) 可直接与日型表按 `market_day` 一对一连接。它覆盖 2015-01-01 至 2026-10-31，每个市场日一行，共 4,322 天、207,456 个半小时。

预测取自 P3 的需求提升树冻结回放。模型学习相对昨日同档的需求残差，预测时加回昨日需求；表中的误差按还原后的绝对需求（MW）计算。权重拟合到 2014-12-31，回放期重训次数为 0。P3 的 CUSUM 自适应策略和最终模型另有输出。本表采用半小时预测文件中标为 `frozen` 的列，该列逐档等于 `frozen_residual_b1`。[P3 冻结配置](https://github.com/TristanLiu233/UNSW-COMP9417-PROJECT/blob/4fab1bd2aa3978e0332c763a327d68d6b732cd9f/hgbr_demand/results/stage2/replay_protocol.json) · [P3 半小时预测](https://github.com/TristanLiu233/UNSW-COMP9417-PROJECT/blob/4fab1bd2aa3978e0332c763a327d68d6b732cd9f/hgbr_demand/results/stage2/half_hour_predictions.csv.gz)

| 列 | 含义 |
| --- | --- |
| `market_day` | 固定 AEST 市场日期。当天 00:30 至次日 00:00 结束的 48 档均属于当天。 |
| `n_slots` | 当日半小时数，每行均为 48。 |
| `frozen_mape_pct` | 先按每档计算 `100 × abs(预测 − 实际) / 实际`，再对当日 48 档取平均；数值 4.93 表示 4.93%。 |
| `frozen_mae_mw` | 当日 48 档绝对预测误差的算术平均，单位 MW。 |

2015-01-01 的 MAPE 为 4.929394%，MAE 为 373.389427 MW。所有日期合计的 MAPE 为 4.575284%，MAE 为 337.560166 MW。已将真实需求按时间戳逐档与课程 CSV 对齐；逐日 MAPE、142 个月的 MAPE 和 MAE、全期两项指标均与 P3 发布结果一致（数值校验容差为 10⁻¹⁰）。

## 与日型表连接

从[已核对的 P2 日型表](https://github.com/TristanLiu233/UNSW-COMP9417-PROJECT/blob/f5aeff7acc0e9019eba321c6c92ce689ef139588/day_type_analysis/outputs/day_type_clustering/day_type_assignments_loyo_k2_pca95.csv)下载 CSV，与本表放在同一目录后运行：

```python
import pandas as pd

errors = pd.read_csv("frozen_demand_daily.csv", parse_dates=["market_day"])
types = pd.read_csv("day_type_assignments_loyo_k2_pca95.csv", parse_dates=["market_day"])
matched = errors.merge(types[["market_day", "day_type"]], on="market_day",
                       how="left", validate="one_to_one")
assert len(matched) == 4322 and matched["day_type"].notna().all()
summary = matched.groupby([matched.market_day.dt.year.rename("year"), "day_type"]).agg(
    days=("market_day", "size"), mape_pct=("frozen_mape_pct", "mean"),
    mae_mw=("frozen_mae_mw", "mean"))
print(summary)
```

4,322 行均能匹配，Type 1 有 3,591 天，Type 2 有 731 天。P2 的日型由当天完整的实际需求曲线确定，适合在日终按类型解释误差。各日档数相等，跨日汇总时可直接平均每日 MAPE 和 MAE。

## 复现

使用课程 Python 3.11 环境，在仓库根目录运行以下命令。生成脚本使用版本固定的 P3 预测和 P2 日型文件，所需 Git 对象由下面的两条分支提供。传入课程发布的原始 CSV 路径即可复核并重新生成交接表。

```shell
git fetch origin hgbr-demand day_type_analysis
python handoff/p2/build_daily_errors.py --data "data/nem_weather_halfhourly_2010_2026.csv"
```

已有这两份来源的本地副本时，可传 `--p3-results <P3的stage2目录> --assignments <P2的日型CSV>`，脚本直接读文件，离线也可以重算。仅使用交接表做分析时，直接读取 CSV 即可。
