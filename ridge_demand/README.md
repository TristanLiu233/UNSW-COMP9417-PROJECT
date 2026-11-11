# Ridge 用电需求预测

本目录研究 NSW 次日 48 个半小时档位的用电需求预测，目标为 `nsw_demand`，单位 MW。全部实验使用课程发布的 CSV，模型采用 `StandardScaler` 和 `Ridge` 组成的 pipeline。

## 1 目录结构

```text
ridge_demand/
├── README.md
├── ridge_base.py                       # 基础版 Ridge，含数据读取、特征构造和评分函数
├── ridge_by_slot.py                    # 按 48 个半小时档位分别建模
├── ridge_hdd_cdd.py                    # 基础版 Ridge 加冷暖度差特征
├── ridge_by_slot_hdd_cdd.py             # 分时段 Ridge 加冷暖度差特征
├── compare_ridge_alpha1_results.py             # 四个固定 alpha=1 版本的汇总
├── tune_ridge_hdd_cdd.py                # 基础版 Ridge + HDD/CDD（18/18°C）的 alpha 搜索、重训和保存
├── tune_ridge_by_slot_hdd_cdd.py        # 分时段 Ridge + HDD/CDD（18/18°C）的 alpha 搜索、重训和保存
├── tune_ridge_hdd17_cdd195.py          # 两种结构在 17/19.5°C 下的调参和保存
├── evaluate_tuned_ridge.py              # 加载保存的模型并评估历史回放
└── outputs/
    ├── base/
    ├── by_slot/
    ├── hdd_cdd/
    ├── by_slot_hdd_cdd/
    ├── comparison_alpha1/
    ├── comparison_tuned/
    └── tune/
        ├── hdd_cdd/                    # 基础版 18/18°C，alpha=0.01 的模型、搜索表和回放结果
        ├── by_slot_hdd_cdd/            # 分时段版 18/18°C，alpha=10 的模型、搜索表和回放结果
        └── hdd17_cdd195/               # HDD 17°C / CDD 19.5°C 的实验
            ├── base/                   # 基础版 alpha=0.01 的模型和回放结果
            └── by_slot/                # 分时段版 alpha=10 的模型和回放结果
```

每个固定参数实验目录保存逐档预测、年度指标、月度指标及月度 MAPE 图，文件后缀与目录名对应。例如 `base/ridge_predictions_base.csv`。`tune/` 的各模型目录还保存 alpha 搜索表和 `.joblib` 模型，调参后回放文件的后缀带 `_tuned`。

## 2 实验设计

### 2.1 工作流程

1. 从小组消融结果接入 Ridge 的 25 个基础特征。
2. 固定 `alpha=1`，比较基础版、分时段版，以及两者分别加入 HDD/CDD（18/18°C）的版本。四个版本用截至 2014 年底的数据拟合并冻结，在相同的历史时段上回放。
3. 对 **HDD 18°C / CDD 18°C**，分别为基础版和分时段版搜索 `alpha`。
4. 对 **HDD 17°C / CDD 19.5°C**，用相同的候选网格为两种结构分别搜索 `alpha`。两组阈值的来源见第 7 节。
5. 两组阈值均用 2010–2013 年训练、2014 年验证来选择 `alpha`。选定后，用截至 2014 年底的数据重新拟合并保存四个模型。
6. 加载四个冻结模型，在 2015-01-01 至 2026-10-31 的同一批 207,456 个半小时档位上回放，比较两种模型结构和两组温度阈值的 MAPE 和 MAE。

### 2.2 四个模型变体

| 变体 | Ridge 数量 | 每个模型的输入列数 | 固定参数实验的 alpha |
|---|---:|---:|---:|
| `base` | 1 | 25 | 1 |
| `by_slot` | 48 | 25 | 1 |
| `hdd_cdd` | 1 | 27 | 1 |
| `by_slot_hdd_cdd` | 48 | 27 | 1 |

基础版模型使用所有档位的训练样本。分时段模型各自只使用对应档位在所有训练日的样本，各自拟合标准化器、系数和截距。预测时按目标时间戳计算档位，再使用对应模型，最后拼回按时间顺序排列的预测。

#### 2.2.1 基础 25 列

这些特征来自小组已有消融选型，按以下顺序构造。

| 类别 | 特征 |
|---|---|
| 档位、季节 | `slot_sin`, `slot_cos`, `season_sin`, `season_cos` |
| 星期 | `weekend`, `weekday_0` 至 `weekday_6` |
| NSW 同档历史 | `nsw_demand_lag1`, `nsw_price_lag1`, `nsw_demand_lag7`, `nsw_price_lag7` |
| NSW 昨日价格摘要 | `prev_price_mean`, `lag1_price_gap` |
| 邻州历史需求 | `qld_demand_lag1`, `qld_demand_lag7`, `vic_demand_lag7` |
| NSW 28 日同档统计 | `nsw_demand_mean28`, `nsw_demand_std28`, `nsw_price_mean28`, `nsw_price_std28` |

`lag1`、`lag7` 分别为昨日、上周同档。28 日统计只使用目标日之前 28 个完整市场日的同一档，标准差采用 `ddof=0`。`prev_price_mean` 是昨日 48 档原始 NSW 电价的均值，`lag1_price_gap` 是昨日同档电价减去该均值。季节编码根据目标年的 365 或 366 天计算。

#### 2.2.2 HDD/CDD 两列

令 T 为课程提供的目标半小时悉尼气温 `syd_temp`，初始实验以 BoM 度日地图提供的 18°C 作为取暖与制冷的共同对照基准（Bureau of Meteorology, n.d.）。新增实验采用 AEMO 附录 A2.1 表 7 的 NSW 取暖阈值 17°C 和制冷阈值 19.5°C（Australian Energy Market Operator, 2023, Table 7）。两份来源的度日方法均使用日尺度温度，本项目将阈值应用于每个半小时的悉尼气温。

```python
hdd = max(heating_base_c - T, 0)
cdd = max(T - cooling_base_c, 0)
```

这两列是连续的半小时冷暖度差，单位 °C，允许 Ridge 为冷、热两侧分别拟合斜率。两组温度阈值各自固定，在每组内搜索 alpha。课程允许使用预测日天气。预测日实际需求和电价只用于事后评分，市场输入特征均来自此前可见的历史。

### 2.3 时间切分与 alpha 搜索

| 用途 | 训练期 | 验证或评估期 |
|---|---|---|
| alpha 搜索 | 2010-01-29 至 2013-12-31，68,784 档 | 2014 年，17,520 档 |
| 冻结模型重训 | 2010-01-29 至 2014-12-31，86,304 档 | — |
| 历史回放 | 使用上述冻结模型 | 2015-01-01 至 2026-10-31，207,456 档、142 个月 |

前 28 天用于积累历史特征。默认回放训练截止到 2014 年底。四个固定参数脚本还支持 `--train-through 2013`，使用该选项时需要在交接结果中注明。

候选 alpha 为 `0.01, 0.1, 1, 10, 100, 1000`。每个候选值重新拟合训练期内的 pipeline，以 2014 年 MAPE 最低为选择规则。若相同，依次按 MAE 和 alpha 排序。分时段版让 48 个子模型共用一个候选 alpha，拼接全年预测后评分。本轮采用固定时间留出验证。

## 3 实验结果与交接建议

### 3.1 固定 alpha=1 的四版回放

| 变体 | 回放期 MAPE (%) | 回放期 MAE (MW) |
|---|---:|---:|
| `base` | 5.010 | 383.13 |
| `by_slot` | 5.253 | 405.43 |
| `hdd_cdd` | **4.849** | **365.78** |
| `by_slot_hdd_cdd` | 6.163 | 466.61 |

完整汇总见 [回放总表](outputs/comparison_alpha1/ridge_comparison_overall.csv)、[年度对比](outputs/comparison_alpha1/ridge_comparison_yearly.csv)和[月度对比图](outputs/comparison_alpha1/ridge_comparison_monthly_mape.png)。

### 3.2 调参后的四模型对比

| 结构 | HDD / CDD (°C) | alpha | 2014 验证 MAPE (%) | 2014 验证 MAE (MW) | 回放 MAPE (%) | 回放 MAE (MW) |
|---|---|---:|---:|---:|---:|---:|
| 基础版 | 18 / 18 | 0.01 | 3.2889 | 258.32 | 4.848894 | 365.78 |
| 分时段版 | 18 / 18 | 10 | 2.8757 | 227.71 | 6.129297 | 464.05 |
| 基础版 | 17 / 19.5 | 0.01 | 3.2622 | 256.31 | **4.820754** | **363.49** |
| 分时段版 | 17 / 19.5 | 10 | **2.8391** | **224.73** | 6.018830 | 455.08 |

#### 3.2.1 验证与回放结论

2014 年验证 MAPE 最低的是分时段版 17/19.5°C，完整历史回放中 MAPE 和 MAE 最低的是基础版 17/19.5°C。若小组以 2014 验证 MAPE 为选型依据，则交接分时段版作为验证首选，并把基础版记录为历史回放表现最佳的候选。需求任务的最终选型还需与组内其他需求模型和官方基线按同一口径比较。

#### 3.2.2 调参回放输出

[四模型全期对比表](outputs/comparison_tuned/ridge_tuned_comparison_overall.csv)和[四模型月度 MAPE 对比图](outputs/comparison_tuned/ridge_tuned_comparison_monthly_mape.png)位于 `outputs/comparison_tuned/`。每个模型的预测 CSV、年度指标 CSV、月度指标 CSV 和月度 MAPE 图保存在各自的模型目录。年度和月度指标按模型分别导出。

## 4 评分与图例

令 y 为实际需求，p 为预测需求，N 为评分档位数。

```text
MAPE (%) = mean(abs(p - y) / y) × 100
MAE (MW) = mean(abs(p - y))
```

MAPE 衡量平均相对偏差，MAE 衡量平均差多少 MW，两者均越低越好。所有半小时档位等权。由年度表计算全期指标时，用各年的 `n_slots` 加权，与用未舍入的逐档误差取平均等价。2026 年只覆盖 1–10 月，年度比较需注意覆盖月份。

单模型图的线是实际月度 MAPE。两种 95% 区间按照 specification Appendix A 计算。

- 半小时独立假设下使用 `MAPE ± 1.96 × s / sqrt(n)`，其中 s 是逐档百分比误差的样本标准差。
- 整日 bootstrap 每次有放回抽取与当月天数相同的完整市场日，重复 2,000 次，取 MAPE 分布的 2.5% 和 97.5% 分位数，随机种子为 17。因每日均为 48 档，代码重采样日均误差实现等价计算。

前者假设半小时独立。后者保留日内相关，但仍把不同日视为可交换抽样单位。区间描述类似当月条件下预期表现的不确定性。比较月份的区间是否重叠，本身不能代替对月份差值的检验。

## 5 复现命令

在仓库根目录运行，使用课程指定的 Python 3.11 和根目录 `requirements.txt`。本地数据放在 `data/nem_weather_halfhourly_2010_2026.csv`。

```bash
conda activate comp9417
python -m pip install -r requirements.txt

python ridge_demand/ridge_base.py
python ridge_demand/ridge_by_slot.py
python ridge_demand/ridge_hdd_cdd.py
python ridge_demand/ridge_by_slot_hdd_cdd.py
python ridge_demand/compare_ridge_alpha1_results.py

python ridge_demand/tune_ridge_hdd_cdd.py
python ridge_demand/tune_ridge_by_slot_hdd_cdd.py
python ridge_demand/tune_ridge_hdd17_cdd195.py
python ridge_demand/evaluate_tuned_ridge.py
```

各训练和评估脚本支持 `--data` 指定课程 CSV 路径。调参脚本应将模型与搜索表保存到 `outputs/tune/<variant>/`，评估脚本从同一路径读取。四版汇总脚本读取固定 alpha=1 的四个目录。调参后成绩从 `tune/` 的年度和月度 CSV 获取。

## 6 给组员的接口与对齐规则

### 6.1 用预测结果与其他算法比较

从所选模型目录取 `ridge_predictions_*_tuned.csv` 进行对比，并注明使用的模型版本。

| 字段 | 定义 |
|---|---|
| `timestamp` | 课程 CSV 的半小时间隔结束时间 |
| `predicted_demand_mw` | 对应档位的 NSW 预测需求，单位 MW |

组员可按 `timestamp` 与真实 `nsw_demand`、提升树和两条基线一一对齐，再统一评分。完整回放从 `2015-01-01 00:30` 到 `2026-11-01 00:00`。应先核对时间戳唯一且四组覆盖完全一致，避免静默丢弃不匹配行。

时间使用固定 AEST（UTC+10，无夏令时）。市场日按 `timestamp - 30 minutes` 的日期计算。目标日 T 的历史截止于时间戳 `T 00:00`。末档 `2026-11-01 00:00` 计入 2026 年 10 月。

### 6.2 加载保存的模型

从下表选择需要加载的模型。回放预测示例见 `evaluate_tuned_ridge.py`。

| 版本 | 保存模型路径（相对本目录） |
|---|---|
| 基础版 18/18°C | `outputs/tune/hdd_cdd/ridge_best_model_hdd_cdd.joblib` |
| 分时段版 18/18°C | `outputs/tune/by_slot_hdd_cdd/ridge_best_model_by_slot_hdd_cdd.joblib` |
| 基础版 17/19.5°C | `outputs/tune/hdd17_cdd195/base/ridge_best_model.joblib` |
| 分时段版 17/19.5°C | `outputs/tune/hdd17_cdd195/by_slot/ridge_best_model.joblib` |

保存对象包含 `variant`、`alpha`、温度阈值、`feature_columns`、`train_through` 和 `model`。18/18°C 版本使用 `base_temp_c`，17/19.5°C 版本分别使用 `heating_base_c` 和 `cooling_base_c`。基础版 `model` 是一套 pipeline。分时段版是以整数档位 0–47 为键的 pipeline 字典。

打包新一天的预测时，按相同公式从最近 28 个完整历史市场日及 `weather_next` 构造 48 行输入，按保存的列顺序调用模型。上述加载示例演示课程历史数据上的复用，最终课程提交还需要把需求模型和组内电价模型接到 `Forecaster.predict(history, weather_next)`，返回与 `weather_next.index` 一致的 `nsw_demand_mw` 和 `nsw_price_up` 两列。

交接时同时注明模型变体、alpha、基准温度、训练截止日和运行环境版本。预测阶段直接使用保存 pipeline 的标准化器。调参表记录的是重训前的 2014 验证成绩，保存模型则已经纳入 2014 年重训。


## 7 参考文献

Australian Energy Market Operator. (2023, August). *Forecasting approach – Electricity demand forecasting methodology*. <https://aemo.com.au/-/media/files/electricity/nem/planning_and_forecasting/nem_esoo/2023/forecasting-approach_electricity-demand-forecasting-methodology_final.pdf>

Bureau of Meteorology. (n.d.). *Annual and monthly heating and cooling degree days – Documentation*. <https://www.bom.gov.au/climate/map/heating-cooling-degree-days/documentation.shtml>
