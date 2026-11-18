# NSW 用电需求日型分析

## 1. 分析目标

根据 NSW 每天48个半小时的需求值识别典型日型，以2010—2013年为固定参照，分析后续年份：

- 日型占比如何变化；
- 需求曲线是否越来越偏离早期典型形状；
- 同一日型内部的形状如何变化；
- 日型变化是否与冻结HGBR预测误差、输入漂移统计量及告警在时间上对应。

本分析属于事后描述性分析，不直接证明概念漂移或因果关系。

## 2. 运行方法

在项目根目录、课程指定的Python 3.11环境中运行：

```bash
python day_type_analysis/cluster_day_types.py
python day_type_analysis/analyze_day_type_changes.py
python day_type_analysis/analyze_day_type_errors.py
```

可选的PCA99%敏感性实验：

```bash
python day_type_analysis/cluster_day_types_pca99.py
```

聚类和曲线变化分析使用课程原始数据：

```text
data/nem_weather_halfhourly_2010_2026.csv
```

原始时间戳表示半小时区间结束时间。市场日通过时间戳减去30分钟确定，因此次日00:00的记录属于前一天最后一档。时间采用固定AEST，不使用夏令时。

## 3. 文件结构

以下路径相对于 `day_type_analysis/`：

| 文件或目录 | 用途 |
| --- | --- |
| `cluster_day_types.py` | 构造日曲线、前向验证选择k、拟合最终PCA和K-means |
| `cluster_day_types_pca99.py` | PCA99%敏感性实验 |
| `analyze_day_type_changes.py` | 年度中心距离及早期、近期曲线比较 |
| `analyze_day_type_errors.py` | 日型与冻结HGBR误差、漂移信号对照 |
| `outputs/day_type_clustering/` | 主分析评分表、标签、典型曲线和年度占比 |
| `outputs_forward_pca99/` | PCA99%前向验证实验结果 |
| `outputs/day_type_changes/centroid_distance/` | 年度中心距离统计与图 |
| `outputs/day_type_changes/profile_comparison/` | 2010—2013与2024—2025曲线比较 |
| `outputs/day_type_errors/` | 日型、误差、漂移合并表及对照图 |

正式分析使用带 `forward` 的文件名。旧 `loyo` 文件仅作为历史实验记录，不与新验证方法混用。

## 4. 分析方法

### 4.1 每天48维需求曲线

将每天48个半小时需求值排列成一行，每行代表一个完整市场日。

代码检查重复档位、缺失市场日、缺失档位和无效需求值。

### 4.2 按当日均值归一化

每一天的48个需求值分别除以当天平均需求。

- 1.00表示当天平均需求；
- 1.20表示比当天平均需求高20%。

这样主要比较日内形状，例如午间低谷、白天平台和傍晚高峰，而不是根据整体用电规模分组。

归一化移除了绝对需求规模，因此归一化曲线下降不能直接解释为绝对需求下降。标签表保留 `daily_mean_mw`，供单独分析需求水平。

### 4.3 PCA与前向验证选择k

主分析PCA累计解释方差阈值为95%，候选簇数为k=2—10。

选k只使用2010—2013年，采用三折前向验证：

| 折 | 拟合PCA和K-means | 验证年份 |
| --- | --- | --- |
| 1 | 2010 | 2011 |
| 2 | 2010—2011 | 2012 |
| 3 | 2010—2012 | 2013 |

每折PCA和K-means都只在该折训练数据上拟合。验证日先用训练PCA转换，再按最近的训练簇中心分配类别。

在验证集PCA空间计算轮廓系数。三折都能评分的候选按平均轮廓系数排名，同分选择较小k。

| 候选k | 三折平均轮廓系数 | 三折标准差 |
| --- | ---: | ---: |
| 2 | **0.4108** | 0.0132 |
| 5 | 0.3502 | 0.0233 |
| 3 | 0.3307 | 0.0595 |

k=2在三个验证年份均排名第一。

轮廓系数支持两类作为早期需求形状的粗略概括，但不能证明不存在有意义的细分类别。没有使用2014年以后的日型分配或预测误差来选择k。

K-means设置：

```python
random_state = 42
n_init = 20
```

### 4.4 固定训练期参照

选定k=2后，在完整2010—2013年共1,461天上拟合最终PCA和K-means。

主分析最终PCA保留5个主成分，解释训练期方差的96.24%。

2014年及以后只使用固定PCA和固定簇中心进行转换、分类和距离计算，不重新拟合。

主分析标签文件为：

```text
outputs/day_type_clustering/day_type_assignments_forward_k2_pca95.csv
```

主要字段：

| 字段 | 含义 |
| --- | --- |
| `market_day` | 市场日期 |
| `day_type` | 固定聚类分配的类别 |
| `daily_mean_mw` | 当日平均需求 |
| `distance_to_centroid` | PCA空间中到最近训练簇中心的欧氏距离 |

### 4.5 PCA99%敏感性检查

将累计解释方差阈值提高到99%，重复相同的前向验证。

- 仍选出k=2；
- 三折平均轮廓系数为0.3955；
- 三折标准差为0.0109；
- 最终保留9个主成分；
- 解释训练期方差的99.17%。

主分析仍采用95%设置。

将原留年验证改为前向验证后，95%主分析仍选k=2，最终标签和中心距离已与旧版核对一致。

### 4.6 日型变化的衡量

1. 年度日型占比：分母为当年实际观测天数。
2. 中心距离：按年、按日型统计中位数、P90及超过本类训练期P90的比例。
3. 分时期曲线：比较2010—2013年与2024—2025年同类曲线。
4. 预测误差：按年、按日型统计冻结HGBR的MAPE、MAE和样本数。
5. 漂移对照：对齐日型占比、中心距离、输入分数及告警日期。

训练期距离P90只是描述性参考线，不是经过在线误报率标定的漂移告警阈值。

曲线比较图的阴影是各档位跨日期的P10—P90范围，不是均值置信区间。

## 5. 日型分析结果

### 5.1 两类典型形状

| 日型 | 训练期典型形状 | 2010—2013年 | 2025年 | 2026年1—10月 |
| --- | --- | ---: | ---: | ---: |
| Type 1 | 早晨上升、午间回落、傍晚高峰突出 | 794天，54.3% | 360天，98.6% | 304天，100% |
| Type 2 | 白天需求相对较高，曲线较平坦 | 667天，45.7% | 5天，1.4% | 0天，0% |

后期绝大多数日期被分配到Type 1，但不代表这些日期与早期Type 1很相似。K-means总会分配最近的中心，即使所有训练中心都已经较远，因此必须同时分析中心距离。

### 5.2 同类内部形状变化

2018年，Type 1年度距离中位数首次超过其训练期P90，约0.3362。

2025年360个Type 1日期中，有344天，即95.6%，超过这条参考线。

比较2010—2013年与2024—2025年，Type 1：

- 12:00平均归一化需求从日均约1.03倍降至0.81倍；
- 18:00从约1.18倍升至1.27倍。

相对于每天自身平均需求，近期午间低谷更深、傍晚高峰更突出。这不直接说明绝对MW变化，也不能单凭曲线认定太阳能的因果作用。

2024—2025年只有20天属于Type 2，近期曲线解释需考虑样本量较小。

## 6. 与冻结HGBR误差对照

### 6.1 模型与评分口径

P3冻结HGBR拟合截至2014-12-31，回放期间不重训。

模型学习相对昨日同档的需求残差，预测时加回昨日需求。本分析评价还原后的绝对需求预测，使用半小时表中的 `frozen` 列。

```text
MAPE (%) = mean(abs(predicted - actual) / actual) × 100
MAE (MW) = mean(abs(predicted - actual))
```

回放覆盖2015-01-01至2026-10-31：

- 4,322个完整市场日；
- 207,456个半小时；
- 全期MAPE为4.5753%；
- 全期MAE为337.56 MW。

误差脚本从半小时预测重算每日MAE和MAPE，与每日监测表中的MAPE核对，绝对容差为1e-10。

### 6.2 主要发现

| Type 1指标 | 2015年 | 2025年 |
| --- | ---: | ---: |
| 占比 | 60.82% | 98.63% |
| 中心距离中位数 | 0.266 | 0.896 |
| 冻结HGBR MAPE | 2.225% | 7.167% |

后期日型分配集中到Type 1，同时同类中心距离和组内误差长期上升。因此，整体退化不能只用两类占比变化解释。

距离与误差并非每年同步变化。例如2017—2018年，Type 1中心距离增加，但MAPE下降。

当前结果是描述性时间对照，没有控制季节、天气及共同时间趋势，不据此推断因果关系或证明P(Y|X)变化。

### 6.3 年度误差贡献

```text
mape_contribution_pp = share_pct × mape_pct / 100
```

各日型贡献相加等于当年整体MAPE。由于每天均为48档，可以直接平均每日指标。

该字段是算术分解，不代表日型变化对误差的因果贡献。

零样本类别：

- 天数和占比为0；
- 误差与距离为空；
- 误差贡献为0。

2026年Type 2没有观测日，不能把它的MAPE解释为0%。

## 7. 与漂移检测对照

### 7.1 输入型检测器

输入检测器由P3实现，监测六个每日输入摘要：

- 昨日NSW平均需求；
- 昨日NSW需求形状；
- 昨日价格的有符号对数变换；
- 昨日VIC平均需求；
- 悉尼平均气温；
- 悉尼平均太阳辐射。

参考期为2010-01-29至2013-12-31，按日历月份计算参考均值和标准差。

统计量为六项标准化距离的RMS，再计算过去28个市场日的滚动均值。

阈值取2014年滚动分数的第95百分位数：

```text
threshold = 1.223969
```

输入告警只记录从阈值下方越到上方的日期。回放期共8条输入告警。

后期没有新告警不代表输入恢复正常：若分数持续高于阈值，就不会再次产生向上越界事件。因此必须结合分数曲线与阈值解释。

### 7.2 冻结模型误差CUSUM

使用每日MAPE的上侧CUSUM，配置来自P3。

```text
参考期：2014年样本外预测误差
参考平均MAPE：2.271159%
allowance k：0.787086个百分点
threshold h：29.724755
```

阈值通过2014年误差的7日移动块重采样标定，重复2,000次，取365日CUSUM最大值的第95百分位数。

回放期共有219条冻结误差CUSUM告警。报警后CUSUM重置，因此这些记录不是219个独立漂移事件，也不是219次重训。

本分析只使用：

```text
input_only
frozen_error_cusum
```

不混入自适应模型的 `adaptive_error_cusum` 告警。

## 8. P3输入文件与复现

### 8.1 输入文件

默认读取仓库内部的：

```text
day_type_analysis/inputs/p3/
```

该目录随实验代码包提供以下文件：

| 文件 | 用途 |
| --- | --- |
| `daily_monitors.csv` | 每日误差、CUSUM和输入漂移分数 |
| `drift_alarms.csv` | 告警日期、类型、分数和阈值 |
| `input_detector.json` | 输入检测器配置 |
| `cusum_config.json` | 需求误差CUSUM配置 |
| `half_hour_predictions.csv.gz` | 真实需求、冻结预测及其他策略预测 |

这些文件是P3实验产生的派生结果，不是新增的外部训练数据。

交接README记录的P3来源提交：

```text
4fab1bd2aa3978e0332c763a327d68d6b732cd9f
```

### 8.2 复现方法

先使用课程CSV生成日型标签，再运行后续分析：

```bash
python day_type_analysis/cluster_day_types.py
python day_type_analysis/analyze_day_type_changes.py
python day_type_analysis/analyze_day_type_errors.py
```

日型误差对照默认读取包内 `inputs/p3/`，无需访问Git或重新训练HGBR。

如需从课程CSV重新生成P3输入，应按P3模块说明运行Stage 1和Stage 2实验。随包输入支持复现本模块后处理；端到端复现还需要P3实验代码与课程CSV。

也可指定其他输入目录：

```bash
python day_type_analysis/analyze_day_type_errors.py --p3-dir /path/to/p3-results
```

### 8.3 后续脚本的版本设置

`analyze_day_type_changes.py`：

```python
MODEL_TAG = "forward_k2_pca95"
```

`analyze_day_type_errors.py`：

```python
DEFAULT_TYPES = (
    HERE / "outputs" / "day_type_clustering"
    / "day_type_assignments_forward_k2_pca95.csv"
)
```

## 9. 误差与漂移分析输出

以下文件位于 `outputs/day_type_errors/`：

| 文件 | 内容 |
| --- | --- |
| `daily_types_errors_drift.csv` | 逐日日型、距离、误差和漂移统计量 |
| `yearly_errors_by_type.csv` | 逐年分日型天数、占比、误差、距离和MAPE贡献 |
| `yearly_overall.csv` | 年度整体误差、距离、覆盖日期和输入分数 |
| `monthly_types_errors_drift.csv` | 月度统计、输入超阈值比例及告警数量 |
| `frozen_and_input_alerts.csv` | 输入型与冻结误差型告警 |
| `yearly_day_types_and_errors.png` | 年度日型占比、中心距离和误差对照 |
| `monthly_day_types_errors_drift.png` | 月度日型、距离、误差、输入分数与告警时间轴 |

年度误差图用外圈标记少于30天的分组。30只是样本量提示，不是统计显著性阈值。

## 10. 图表使用与解释限制

正文优先展示：

1. 2010—2013年两类典型需求曲线；
2. 逐年日型占比；
3. 中心距离与冻结HGBR误差的紧凑对照。

完整五层月度图用于分析及组内交接，正式报告应按版面选择面板，并遵循P4发布的图表规范。

解释时注意：

- 日型归一化后只反映形状，不反映绝对需求规模。
- K-means会把远离所有训练中心的日期也分配给最近中心。
- PCA空间的中心距离不能捕捉所有被舍弃方向的变化。
- 2026年仅覆盖1—10月，年度比较受季节覆盖影响。
- 后期Type 2样本很少，应同时报告天数。
- 日型来自当天完整真实需求，只用于事后分析；预测D+1时不能使用D+1真实日型。
- 时间接近和共同趋势不能证明因果关系。
- 输入变化或误差告警均不能单独证明P(Y|X)变化。
- 本分析没有替代P3负责的正式月度误差置信区间分析。

## 11. 相关研究

以下研究作为用电曲线聚类的背景材料，不是本项目实验结果的来源。正式引用前应核对书目信息与原文支持范围。

Afzalan等人（2021）使用两阶段聚类提取代表性家庭用电曲线。本项目采用PCA与K-means，并以2010—2013年建立固定参照。

Jessen等人（2022）研究半小时用电曲线的K-means聚类及异常模式。本项目使用到固定训练中心的距离描述后期需求形状偏离。

### 参考文献

Afzalan, M., Jazizadeh, F., & Eldardiry, H. (2021).
Two-stage clustering of household electricity load shapes for improved temporal pattern representation.
IEEE Access, 9, 151667–151680.

出版记录：
https://sanghani.cs.vt.edu/research/publications/2021/two-stage-clustering-of-household-electricity-load-shapes-for-improved-temporal-pattern-representation.html

Jessen, S. H., Ma, Z. G., Danang Wijaya, F., Vasquez, J., Guerrero, J., & Jørgensen, B. N. (2022).
Identification of natural disaster impacted electricity load profiles with k means clustering algorithm.
Energy Informatics, 5(Suppl. 4), Article 59.

https://doi.org/10.1186/s42162-022-00237-0