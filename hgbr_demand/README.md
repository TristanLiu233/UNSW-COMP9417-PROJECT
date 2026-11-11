# COMP9417 小组项目 · P3 NSW 用电需求预测

这个交接包包含需求模型的实验代码、结果表、图、训练好的模型和配置。`p3_hgbr.demand.DemandForecaster` 输出每半小时的 NSW 用电需求，单位为 MW。

## 文件和结果

| 内容 | 对应文件 |
|---|---|
| 29 个输入特征、市场日对齐与预测接口 | `p3_hgbr/demand.py`。特征顺序见 `results/stage1/p3_stage1_candidate.json`。 |
| 2014 年验证，两个预测目标各试 36 组参数 | `results/stage1/tuning_2014.csv`、`validation_2014.csv`、`validation_predictions.csv.gz` |
| 小组共享表所需的 P3 结果行 | `results/stage1/p3_shared_row.csv` |
| 按星期和 NSW 公共假日分析误差 | `results/stage1/validation_weekday_holiday.csv`、`figures/validation_calendar.png` |
| 特征敏感度 | `results/stage1/validation_feature_importance.csv`。每列分别打乱五次，特征间的相关性会影响结果。 |
| 需求 CUSUM 标定和冻结模型 | `results/stage1/cusum_config.json`、`cusum_reference_daily_2014.csv`、`frozen/demand_hgbr.joblib` |
| 历史回放与两条官方基线 | `results/stage2/half_hour_predictions.csv.gz`、`monthly.csv`、`yearly.csv` |
| 漂移监测和重训策略 | `results/stage2/drift_alarms.csv`、`daily_monitors.csv`、`input_detector.json`、`retraining_log.csv`、`strategy_comparison.csv` |
| 与 P2 Ridge 的月度对比 | `results/stage2/monthly_comparison_with_p2_ridge.csv`。对比按月份和样本数核对，P2 原始汇总表见 `references/`。 |
| 最终需求模型和训练配置 | `results/stage2/final_artifacts/demand_hgbr.joblib`、`p3_final_demand_config.json` |
| 12 月 7–13 日评分周的历史类比 | `results/stage2/live_week_demand_analogs.csv`、`live_week_demand_analog_summary.json` |
| 接口和复现检查 | `results/verification/` |

## 实验设置

- **市场日和特征。** 课程 CSV 的时间戳表示半小时区间的结束时间。市场日按固定 AEST（UTC+10）下 `timestamp - 30 min` 的日期划分。特征只使用历史市场数据、目标日的课程天气和日历。近 7 天的同档统计先移一天再计算。2010 年头 28 天用于积累历史数据。
- **第一阶段。** 用 2010–2013 年训练、2014 年验证。29 列输入沿用小组消融的 A2 配置，仓库中的说明与群附件是同一个 Git blob。组内 seed 17 的参考成绩为 MAPE 2.435525%、MAE 192.4968 MW。本地用相同数据前缀和课程指定的依赖版本复跑，得到 MAPE 2.438196%、MAE 192.6987 MW。详见 `results/stage1/group_reference_check.json`。P3 的两种训练目标均用 seed 0、平方误差损失和 `early_stopping=False`。按 2014 年 MAPE 选型，MAPE 相同则先看 MAE。残差模型学习实际需求减去昨日同档需求（`b1`），评分前加回 `b1`。[小组消融说明](https://github.com/TristanLiu233/UNSW-COMP9417-PROJECT/blob/baseline/ablation/%E6%B6%88%E8%9E%8D%E8%AF%B4%E6%98%8E.md)
- **第二阶段。** 用 2010–2014 年重训后冻结模型，回放 2015-01-01 至 2026-10-31，共 4,322 个市场日。MAE 的单位为 MW，MAPE 以百分数表示。预测表的 `frozen` 列是入选冻结模型的预测，`frozen_level`、`frozen_residual_b1` 和 `frozen_group_reference` 列分别保存各个模型的预测。
- **月度 MAPE 的 95% 区间。** 一种方法把半小时误差视为独立，用正态近似计算区间。另一种方法对完整市场日重采样 2,000 次，保留日内相关性，并把各市场日作为可交换的抽样单位。
- **误差 CUSUM。** 用只在 2010–2013 年拟合的模型预测 2014 年，计算每日 MAPE 作为参考。每天结束更新 `S = max(0, S + 当日MAPE - 参考均值 - k)`，其中 `k` 为参考日误差样本标准差的一半。用 7 日移动块模拟 2,000 个 365 日序列，以各序列 CUSUM 最大值的第 95 百分位数作为阈值。每次越阈后清零。该策略的两次重训至少间隔 365 天，最多重训 6 次。
- **输入监测。** 对 6 个预测时可获得的日尺度输入，使用 2010–2013 年对应月份的均值和标准差做标准化。阈值取 2014 年滚动 28 天平均距离的第 95 百分位数。日志记录向上跨阈日期以及每次报警前后 30 天的误差，供事后对照。
- **重训比较。** 回放对比冻结、定期扩展训练窗口、定期只用最近 4 年、CUSUM 触发后扩展训练窗口四种策略。定期策略从 2017-01-01 开始，每两年重训一次。每个策略最多重训 6 次。第 D 天结束时触发的重训最早用于 D+1，训练截止和生效日期见重训日志。
- **第三阶段。** 用 2015–2024 年的结果排名，2025 年至 2026 年 10 月留作事后检验。CUSUM 触发的扩展窗口策略在排名期表现最好，定期扩展窗口策略在留出期的 MAPE 更低。两种扩展窗口策略均可用截至 2026-10-31 的已发布历史数据一次性拟合最终模型。最终需求模型按该截止日训练，并在评分周冻结。11 个历史类比周用于估计需求预测误差的经验分布。
- **历史事件和天气。** `event_context_descriptive_only.csv` 引用 AEMO 的事件日期用于事后解释，训练、特征、调参和报警标定均未使用这些日期。实验使用课程提供的目标日再分析天气，实际部署中的天气预报误差未计入。
- **组内交接。** P1 的共享特征缓存仍待确认。本实现采用已核对的小组 A2 特征约定与课程 CSV。Ridge 对比采用 P2 发布的汇总结果。[P2 Ridge 说明和结果](https://github.com/TristanLiu233/UNSW-COMP9417-PROJECT/blob/ridge_demand/ridge_demand/README.md)

## 本地复现

在此目录使用课程指定的 **Python 3.11** 环境，并安装 `requirements-p3.txt`。把 `course_kit` 改为解压后的课程套件路径，套件中的 CSV 是运行所需的数据。命令会将结果写入 `reproduced/`。

```sh
course_kit="/path/to/comp9417_project_kit"
data_file="$course_kit/data/nem_weather_halfhourly_2010_2026.csv"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pip install -r requirements-p3.txt
python -m p3_hgbr.stage1 --data "$data_file" --out reproduced/stage1 --jobs 4
python -m p3_hgbr.stage2 --data "$data_file" --stage1 reproduced/stage1 --out reproduced/stage2 --ridge-dir references
python -m p3_hgbr.figures --stage1 reproduced/stage1 --stage2 reproduced/stage2 --out reproduced/figures
python -m p3_hgbr.event_context reproduced/stage2
python -m p3_hgbr.verify --kit "$course_kit" --data "$data_file" --results reproduced
python -m p3_hgbr.smoke --kit "$course_kit" --data "$data_file" --stage2 reproduced/stage2 --work reproduced/smoke
```

仅重建最终需求模型时，运行以下命令。

```sh
python -m p3_hgbr.train_demand "$data_file" \
  --config results/stage2/p3_final_demand_config.json \
  --artifact-dir reproduced/final_artifacts
```

## 接入小组预测接口

```python
from p3_hgbr.demand import DemandForecaster

demand = DemandForecaster(artifact_dir)
output_demand = demand.predict(history, weather_next)
# 返回 48 行，索引与 weather_next 一致，唯一列为 nsw_demand_mw。
```

将 `p3_hgbr/__init__.py`、`p3_hgbr/demand.py`、`p3_hgbr/train_demand.py` 和选定的配置复制到小组包，把 `demand_hgbr.joblib` 放进 `artifacts/`。在小组的 `model.py` 中合并需求预测与所选电价分类器的 `nsw_price_up`，然后对完整小组包运行官方检查器。

P3 本地接口检查用昨日电价基线补齐价格列。用于检查的 14 天在最终需求模型的训练期间内。样本外成绩见历史回放表。
