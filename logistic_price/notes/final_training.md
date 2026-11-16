# P4 扩展窗口训练与交接（2026-11-16）

## 已核对的规则与实现决策

- 本地课程规范 `specs/project_spec.pdf` 第 4–7 页：Stage 3 可用发布数据至 2026-10-31；重训只能用当时已知数据；live 周模型冻结，初始化与 predict 均不得训练。Stage 2 每模型/策略最多六次替换部署模型，初始训练不计。
- `handoff/selected_price_model.json`：已确认 log_interactions31、C=1、无 class_weight、threshold=0.42。共享 36 列中选 P4 的 22 列再变换成 31 列。
- 用户补充所指是形似 T 的弯钩符号；结合仓库 HGBC 的“nested tau selection”，按希腊字母 τ（tau，概率分类阈值）实现。它与 `protocol.md` 中的市场日 T 不同。
- 默认沿用 P4 `protocol.md` 的阈值网格 0.25–0.60、步长 0.01、最大化 BalAcc、并列优先接近 0.5。固定 C、权重、特征配方不再选择。
- 最终训练的内部窗口参考已合入的 `price_hgbc/src/price_hgbc/m6.py::retrain`：先训练至 cutoff 减一个日历年，再验证其后至 cutoff 的全部市场日；选 τ 后全部历史重拟合。这是本次显式采用的实现决策，课程规范只要求时间因果性，并未指定一年窗口。
- 对 cutoff=2026-10-31，内部训练至 2025-10-31，验证 2025-11-01–2026-10-31；最终重拟合至 2026-10-31。因此 2025–2026 不再是此最终选择的独立留出集。

## 接口

`logistic_price.final_training.fit_final(data, cutoff, validation_folds=None, thresholds=None, fixed_threshold=False)` 返回 `(bundle, audit)`；`train_final(data, cutoff, output, **kwargs)` 保存新工件目录。cutoff 是固定 AEST 的市场日日期，包含其结束时间戳（下一天 00:00）。

所有调用首先截断未来行，然后验证从 2010-01-01 开始的完整连续历史，计算共享特征，排除前 28 天预热。可训练起点恒为 2010-01-29，截止越晚训练窗口越大。C、class_weight、recipe 读取已确认配置。

默认按上述最后一年窗口嵌套选 τ。若需覆盖默认计划，必须同时传入 validation_folds（按顺序、不重叠的闭区间市场日对）和 thresholds。传 fixed_threshold=True 可单独复现原固定阈值 0.42，不允许同时传选择计划。每折用验证起点之前所有可用行单独拟合 scaler 和 Logistic；汇合所有折的验证概率，以 pooled BalAcc 选阈值，并列优先接近 0.5，再取较小阈值。随后重新初始化并用截止日以前全部样本拟合最终 scaler/模型。外部未来标签不参与选择。记录中的验证分数是选择分数，不是独立外层测试成绩。

示例选择计划（仅说明 JSON 结构，复用已有 2013 两折；不是已确认的 Stage 3 计划）：

```json
{"validation_folds": [["2013-01-01", "2013-06-30"], ["2013-07-01", "2013-12-31"]], "thresholds": [0.40, 0.42, 0.50]}
```

运行嵌套选 τ 后的最终重拟合：

```sh
python -m logistic_price.final_training --data /absolute/path/nem_weather_halfhourly_2010_2026.csv --cutoff 2026-10-31 --output /absolute/path/new-price-artifact
```

覆盖默认验证计划时，额外传 `--selection-plan /absolute/path/plan.json`；使用固定旧阈值则传 `--fixed-threshold`。CLI 只读取截止所需 CSV 前缀；输出已有则拒绝覆盖。`logistic.joblib` 直接供 `PriceClassifier(output)` 使用，`training.json` 记录截止、行数、折边界、候选分数、配置、种子和输入前缀摘要。旧冻结模型及结果不被替换。

此函数是 P4 价格训练组件，不负责部署调度、六次重训预算管理或需求模型。每个外层历史预测段必须使用该段开始前的 cutoff；用最终工件回测其训练期不能声称是样本外评估。组合 Forecaster 及完整 submission 检查由集成步骤承担。

## 验证

```sh
python -m unittest discover -s logistic_price/tests -v
COMP9417_DATA=/absolute/path/nem_weather_halfhourly_2010_2026.csv python -m unittest discover -s tests -v
python -m logistic_price.selfcheck
```

新增测试覆盖未来数据污染隔离、各折和最终 scaler 的样本数/均值、保存后的逐日与批量预测一致、完整市场日边界、非法验证范围和拒绝覆盖。

## 本次真实数据执行

截至 2026-10-31：内部拟合 276,240 行，验证 17,520 行，36 个候选选出 τ=0.49；最终拟合 293,760 行。内部选择 BalAcc=78.896947%，Accuracy=78.835616%，不能作为独立未来表现或 live 周成绩。最终工件单独输出，不改已提交的 Stage 1 配置或冻结结果。
