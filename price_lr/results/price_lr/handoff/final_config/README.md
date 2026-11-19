# P5 → P1、P4：电价最终训练配置（2026-11-18）

只有配置和事实，不是报告正文。决定过程见 `results/price_lr/stage3/decision.md`。

## 配置

| 项 | 值 |
|---|---|
| 最终策略 | S1（expanding，Stage 3 第 11.4 条否决未触发） |
| 训练函数 | P4 `logistic_price.final_training.fit_final`（main `c23c959`），不改代码 |
| 截止日 | 2026-10-31（CSV 只读到 2026-11-01 00:00） |
| 训练窗口 | expanding_from_2010-01-29，最终拟合 293,760 行 |
| τ 选择 | 内部训练 2010-01-29 … 2025-10-31（276,240 行），验证 2025-11-01 … 2026-10-31（17,520 行），网格 0.25–0.60 步长 0.01 共 36 个，pooled BalAcc 最高 |
| τ | **0.49** |
| recipe | `log_interactions31`（31 列） |
| C / class_weight / 种子 | 1 / None / 17 |
| 拟合设置 | `threadpool_limits(1)`；ConvergenceWarning 视为错误 |
| live 周 | 模型冻结；`predict` 内部不训练；S3 的 CUSUM 状态不进 package |

## 复现命令（在小组仓库根目录）

```sh
python -m logistic_price.final_training \
  --data /absolute/path/nem_weather_halfhourly_2010_2026.csv \
  --cutoff 2026-10-31 \
  --output /absolute/path/NEW_DIRECTORY
```

输出 `logistic.joblib`（供 `PriceClassifier(output)` 直接加载）和 `training.json`。本机单次墙钟 2.74 秒（含解释器启动）；P4 记录 3.35 秒。整个 `train.py` 的两小时要求由 P1 在最终组合包里测。

## 工件核对

| 项 | 结果 |
|---|---|
| τ、行数、折边界、候选数、recipe、C、种子 | 全部和 P4 记录一致 |
| τ 处内部选择分数 | 78.896947 / 78.835616，和 P4 笔记一致（6 位小数；不是测试成绩） |
| 本机 `logistic.joblib` SHA-256 | `b44d5df38c9a49825f31913ec844997b40c65d53ce656c31274f758fbe318c12` |
| P4 记录的 SHA-256 | `7e898b2aebd146b83f40df869a9ffbc728bc447ca22970e872b51b103131dd50` |
| SHA 相同？ | **否**。本机重跑两次 SHA 相同；CLI 结果和进程内 `fit_final` 逐位一致；2026-10-18 … 10-31 共 672 档 `PriceClassifier.predict` 正常（一致性检查，不是成绩） |
| 待办 | P4 发来自己的 `logistic.joblib`、`training.json` 后，P5 运行 `python -m src.price_lr.final_check --reference <目录>`：要求 `training.json` 相同、系数差 ≤ 1e-12、672 档类别相同 |
| 提交用哪个工件 | **P4 的工件**；本机工件只用于核对 |

本机环境：Python 3.11.8，macOS-26.6.2-arm64-arm-64bit，scikit-learn 1.9.1，numpy 2.4.6，pandas 3.1.0，scipy 1.17.1，joblib 1.6.0，threadpoolctl 3.8.0。P4：Python 3.11.18，macOS 26.3.1，其余包版本相同。

`training_local_P5.json` 是本机生成的 `training.json`；`final_artifact_check.json` 是逐项核对记录。
