# 保留没有带来提升的搜索

在引入新的电价表达之前，先尝试过两轮原始22列Logistic搜索。两轮都只在2013两个时间折选参数，再以2010–2013训练、2014验证。这些记录用于解释为什么后续工作转向特征表达。

| 搜索 | 候选数 | 按2013选出的参数 | 2014 BalAcc |
|---|---:|---|---:|
| standard | 30 | C=1，balanced，阈值0.40 | 76.1096% |
| expanded | 2,070 | C=10，{0:1,1:1.5}，阈值0.39 | 75.3640% |

标准搜索使用C={0.01,0.1,1}、None/balanced两种权重、0.40–0.60步长0.05的阈值。扩展搜索使用9档C、5种权重以及0.20–0.65步长0.01的阈值。扩展配置2013分数更高，但2014反而下降，不能因网格更大就认定模型更好。

记录保存在 `results/tuning_checks/standard/` 和 `expanded/`，各含 `search.csv`、`ablation.csv`、`run.json`。这些早期结果来自Python3.13环境，未在分支包中重复放入旧模型工件；本轮正式研究的六套冻结模型均已在课程Python3.11环境重新训练。

如需复查早期搜索，使用独立输出目录：

```sh
python -m logistic_price.tuning --data data/nem_weather_halfhourly_2010_2026.csv --output logistic_price/_scratch/standard
python -m logistic_price.tuning --data data/nem_weather_halfhourly_2010_2026.csv --output logistic_price/_scratch/expanded --search-profile extended
```

这两条命令仅运行早期2014实验。四种输入表达和完整历史回放由 `python -m logistic_price.study` 负责。当前采用的正式参数与结果以 `results/frozen/` 为准。
