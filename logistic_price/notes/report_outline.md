# P4报告填充提纲

这是待小组填充的结构和证据索引，不是最终报告正文。

| 小组报告位置 | P4需填写/核对的问题 | 可引用的本地证据 |
|---|---|---|
| Methodology | 标签定义、预测日信息边界、Logistic为何适合做线性分类参照 | protocol.md、features.py |
| Features | 小组基础22列出处；价格变换/价差/交互的假设与公式 | transforms.py、protocol.md |
| Validation | 时间折、超参数选择、2014选型；哪些对照没有提高 | search_2013.csv、validation_2014.csv、search_history.md |
| Results | 2014与长期回放分别比较；两条基线；不可混淆训练年份 | overall.csv、yearly.csv、monthly.csv |
| Degradation | 各年月度Accuracy与两种区间；类别比例变化及解释限制 | monthly_accuracy_intervals.png、yearly_class_share.png |
| Ablation | 引用小组基础消融；说明本地只补充新配方对比及删减 | selected_ablation_2014.csv |
| Live week | 训练截止、模型年龄、37–43天历史窗口；不足与待补证据 | live_week_framework.md |
| Limitations / integration | v1接口待验收、全组算法选择、后续漂移/适应由谁承担 | p4_schedule_check.md |

写作前待填写：小组统一模型名称、选定特征版本、所用图号、P4最终交接配置、与HGBC的同口径对比结果。不要把尚未确认的项目写成已经完成。
