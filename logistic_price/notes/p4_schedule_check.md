# P4分工表逐项核对

核对日期：2026-11-11。范围仅为用户提供分工表中的P4职责；当天全组完成标准不全部属于P4个人任务。

| 日期 | P4事项 | 当前状态与证据 |
|---|---|---|
| 11-07 | Python3.11锁版本环境、starter、官方example检查 | 已验证所用核心依赖版本；starter六个原始代码单元顺序执行通过；官方example_package连续14天检查通过。这是课程示例，不是小组最终包。 |
| 11-08 | Pipeline、2014评估函数、概率阈值搜索 | 已完成：tuning.py、study.py和validation_2014.csv。 |
| 11-09 | 换到P1特征表v1 | **待P1确认。** 本模块独立实现与小组旧baseline配方对齐，未拿到明确标记的正式v1接口/缓存，不能当作已完成v1集成。 |
| 11-10 | class_weight、C网格、阈值搜索；2014结果行 | 已完成搜索和评估；可用于共享表的一行在handoff/p4_2014_shared_row.csv，尚未实际写入小组共享表。 |
| 11-11 | 消融、Logistic基本定稿 | 基础消融引用小组baseline；本目录保存新增模型的对比和补充消融，代码与候选模型已定稿供review。新增表达仍需按小组规则审查。 |
| 11-12 | 报告模板、统一图格式、类别比例随年份图 | 本地已备好notes/report_outline.md、notes/figure_conventions.md及handoff/yearly_class_share.png；全组统一采用需协调。 |
| 11-13 | live周分析框架：训练后37–43天的周误差 | 已提供notes/live_week_framework.md及一个实际冻结模型窗口的结果；只有一周，不假装拥有多次重训的独立样本。 |
| 11-14 | Logistic在2010–2014重拟合并冻结 | 已完成，六套工件及训练截止记录在results/frozen/artifacts/。全组selected model与公共results/stage1_config.json的冻结仍待集体决定。 |

## 不把这些状态混在一起

- **可以先push个人工作分支**，用于小组查看、比较和对齐；这不会自动代表已合并或验收通过。
- 目前不能宣称“已切换P1正式v1”。实际需要P1提供版本/接口/缓存，再按相同日期核对输入、顺序、目标和逐日结果。
- 新的31列配方属于候选表达。本地没有改P1公共features；若采用它需要按分工表走PR并通知全组。把工作推到个人分支不是合并批准。
- 共享表的P4结果行已准备，但未访问/写入真正的共享表。
- P1负责的最小双目标package、练习评测上传和公共Stage1配置，不由本次检查替代。已通过的官方example检查不证明团队package通过。

本地提供handoff/p4_stage1_candidate.json便于P1汇总，明确标注候选和接口待确认，不会覆盖全组公共配置。
