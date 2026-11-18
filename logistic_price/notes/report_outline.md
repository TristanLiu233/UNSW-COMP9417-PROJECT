# P4 report planning notes
Use this index to locate evidence while preparing the group report.

| Topic | Question to address | Evidence |
|---|---|---|
| Methodology | How is the label defined, what is known at forecast time, and why use Logistic Regression as the linear classifier? | protocol.md, features.py |
| Features | Where did the 22 inputs come from? What motivates the price transformations, gaps and interactions? | transforms.py, protocol.md |
| Validation | How did the folds, parameter search and 2014 selection work? Which alternatives scored lower? | search_2013.csv, validation_2014.csv, search_history.md |
| Results | Compare each period with both baselines and give the training cutoff for each model. | overall.csv, yearly.csv, monthly.csv |
| Degradation | How do monthly accuracy, both intervals and class shares vary? What limits their interpretation? | monthly_accuracy_intervals.png, yearly_class_share.png |
| Ablation | Cite the group baseline and distinguish the additional P4 comparisons. | selected_ablation_2014.csv |
| Live week | Describe training cutoff, model age, the historical days 37–43 window and remaining evidence. | live_week_framework.md |
| Limitations and integration | Check shared v1, group selection and responsibilities for drift and adaptation. | p4_schedule_check.md |

Before writing, confirm group model names, feature version, figure numbers, final P4 configuration and HGBC scores on the same evaluation intervals.
