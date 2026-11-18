# P4 schedule as of 15 November 2026
This historical checklist covers main snapshot `5e8161224f248009d5ed577ba7fb09752acf1399`.

| November date | Task | Status at review |
|---|---|---|
| 7 | Python 3.11, starter notebook and course example | Verified in the course environment, along with the shared interface and P4 checks. |
| 8 | Pipeline, 2014 evaluation and threshold search | Complete. See tuning.py, study.py and results/frozen/. |
| 9 | Adopt P1 feature table v1 | Training and daily prediction use src.common.features. See v1_integration.md. |
| 10 | Search class weights, C and threshold | Complete. handoff/p4_2014_shared_row.csv was ready for the group table. |
| 11 | Ablation and candidate model | The baseline supplies the initial ablation. P4 completed the transformation comparison and additional ablations for review. |
| 12 | Report outline, figure conventions and annual class shares | Materials ready for group coordination. |
| 13 | Live-week analysis framework | Framework and one historical window covering days 37–43 available. Further windows depend on retraining records. |
| 14 | Refit through 2014 and freeze | Six saved models and replay results available. Group selection and the shared stage1_config.json were pending. |

At review, the remaining handoff tasks were to merge logistic_price/ into main, compare the two price classifiers, record the group selection, combine both tasks in Forecaster and check the submission. Drift, adaptation and report work followed the meeting assignments. Wangqi owned day-type analysis. The P4 changes were confined to logistic_price/.
