# Earlier searches on the original 22 features
Both searches chose hyperparameters on two chronological folds in 2013, then fitted on 2010–2013 and evaluated on 2014.

| Search | Candidates | Settings selected on 2013 | 2014 balanced accuracy |
|---|---:|---|---:|
| Standard | 30 | C=1, balanced weights, threshold 0.40 | 76.1096% |
| Expanded | 2,070 | C=10, weights {0:1, 1:1.5}, threshold 0.39 | 75.3640% |

The standard grid used C values 0.01, 0.1 and 1, two weight settings (None and balanced), and thresholds from 0.40 to 0.60 in steps of 0.05. The expanded grid used nine C values, five weight settings and thresholds from 0.20 to 0.65 in steps of 0.01. Its higher 2013 score did not carry through to 2014. Subsequent experiments explored alternative price features.

Each search directory under `results/tuning_checks/` (`standard/` and `expanded/`) contains `search.csv`, `ablation.csv` and `run.json`. The early runs used Python 3.13. The package omits their old binaries. All six frozen models in the main study were fitted in the course Python 3.11 environment.

Run the early searches in separate output directories.
```sh
python -m logistic_price.tuning --data data/nem_weather_halfhourly_2010_2026.csv --output logistic_price/_scratch/standard
python -m logistic_price.tuning --data data/nem_weather_halfhourly_2010_2026.csv --output logistic_price/_scratch/expanded --search-profile extended
```

Use `python -m logistic_price.study` for all four feature configurations and full replay. The reference parameters and scores are in `results/frozen/`.
