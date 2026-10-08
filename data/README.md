# Data directory

The scripts expect two cohort CSVs. They are **not** committed — put them here,
or one directory up next to this repo (the loader checks both).

| File | Grain | Required columns |
|---|---|---|
| `jm_long.csv` | one row per lab measurement | `id, marker, time_d, valuenum, log_value` |
| `jm_surv.csv` | one row per patient | `id, event_time_d, event, Liver_transplantation, Vasopressin, rrt, Age, gender_male` |

Conventions assumed by the code:

* `time_d` is **days since ICU admission** (t0 = ICU entry). The main landmark
  is Day 2.
* `event == 1` is in-hospital death; `event == 0` is censoring (discharge).
* `event_time_d` is time to death or censoring, whichever came first.
* `Liver_transplantation == 1` marks the competing event (104 patients).

Reference numbers for the cohort this repo was written against:

```
jm_long.csv        907,046 rows, 19 markers, 2,501 patients, time_d in [0, 30]
jm_surv.csv        2,508 patients, 962 events (38.4%)
Day-2 landmark     2,316 patients, 783 events (33.8%)   [transplant censored]
sampling density   median 11 distinct time points per patient within [0, 2] d
                   (P90 = 24, max = 50)
```

For other cohorts, override with `--long` / `--surv`.
