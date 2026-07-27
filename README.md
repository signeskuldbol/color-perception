# color-perception

Analysis pipeline for the Color Crush app. The project consists of two scripts, run in order.

| Script | What it does |
|---|---|
| `create_excel_files_and_sort_data.py` | Reads the raw participant `.txt` files, cleans and filters them, writes the dataset sheets |
| `analyse_and_plot_data.py` | Reads those sheets, produces all plots and all statistical results |

## Requirements

Python 3.10 or newer.

```
pip install pandas numpy scipy matplotlib scikit-image openpyxl statsmodels
```

`statsmodels` is optional but strongly recommended: without it the
`min_detectable_effect` column is left empty, and that column is what
distinguishes "no effect" from "not enough participants to tell".

## Running the code

**1. Open `create_excel_files_and_sort_data.py`, search for `TODO`, and set
`input_folder` to the folder holding the raw participant `.txt` files.** Then run it.

**2. Open `analyse_and_plot_data.py` and check the settings block at the top**
before running:

- `PRIMARY_GROUP_VARIABLES`, `PRIMARY_SCOPES`, `PRIMARY_OUTCOMES` — the
  confirmatory comparisons. Everything else is treated as exploratory.
- `MIN_GROUP_SIZE` — smallest group allowed into a comparison. Marked `TODO`.
- `MAKE_PARTICIPANT_DIAGNOSTIC_PLOTS` — set `False` to skip the per-participant
  plots for a much faster run.

Then run it.

## Outputs

### `excel_files/combined_final_results.xlsx` — the cleaned dataset

| Sheet | Contents |
|---|---|
| `participant_summary` | One row per participant: demographics, levels completed, non-response rate |
| `all_levels_short` | One row per participant per level **attempt** (repeated attempts get their own row) |
| `all_final_colors_long` | One row per sublevel. The most complete sheet, and the one the analysis reads |
| `level_1` … `level_8` | One sheet per level, one row per attempt, sublevels spread across columns |
| `payload_qc` | Per-attempt structural checks on the raw log format |

### `excel_files/discarded_results.xlsx` — what was excluded and why

| Sheet | Contents |
|---|---|
| `empty_files` | Participant files with no usable log rows |
| `discarded_level_attempts` | Attempts removed by the filters, each with a `discard_reason` |

Note: the forced tutorial (`Color ID: 000000`) is dropped entirely and does not
appear in either workbook. The console reports how many tutorial attempts were
dropped.

### Statistical results — four workbooks, one per hypothesis family

| File | Question it answers |
|---|---|
| `Delta2000_analysis.xlsx` | Do groups differ in **average** accuracy? Also holds the non-response sensitivity check and age as a continuous predictor |
| `spread_analysis.xlsx` | Is one group more **variable / inconsistent** than another? |
| `axis_analysis.xlsx` | Is a group specifically imprecise on **one axis** (lightness vs chromatic), or biased toward one end of it? |
| `nonresponse_analysis.xlsx` | Does the rate of **skipped sublevels** differ between groups? |

Every results row carries the group means and SDs, the direction of the
difference, which test was chosen and why, and the minimum effect size that
comparison could have detected.

### `plots/`

| File or folder | Contents |
|---|---|
| `01_attempts_by_level.png` | Unique, repeated and missing attempts per level |
| `04_age_vs_performance_per_level.png` | Age against accuracy, one panel per level |
| `05_deltaE2000_by_level.png`, `05_magnitude_by_level.png` | Accuracy per level, both measures |
| `06_duration_by_level.png` | Time spent per level |
| `09_deltaE2000_by_axis*.png` | Error per manipulated axis, pooled and per level |
| `10_nonresponse_rate.png` | Skipped-sublevel rate by axis and by level |
| `group_overview/` | Participant counts per demographic group |
| `group_comparisons/` | Accuracy by demographic group, pooled and per level |
| `chosen_colors_heatmaps/` | Per participant: colours chosen in each sublevel |
| `participant_sublevel_plots/` | Per participant: error across every sublevel |
| `compass_spiderwebs/` | Per attempt: the eight logged compass magnitudes |
| `axis_radars/` | Per attempt: the same data on its six independent axes |

## Expected project structure

```
color-perception/
└── data_color_crush/
    ├── create_excel_files_and_sort_data.py
    ├── analyse_and_plot_data.py
    ├── "-Folder Name-"/
    │   ├── participant_file_1.txt
    │   └── ...
    ├── plots/
    │   └── ...
    └── excel_files/
        ├── combined_final_results.xlsx
        ├── discarded_results.xlsx
        ├── Delta2000_analysis.xlsx
        ├── spread_analysis.xlsx
        ├── axis_analysis.xlsx
        └── nonresponse_analysis.xlsx
```
