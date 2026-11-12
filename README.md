# Mortgage Credit Access and Pricing in 2024 (HMDA), Group 9

Final project for Big Data Analysis (instructor: Huan Vu, Faculty of DS&AI, National Economics University).
The project analyses the 2024 HMDA One-Year loan-level file with two independent implementations, Pandas and PySpark,
checks that they return the same results, and compares their speed and memory use.

The report is `report.pdf`. This file explains what is in the folder and how to reproduce every number.

## 1. Folder structure

```
README.md
requirements.txt
report.pdf                     technical report (ACM large two-column format)
config/
  rules.yaml                   every filter, code mapping, band, threshold and formula (version 1.1, frozen)
  expected.yaml                verified counts and results that each run checks itself against
src/
  convert_to_parquet.py        step 0: CSV to Parquet, all columns as strings, adds row_id
  pandas_pipeline.py           step 1: full analysis in Pandas
  spark_pipeline.py            step 2: the same analysis in PySpark
  check_equivalence.py         step 3: compares the outputs of steps 1 and 2
  perf_pipeline.py             step 4: performance and scalability experiment
  build_report_assets.py       step 5: turns results/ into the report's numbers, tables and figures
  py_to_ipynb.py               converts the .py files (with "# %%" cell markers) to notebooks
  gen_synth.py                 synthetic file with the HMDA schema, used to test the code before the real run
  columns.txt                  the 99 field names in file order, read by gen_synth.py
notebooks/                     the four notebooks as executed on Kaggle (version 1), with all cell outputs;
                               their code cells are identical to the files in src/
results/
  pandas/, spark/              all result tables of the two pipelines, including eq_*.csv and key_numbers.json
  equivalence/                 comparison of the two (summary and every compared value)
  perf/                        timings of every run, medians, peak memory, environment
  external/                    counts from the official HMDA Data Browser, used only to verify our read of the file
figures/
  pandas/, perf/               figures written by the notebooks
report/                        LaTeX source of the report (compile main.tex with XeLaTeX);
                               report/figures/ and report/tables/ are generated from results/ by step 5
```

Comments and notebook text are in Vietnamese; the report and this file are in English.

## 2. Data

- Source: FFIEC/CFPB HMDA data publication, https://ffiec.cfpb.gov/data-publication/
- Dataset: **2024 One-Year National Loan-Level Dataset** (Loan/Application Records, LAR), file
  `2024_public_lar_one_year_csv.csv`, 4.64 GB, 12,259,199 records, 99 fields.
- Lender table: 2024 Transmittal Sheet,
  https://files.ffiec.cfpb.gov/static-data/one-year/2024/2024_public_ts_one_year_csv.zip
- Documentation: https://ffiec.cfpb.gov/documentation/

The data files are public and large, so they are not copied here. No other data enter the analysis.
The Data Browser counts in `results/external/` are used only as an external check.

## 3. Installation

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
```

PySpark also needs Java 17 or 21. The results in this folder were produced on a Kaggle CPU session
(4 CPUs, 30 GiB RAM, Python 3.13, Linux); we have not measured the smallest machine the full run fits on.

## 4. How to run

### Step 0: convert the CSV once

```
python src/convert_to_parquet.py 2024_public_lar_one_year_csv.csv data/hmda_2024_raw.parquet
```

Every column is stored as a string so that codes such as `NA` and `Exempt` are kept as published, and a `row_id`
column records the position of each record in the CSV. Both pipelines start from this file.

### Steps 1 to 4 on a local machine

Put `hmda_2024_raw.parquet`, the Transmittal Sheet zip, the LAR CSV (needed only by step 4), `config/rules.yaml` and
`config/expected.yaml` anywhere under one input folder, then:

```
export HMDA_INPUT=/path/to/input      # searched recursively for the files above
export HMDA_OUT=output                # results go to output/results/... and output/figures/...
export HMDA_WORK=/path/to/scratch     # temporary files (several GB for step 4)

python src/pandas_pipeline.py         # about 14 minutes on the Kaggle machine
python src/spark_pipeline.py          # about 26 minutes
python src/check_equivalence.py output/results/pandas output/results/spark output/results/equivalence
python src/perf_pipeline.py           # about 70 minutes
```

Optional switches of the performance script: `PERF_SIZES` (for example `10%,25%,50%,100%`), `PERF_RUNS`,
`PERF_WARMUP`, `PERF_BUDGET_GB`, `PERF_SKIP_W0=1`.

### Steps 1 to 4 on Kaggle (how the results here were produced)

1. Create three datasets: the Parquet file (`hmda-2024-one-year-raw`), the two files of `config/` (`hmda-config`),
   and the original LAR CSV and Transmittal Sheet imported from the FFIEC links (`hmda-2024-ffiec-files`).
2. Upload the four notebooks of `notebooks/`, add the three datasets as input, accelerator none, internet off, Run All.
3. For the equivalence notebook, add the outputs of the Pandas and PySpark notebooks as input.

### Step 5: rebuild the report

```
HMDA_STYLE=acm python src/build_report_assets.py results report
cd report && xelatex main && bibtex main && xelatex main && xelatex main
```

The script writes `numbers.tex`, the table bodies and the figures from `results/`, and stops if the Data Browser
counts or the two engines disagree. Running it on this folder reproduces `report/numbers.tex` and `report/tables/` exactly.

## 5. Processing steps

1. **Read and profile.** All 99 fields are profiled for each special code (`NA`, `Exempt`, `1111`, `8888`, `9999`) and
   for numeric validity; exact duplicates are counted.
2. **Core sample.** Applications (action taken 1 to 5), first lien, closed-end, not for business, not reverse,
   site-built 1-4 units, principal residence: 12,259,199 records become 5,775,325.
3. **Samples.** Decisioned (approved or denied): 4,514,505. Usable (valid DTI, CLTV and income): 4,100,051.
   Pricing (originated, valid rate spread): 3,616,549.
4. **Features.** DTI as a 19-level ordinal and six bands, CLTV in five bands, loan-to-income tertiles; their product
   gives 90 risk cells. The expected denial probability of an application is the national rate of its risk cell,
   loan type and purpose.
5. **Analysis.** Outcomes by product (SQ1); DTI jump test, low-DTI diagnostic, Kitagawa decomposition, pricing (SQ2);
   observed/expected ratios for states, MSAs and LMI tracts (SQ3); lender dispersion, two-way lender and MSA effects,
   concentration (SQ4); optional denial model with lender-grouped five-fold cross-fitting.

All definitions are in `config/rules.yaml`, which both pipelines read, so the cleaning rules are identical by construction.
The copy here differs from the one used in the Kaggle run only in two comment lines of its change log; its parsed
content is identical.

### Where each part of the report comes from

| Report item | Result files (in `results/pandas/`, with the same names in `results/spark/`, unless a folder is given) |
|---|---|
| Table 1, data quality | `dq_summary.csv`, `dq_special_codes.csv`, `dq_missing_by_action.csv`, `dq_field_profile.csv` |
| Table 2 and Figure 1, products | `sq1_outcomes_by_type.csv`, `sq1_loan_amount_originated_by_type.csv`, `sq2_rate_spread_by_type.csv`, `sq1_denial_type_x_purpose.csv` |
| Figure 2 and Table 3, DTI | `sq2_denial_by_dti_19.csv`, `sq2_dti_jump_table.csv`, `sq2_dti_jump_test.csv`, `sq2_dti_bunching.csv` |
| Low-DTI diagnostic | `sq2_low_dti_profile.csv`, `sq2_low_dti_reasons.csv`, `sq2_low_dti_by_type_purpose.csv` |
| Table 4, decomposition; pricing | `sq2_kitagawa_denial.csv`, `sq3_lmi_kitagawa.csv`, `sq2_kitagawa_rate_spread.csv`, `sq2_risk_cells.csv` |
| Geography | `sq3_state_oe.csv`, `sq3_msa_oe.csv`, `sq3_lmi_oe.csv`, `sq3_state_separate_lines.csv` |
| Figure 3 and lender results | `sq4_lender_oe.csv`, `sq4_oe_by_agency.csv`, `sq4_two_way.csv`, `sq4_sensitivity.csv`, `sq4_spread_dispersion.csv`, `sq4_hhi_msa.csv` |
| Tables 5 and 6, correctness | `results/equivalence/equivalence_summary.csv`, `equivalence_details.csv` |
| Table 7, Figure 4, Table 12, performance | `results/perf/perf_summary.csv`, `perf_memory.csv`, `perf_runs.csv` |
| Table 8, prediction | `results/pandas/pred_metrics.csv`, `pred_lender_oe_comparison.csv`, `pred_calibration_deciles.csv`; MLlib version in `results/spark/pred_lender_oe_comparison_spark.csv` and `key_numbers.json` |
| Single numbers quoted in the text | `key_numbers.json` |

## 6. Checking that Pandas and PySpark agree

`src/check_equivalence.py` compares six required outputs (`eq_1` to `eq_6` in each results folder), the full field
profile and every key number both pipelines write to `key_numbers.json`. Counts must match exactly, means and rates
within 1e-9 and results of the iterative two-way model within 1e-6 (the difference is divided by the larger magnitude
when that exceeds one, and taken as is below one). Result of the run in this
folder: 1,682 values agree, largest relative difference 2.7e-12 (`results/equivalence/equivalence_summary.md`).

## 7. Performance experiment

`src/perf_pipeline.py` builds a 15-column table of the core sample, samples it at 10, 25, 50 and 100% and replicates
it 2, 4, 8 and 16 times (replicas are used only for this benchmark). Each engine and size runs in its own process;
the parent process samples memory every 0.2 s and stops a run at 90% of RAM. Each measure is the median of five runs
after one warm-up. Spark session start, cache building, cached and uncached runs are timed separately.

- `results/perf/perf_runs.csv`: every timed run
- `results/perf/perf_summary.csv`, `perf_table_medians.csv`: medians
- `results/perf/perf_memory.csv`: peak memory and the stage where a run was stopped
- `results/perf/perf_meta.json`: hardware and software

## 8. What is not included

- The raw data (see section 2).
- Two per-record prediction files (`pred_oof.parquet`, about 85 MB, and its MLlib counterpart, about 50 MB); the
  pipelines write them again when run.
