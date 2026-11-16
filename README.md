# Mortgage Credit Access and Pricing in 2024 (HMDA), Group 9

This is our final project for Big Data Analysis (instructor: Huan Vu, Faculty of DS&AI, National Economics University).
We analyse the 2024 HMDA One-Year loan-level file twice, once in Pandas and once in PySpark, check that both
versions give the same answers, and then compare how fast they run and how much memory they need.

The report is `report.pdf`. This README explains what is in the folder and how to get every number in the report
back from the raw data.

## 1. What is in the folder

```
README.md
requirements.txt
requirements-dev.txt           extra packages for the tests and the SQL check (pytest, duckdb)
pytest.ini
report.pdf                     the technical report (ACM large two-column format)
config/
  rules.yaml                   all filters, code mappings, bands, thresholds and formulas (version 1.1, frozen)
  expected.yaml                counts and results we verified; every run checks itself against them
src/
  convert_to_parquet.py        step 0: CSV to Parquet, every column kept as text, plus a row_id
  pandas_pipeline.py           step 1: the full analysis in Pandas
  spark_pipeline.py            step 2: the same analysis in PySpark
  check_equivalence.py         step 3: compares the outputs of steps 1 and 2
  perf_pipeline.py             step 4: the performance and scalability experiment
  build_report_assets.py       step 5: turns results/ into the numbers, tables and figures of the report
  independent_check.py         check: the main numbers again in DuckDB SQL, with every rule written out by hand
  compare_data_browser.py      check: record counts and loan-amount sums against the FFIEC Data Browser
  export_clean_core.py         keeps the clean core table as Parquet and compares it row by row with Pandas
  sensitivity_open_end.py      sensitivity run: adds back records whose open-end flag is exempt
  perf_cache_diagnosis.py      why the Spark cache was slower at large sizes (storage status, column-only cache)
  pred_subgroups.py            calibration of the out-of-fold predictions by loan type, purpose, tract and DTI band
  descriptive_extra.py         distributions of loan amount, income, interest rate, payment-to-income and loan costs,
                               computed in both Pandas and PySpark and compared
  py_to_ipynb.py               turns the .py files (cells marked with "# %%") into notebooks
  gen_synth.py                 makes a fake file with the HMDA schema, which we used to test the code first
  columns.txt                  the 99 field names in file order (used by gen_synth.py)
notebooks/                     the notebooks we ran on Kaggle on 15 November 2026, with all their outputs
                               (pandas, pyspark, performance, check_equivalence). Their code is the same as in src/.
                               HMDA_convert contains the code of step 0.
  extra/                       the four check notebooks run on 16 November 2026, with all their outputs (section 8)
tests/                         pytest suite (section 6)
results/
  pandas/, spark/              every result table of the two pipelines, including eq_*.csv and key_numbers.json
  equivalence/                 the Pandas vs PySpark comparison (summary and each compared value)
  perf/                        timings of every run, medians, peak memory, machine details
  external/                    counts and loan-amount sums from the official HMDA Data Browser, and our comparison with them
  independent/                 outputs of the DuckDB check, its comparison with Pandas, and loan and income medians
  sensitivity/                 the open-end exempt run compared with the main run
  predictive/                  calibration of the out-of-fold predictions by subgroup
  clean/                       description of the clean core table: rows per state, column types, row-by-row check
  descriptive/                 summary statistics, histograms, loan amount by income band (Pandas and PySpark agree)
figures/
  pandas/                      figures made by the Pandas notebook
  perf/                        the two performance figures, drawn from results/perf with the notebook's plotting code
report/                        LaTeX source of the report (compile main.tex with XeLaTeX);
                               report/figures/ and report/tables/ are generated from results/ in step 5
```

Code comments and notebook text are in Vietnamese; the report and this README are in English.

## 2. Data

- Source: the FFIEC/CFPB HMDA data publication, https://ffiec.cfpb.gov/data-publication/
- Dataset: the **2024 One-Year National Loan-Level Dataset** (Loan/Application Records, LAR), file
  `2024_public_lar_one_year_csv.csv`: 4.64 GB, 12,259,199 records, 99 fields.
  Download: https://files.ffiec.cfpb.gov/static-data/one-year/2024/2024_public_lar_one_year_csv.zip (unzip it to get the CSV).
- Lender table (2024 Transmittal Sheet):
  https://files.ffiec.cfpb.gov/static-data/one-year/2024/2024_public_ts_one_year_csv.zip
- Documentation: https://ffiec.cfpb.gov/documentation/

The data files are public and large, so we do not copy them here. The analysis uses no other data; the Data Browser
counts in `results/external/` are only there to check our reading of the file.

## 3. Installation

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
```

PySpark also needs Java 17 or 21. We produced the results on a Kaggle CPU session (4 CPUs, 30 GiB RAM, Python 3.13,
Linux). We did not test how small a machine can still run the full analysis.

## 4. How to run it

### Step 0: convert the CSV (once)

```
python src/convert_to_parquet.py 2024_public_lar_one_year_csv.csv data/hmda_2024_raw.parquet
```

We keep every column as text, so codes like `NA` and `Exempt` stay exactly as published, and we add a `row_id` column
with the position of each record in the CSV. Both pipelines start from this Parquet file.

### Steps 1 to 4 on your own machine

Put `hmda_2024_raw.parquet`, the Transmittal Sheet zip, the LAR CSV (only step 4 needs it), `config/rules.yaml` and
`config/expected.yaml` somewhere inside one input folder. Then run:

```
export HMDA_INPUT=/path/to/input      # the scripts search this folder for the files above
export HMDA_OUT=output                # results go to output/results/... and output/figures/...
export HMDA_WORK=/path/to/scratch     # temporary files (step 4 needs several GB)

python src/pandas_pipeline.py         # about 14 minutes on the Kaggle machine
python src/spark_pipeline.py          # about 22 minutes
python src/check_equivalence.py output/results/pandas output/results/spark output/results/equivalence
python src/perf_pipeline.py           # about 61 minutes
```

The performance script also reads a few optional settings: `PERF_SIZES` (for example `10%,25%,50%,100%`),
`PERF_RUNS`, `PERF_WARMUP`, `PERF_BUDGET_GB` and `PERF_SKIP_W0=1`.

### Steps 1 to 4 on Kaggle (this is how we produced the results)

1. Create two datasets: one with the two files of `config/` (`hmda-config`), and one with the original LAR CSV and
   Transmittal Sheet, imported from the FFIEC links (`hmda-2024-ffiec-files`).
2. Run `HMDA_convert` first; it writes the Parquet file. Then upload the other notebooks from `notebooks/`, add the two
   datasets and the output of `HMDA_convert` as input, set the accelerator to none and internet off, and Run All.
3. The equivalence notebook also needs the outputs of the Pandas and PySpark notebooks as input.

### Checks (run after steps 1 to 3)

```
pip install -r requirements-dev.txt
python src/independent_check.py data/hmda_2024_raw.parquet output/results/pandas output/results/independent
python src/compare_data_browser.py --fetch      # needs Internet; without --fetch it uses results/external/
python src/pred_subgroups.py output/results/pandas/pred_oof.parquet data/hmda_2024_raw.parquet output/results/predictive
python src/sensitivity_open_end.py             # a second Pandas run with exempt open-end flags kept
python src/perf_cache_diagnosis.py             # Spark only, about 11 minutes
python src/export_clean_core.py                # writes output/clean_core/ and output/results/clean/
python src/descriptive_extra.py                # needs the clean_core/ folder from the previous step
pytest                                         # fast tests, a few seconds
pytest -m slow                                 # runs the pipelines on a synthetic file, about 4 minutes
```

### Step 5: rebuild the report

```
HMDA_STYLE=acm python src/build_report_assets.py results report          # from the results in this folder
HMDA_STYLE=acm python src/build_report_assets.py output/results report   # or from your own run of steps 1 to 4
cd report && xelatex main && bibtex main && xelatex main && xelatex main
```

The script writes `numbers.tex`, the table bodies and the figures from `results/`. It stops with an error if our counts
differ from the Data Browser or if the two engines disagree. Run on this folder, it gives back exactly the
`report/numbers.tex` and `report/tables/` that are committed here.

## 5. Processing steps

1. **Read and profile.** For each of the 99 fields we count the special codes (`NA`, `Exempt`, `1111`, `8888`, `9999`)
   and the valid numbers, and we count exact duplicate rows.
2. **Core sample.** We keep applications (action taken 1 to 5) that are first lien, closed-end, not for business,
   not reverse mortgages, for site-built 1–4 unit homes and for the borrower's principal residence. This takes the
   12,259,199 records down to 5,775,325.
3. **Samples.** Decisioned (approved or denied): 4,514,505. Usable (valid DTI, CLTV and income): 4,100,051.
   Pricing (originated, with a valid rate spread): 3,616,549.
4. **Features.** DTI as a 19-level ordinal and in six bands, CLTV in five bands and loan-to-income in tertiles.
   Crossing them gives 90 risk cells. The expected denial probability of an application is the national denial rate
   of its risk cell, loan type and purpose.
5. **Analysis.** Outcomes by product (SQ1); the DTI jump test, the low-DTI check, the Kitagawa decomposition and
   pricing (SQ2); observed/expected ratios for states, MSAs and LMI tracts (SQ3); how much lenders differ, two-way
   lender and MSA effects, and market concentration (SQ4); and an optional denial model, cross-fitted in five folds
   grouped by lender.

**Prepared analytical data.** The PySpark pipeline saves the core sample as Parquet in `$HMDA_WORK/clean_core`. The
table holds the row identifier, nine raw fields as published and all derived fields with proper types (doubles,
booleans, text labels). It is partitioned by `state_code`: there are 55 values and most geographic summaries group by
state, whereas county, with more than 3,000 values, would create a large number of small files. The Pandas pipeline
builds the same table in memory. `src/export_clean_core.py` keeps the Spark-written table as a notebook output on
Kaggle (`clean_core/`, too large for this repository) and checks it against the Pandas table row by row and column by
column; `results/clean/` describes it (rows per state folder, column types and null counts, the comparison).

Both pipelines read every definition from `config/rules.yaml`, so they apply the same cleaning rules. The files in
`config/` are the exact copies we used on Kaggle.

### Where each part of the report comes from

| Report item | Result files (in `results/pandas/` unless another folder is given; the PySpark run writes the same names to `results/spark/`, see section 9) |
|---|---|
| Table 1, data quality | `dq_summary.csv`, `dq_special_codes.csv`, `dq_missing_by_action.csv`, `dq_field_profile.csv` |
| Table 2, the four sub-questions at a glance | the files of the rows below |
| Data Browser check (Section 2) | `results/external/data_browser_action_taken_2024.json`, `data_browser_slices_2024.csv`, `data_browser_comparison.csv` |
| Table 3 and Figure 1a, products | `sq1_outcomes_by_type.csv`, `sq1_loan_amount_originated_by_type.csv`, `sq2_rate_spread_by_type.csv`, `sq1_denial_type_x_purpose.csv` |
| Figure 1b-d and Table 14, distributions | `results/descriptive/summary_stats.csv`, `histograms.csv`, `loan_by_income_band.csv`, `loan_income_correlation.csv`, `pandas_vs_spark.csv` |
| Figure 2 and Table 12, DTI | `sq2_denial_by_dti_19.csv`, `sq2_dti_jump_table.csv`, `sq2_dti_jump_test.csv`, `sq2_dti_bunching.csv` |
| Low-DTI check | `sq2_low_dti_profile.csv`, `sq2_low_dti_reasons.csv`, `sq2_low_dti_by_type_purpose.csv` |
| Figure 3 and Table 15, decomposition; pricing | `sq2_kitagawa_denial.csv`, `sq3_lmi_kitagawa.csv`, `sq2_kitagawa_rate_spread.csv`, `sq2_risk_cells.csv` |
| Geography | `sq3_state_oe.csv`, `sq3_msa_oe.csv`, `sq3_lmi_oe.csv`, `sq3_state_separate_lines.csv` |
| Figure 4 (state map and funnel plots) and lender results | `sq4_lender_oe.csv`, `sq4_oe_by_agency.csv`, `sq4_two_way.csv`, `sq4_sensitivity.csv`, `sq4_spread_dispersion.csv`, `sq4_hhi_msa.csv` |
| Tables 4 and 13, Pandas vs PySpark | `results/equivalence/equivalence_summary.csv`, `equivalence_details.csv` |
| Table 5, the three checks | `results/equivalence/`, `results/independent/independent_vs_pandas_summary.csv`, `results/external/data_browser_comparison.csv` |
| Table 6, Figure 5, Table 11, performance | `results/perf/perf_summary.csv`, `perf_memory.csv`, `perf_runs.csv` |
| Table 17, cache diagnosis | `results/perf/perf_cache_diagnosis.csv` |
| Table 7, prediction, and the subgroup calibration | `pred_metrics.csv`, `pred_lender_oe_comparison.csv`, `pred_calibration_deciles.csv`, `results/predictive/pred_subgroups.csv`; the MLlib version is in `results/spark/key_numbers.json` and `pred_mllib_vs_sklearn_by_purpose.csv` |
| Table 16, open-end sensitivity | `results/sensitivity/open_end_exempt.csv`, `open_end_exempt_funnel.csv` |
| Table 18, clean core table | `results/clean/clean_core_summary.json`, `clean_core_manifest.csv`, `clean_core_columns.csv`, `clean_core_pandas_vs_spark.csv` |
| Dollar amounts in Section 10 | `results/independent/ind_magnitudes.json` |
| Single numbers quoted in the text | `key_numbers.json` |

## 6. Checking that Pandas and PySpark agree

`src/check_equivalence.py` compares the six required outputs (`eq_1` to `eq_6` in each results folder), the full field
profile, and every key number that both pipelines write to `key_numbers.json`. Counts have to match exactly. Means and
rates may differ by at most 1e-9 in relative terms, and results of the iterative two-way model by at most 1e-6
(relative difference = |a − b| / max(|a|, |b|); differences below 1e-15 count as zero).

For the files in this folder, all 1,684 values agree and the largest relative difference is 2.5e-12
(`results/equivalence/equivalence_summary.md`). The equivalence notebook on Kaggle reached the same result.

Agreement between two pipelines that read the same rules file does not show that the rules are applied correctly, so
there are three further checks:

- **A third implementation.** `src/independent_check.py` recomputes the main numbers in DuckDB SQL without reading
  `rules.yaml`: every filter, code mapping, band edge and the 30-case fallback is written out in the SQL. All 2,654
  values compared with Pandas agree (largest relative difference 7.7e-13;
  `results/independent/independent_vs_pandas_summary.csv`).
- **The FFIEC's own aggregates.** `src/compare_data_browser.py` compares record counts and summed loan amounts for
  every combination of action taken with loan type, purpose, lien status, construction method and state (544 slices)
  with the HMDA Data Browser. All agree exactly (`results/external/data_browser_comparison.csv`).
- **Tests.** `pytest` checks the rules file, recomputes headline numbers (tau, the DTI jump, the Kitagawa identity)
  from the saved tables, plants errors in one engine's output to make sure `check_equivalence.py` catches them, and,
  with `-m slow`, runs Pandas, PySpark and the SQL check end to end on a synthetic file.

## 7. Performance experiment

`src/perf_pipeline.py` takes a 15-column table of the core sample and samples it at 10, 25, 50 and 100%. It then
makes 2, 4 and 8 copies of it and keeps doubling (16, 32, 64 copies) until Pandas can no longer cope. The copies exist
only for this benchmark. On Kaggle, Pandas finished the 16-copy table (92.4 million rows, 24.2 GB) and was stopped
while loading the 32-copy table (184.8 million rows); Spark finished every size.

W0 reads the original CSV, with Pandas in three ways (the C and PyArrow parsers with every field as text, and PyArrow
with Arrow-backed types) and with Spark. Every engine and size runs in its own process. The parent process checks
memory every 0.2 s and stops a run once it reaches 90% of RAM. Each figure we report is the median of five runs after
one warm-up run. Spark session start, cache building, and cached and uncached runs are timed separately.

- `results/perf/perf_runs.csv`: every timed run
- `results/perf/perf_summary.csv`, `perf_table_medians.csv`: medians
- `results/perf/perf_memory.csv`: peak memory, and the step at which a run was stopped
- `results/perf/perf_meta.json`: hardware and software, the sizes, and the check that both engines agree at 100%
- `results/perf/perf_log_summary.md`: the same information as readable tables
- `results/perf/perf_cache_diagnosis.csv`: why cached Spark runs were slower from 4 copies up. The cached table always
  fitted in Spark's storage memory (2.6 GB of 12.1 GB at 16 copies, nothing on disk), and caching only the five
  columns W1 reads did not help, so the cost is the scan of Spark's in-memory format rather than memory pressure.

## 8. Where the results come from

The main results (`pandas/`, `spark/`, `perf/`, `equivalence/`) come from a single Kaggle run on 15 November 2026
(rules 1.1, code as in `src/`). `results/equivalence/` can be recomputed from `results/pandas/` and `results/spark/`
with `src/check_equivalence.py` and gives the same files as the equivalence notebook on Kaggle.

The checks were run on Kaggle on 16 November 2026 in five more notebooks (`notebooks/extra/`): `HMDA_extra_checks`
(DuckDB check, Data Browser slices, subgroup calibration), `HMDA_sensitivity_open_end`, `HMDA_cache_diagnosis`,
`HMDA_clean_core_export` and `HMDA_descriptive_distributions`. The executed notebooks, with every cell output, are in `notebooks/extra/`. Their small output
files were copied into `results/` and checked against the MD5 sums the notebooks printed.

## 9. What is not included

- The raw data (see section 2).
- Five large per-lender, per-county and per-MSA tables from the PySpark run (`sq4_lender_oe.csv`,
  `sq3_county_oe_display.csv`, `sq4_two_way_lender_effects.csv`, `sq4_two_way_msa_effects.csv`,
  `pred_lender_oe_comparison_spark.csv`, about 500 KB together). The report uses the Pandas versions in
  `results/pandas/`; the Spark side of these tables is compared through `key_numbers.json` and
  `eq_6_top50_lenders.csv`. The PySpark notebook writes them when it runs.
- From the checks: the clean core table itself (146 MB, a Kaggle notebook output, see section 5).
- Two per-record prediction files (`pred_oof.parquet`, about 85 MB, and its MLlib counterpart, about 50 MB); the
  pipelines create them again when run.
