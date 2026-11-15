# HMDA performance experiment, Kaggle run (2026-11-15)

Môi trường: 4 CPU, RAM 32.2 GB, ngân sách 29.0 GB; pandas 2.3.3, pyarrow 23.0.1, pyspark 4.0.4, Python 3.13.18, rules 1.1.
Thời gian chạy notebook: 3646 s. W0 đọc CSV gốc FFIEC (4.64 GB, 99 cột); Parquet thô 0.61 GB.
Năm file trong thư mục này được chép từ output Kaggle và kiểm tra md5 khớp với md5 in trong log Kaggle.

## RAM cao nhất (GB) và trạng thái

| label | status | peak_rss_gb | seconds | stage | rows |
|---|---|---|---|---|---|
| pandas_W0_pyarrow | oom_budget | 29.012 | 166.3 | W0 pyarrow run 0 |  |
| pandas_W0_c | oom_budget | 29.000 | 84.4 | W0 c run 0 |  |
| pandas_W0_arrow | ok | 19.261 | 99.1 | done |  |
| spark_W0 | ok | 2.677 | 122.9 | done |  |
| pandas_10% | ok | 0.435 | 4.2 | done | 576771 |
| pandas_25% | ok | 0.580 | 7.2 | done | 1443465 |
| pandas_50% | ok | 0.960 | 12.0 | done | 2887345 |
| pandas_100% | ok | 1.733 | 21.9 | done | 5775325 |
| pandas_x2 | ok | 3.147 | 39.7 | done | 11550650 |
| pandas_x4 | ok | 6.238 | 76.1 | done | 23101300 |
| pandas_x8 | ok | 12.185 | 150.5 | done | 46202600 |
| spark_10% | ok | 4.961 | 76.2 | done | 576771 |
| spark_25% | ok | 5.781 | 81.5 | done | 1443465 |
| spark_50% | ok | 5.444 | 87.8 | done | 2887345 |
| spark_100% | ok | 6.228 | 100.9 | done | 5775325 |
| spark_x2 | ok | 6.523 | 137.5 | done | 11550650 |
| spark_x4 | ok | 6.865 | 179.5 | done | 23101300 |
| spark_x8 | ok | 6.803 | 272.2 | done | 46202600 |
| pandas_x16 | ok | 24.184 | 326.8 | done | 92405200 |
| spark_x16 | ok | 8.439 | 457.5 | done | 92405200 |
| pandas_x32 | oom_budget | 29.071 | 51.7 | load run 0 | 184810400 |
| spark_x32 | ok | 13.299 | 858.6 | done | 184810400 |

Điểm Pandas hết bộ nhớ: x32. 184.810.400 dòng. RAM 29.071 GB. giai đoạn load run 0. sau 51.7 s.

So kết quả ở 100%: W1_max_abs_diff 0.0; W2_equal True; W3_oe_sum_rel_diff 1.6117737123702159e-15; W3_lender_sd_rel_diff 1.8708708170987051e-16; W3_msa_sd_rel_diff 0.0; W3_top_equal True; all_equal True.

## Trung vị thời gian (s), 5 lần đo sau 1 lần làm nóng

| size | pa W1 | pa W1_cold | pa W2 | pa W2_cold | pa W3 | pa W3_cold | pa load | sp W1_cache | sp W1_nocache | sp W2_cache | sp W2_nocache | sp W3_cache | sp W3_nocache | sp cache_build | sp init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 0.05 | 0.13 | 0.05 | 0.13 | 0.35 | 0.43 | 0.08 | 0.14 | 0.45 | 0.20 | 0.41 | 3.36 | 4.27 | 2.64 | 5.70 |
| 25% | 0.09 | 0.31 | 0.11 | 0.32 | 0.62 | 0.84 | 0.21 | 0.18 | 0.45 | 0.27 | 0.48 | 3.59 | 4.66 | 3.61 | 6.06 |
| 50% | 0.18 | 0.59 | 0.21 | 0.62 | 1.03 | 1.45 | 0.42 | 0.26 | 0.53 | 0.31 | 0.57 | 3.95 | 4.79 | 4.66 | 6.01 |
| 100% | 0.34 | 1.18 | 0.41 | 1.26 | 1.82 | 2.66 | 0.84 | 0.32 | 0.61 | 0.54 | 0.60 | 4.59 | 4.92 | 8.29 | 6.00 |
| x2 | 0.65 | 2.25 | 0.82 | 2.43 | 3.24 | 4.85 | 1.61 | 0.88 | 0.86 | 0.79 | 0.92 | 6.14 | 6.20 | 16.35 | 5.94 |
| x4 | 1.29 | 4.46 | 1.63 | 4.80 | 5.96 | 9.12 | 3.16 | 1.08 | 1.05 | 1.35 | 1.23 | 8.96 | 7.59 | 26.79 | 6.61 |
| x8 | 2.62 | 9.51 | 3.23 | 10.11 | 11.62 | 18.51 | 6.89 | 1.72 | 1.58 | 2.70 | 2.00 | 13.95 | 10.92 | 52.67 | 6.16 |
| x16 | 5.35 | 18.69 | 6.79 | 20.12 | 23.97 | 37.30 | 13.33 | 3.39 | 2.80 | 5.16 | 3.70 | 23.10 | 17.56 | 96.35 | 7.13 |
| x32 |  |  |  |  |  |  |  | 6.94 | 5.16 | 10.44 | 6.65 | 42.61 | 31.59 | 206.49 | 7.15 |
