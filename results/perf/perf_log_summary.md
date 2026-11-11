# HMDA v1 performance, Kaggle run 1 (2026-11-07), chép từ log notebook

Môi trường: 4 CPU, RAM container 32,2 GB (cgroup 30 GiB), ngân sách 29,0 GB; pandas 2.3.3, pyarrow 23.0.1, pyspark 4.0.4, Python 3.13.
Thời gian chạy notebook: 4119 s. W0 đọc CSV gốc FFIEC (4,64 GB, 99 cột).

## RAM cao nhất (GB) và trạng thái

| engine | size | rows | status | peak_rss_gb | stage |
|---|---|---|---|---|---|
| pandas_pyarrow | W0 | | oom_budget | 29.063 | W0 pyarrow run 0 |
| pandas_c | W0 | | oom_budget | 28.994 | W0 c run 0 |
| spark | W0 | | ok | 2.202 | done |
| pandas | 10% | 576771 | ok | 0.392 | done |
| pandas | 25% | 1443465 | ok | 0.664 | done |
| pandas | 50% | 2887345 | ok | 1.223 | done |
| pandas | 100% | 5775325 | ok | 2.184 | done |
| pandas | x2 | 11550650 | ok | 4.130 | done |
| pandas | x4 | 23101300 | ok | 7.732 | done |
| pandas | x8 | 46202600 | ok | 15.992 | done |
| pandas | x16 | 92405200 | oom_budget | 29.002 | W3 run 0 |
| spark | 10% | 576771 | ok | 4.644 | done |
| spark | 25% | 1443465 | ok | 7.624 | done |
| spark | 50% | 2887345 | ok | 5.075 | done |
| spark | 100% | 5775325 | ok | 6.054 | done |
| spark | x2 | 11550650 | ok | 5.803 | done |
| spark | x4 | 23101300 | ok | 6.344 | done |
| spark | x8 | 46202600 | ok | 6.872 | done |
| spark | x16 | 92405200 | ok | 9.370 | done |

Wall time của tiến trình con (s): pandas_W0_pyarrow 246.6, pandas_W0_c 144.8, spark_W0 216.9 (1 làm nóng + 5 lần đo);
pandas 10% 6.8, 25% 12.0, 50% 21.1, 100% 38.5, x2 72.3, x4 142.8, x8 274.3, x16 411.1 (hết RAM);
spark 10% 107.8, 25% 124.3, 50% 134.4, 100% 156.4, x2 200.5, x4 292.9, x8 469.1, x16 815.7.

Điểm Pandas hết bộ nhớ: x16, 92.405.200 dòng, RAM 29,002 GB, giai đoạn W3 run 0, sau 411,1 s.

So kết quả ở 100%: W1_max_abs_diff 0.0; W2_equal True; W3_oe_sum_rel_diff 1.61e-15; W3_lender_sd_rel_diff 1.87e-16; W3_msa_sd_rel_diff 0.0; W3_top_equal True.

## Trung vị thời gian (s), 5 lần đo sau 1 lần làm nóng

| size | pd W1 | pd W1 cold | pd W2 | pd W2 cold | pd W3 | pd W3 cold | pd load | sp W1 cache | sp W1 nocache | sp W2 cache | sp W2 nocache | sp W3 cache | sp W3 nocache | sp cache_build | sp init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 0.10 | 0.23 | 0.10 | 0.23 | 0.59 | 0.71 | 0.13 | 0.19 | 0.57 | 0.25 | 0.59 | 5.02 | 6.16 | 3.18 | 8.08 |
| 25% | 0.23 | 0.56 | 0.22 | 0.54 | 1.06 | 1.38 | 0.32 | 0.29 | 0.73 | 0.37 | 0.64 | 6.05 | 7.28 | 6.12 | 7.98 |
| 50% | 0.43 | 1.06 | 0.45 | 1.08 | 1.74 | 2.37 | 0.64 | 0.33 | 0.78 | 0.47 | 0.79 | 6.36 | 7.98 | 7.57 | 7.95 |
| 100% | 0.84 | 2.13 | 0.86 | 2.15 | 3.08 | 4.37 | 1.29 | 0.55 | 0.85 | 0.81 | 1.01 | 8.11 | 7.38 | 14.44 | 7.93 |
| x2 | 1.66 | 4.19 | 1.73 | 4.26 | 5.65 | 8.18 | 2.53 | 0.86 | 1.12 | 1.34 | 1.28 | 9.97 | 10.13 | 24.16 | 8.20 |
| x4 | 3.36 | 8.97 | 3.53 | 9.14 | 10.65 | 16.26 | 5.61 | 1.66 | 1.58 | 2.47 | 1.81 | 15.80 | 11.83 | 45.60 | 8.29 |
| x8 | 6.72 | 17.05 | 6.96 | 17.29 | 20.54 | 30.87 | 10.33 | 3.11 | 2.46 | 4.79 | 3.34 | 24.92 | 17.29 | 92.52 | 12.02 |
| x16 | | | | | | | | 6.17 | 4.56 | 9.32 | 5.95 | 43.15 | 29.33 | 177.44 | 15.07 |
