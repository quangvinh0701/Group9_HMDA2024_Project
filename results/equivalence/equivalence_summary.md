| Quantity | Values compared | Agree | Max relative difference | Explanation |
|---|---|---|---|---|
| (1) Row counts after each filter | 13 | yes | 0.0e+00 | khớp tuyệt đối |
| (2) Special-code counts, 10 fields | 60 | yes | 0.0e+00 | khớp tuyệt đối |
| (2b) Field profile, all 99 fields (codes and numeric values) | 693 | yes | 0.0e+00 | khớp tuyệt đối |
| (3) action_taken counts and shares | 16 | yes | 0.0e+00 | khớp tuyệt đối |
| (4) Mean and median of income, loan amount, rate spread by loan type | 36 | yes | 1.2e-14 | khớp trong dung sai, sai lệch tương đối lớn nhất 1.2e-14 (thứ tự cộng số thực) |
| (4b) Rate spread: numeric, NA, Exempt, out of range | 20 | yes | 0.0e+00 | khớp tuyệt đối |
| (5) Denial rate by state (all codes, including NA) | 165 | yes | 0.0e+00 | khớp tuyệt đối |
| (6) Aggregates of the 50 largest lenders | 450 | yes | 1.8e-13 | khớp trong dung sai, sai lệch tương đối lớn nhất 1.8e-13 (thứ tự cộng số thực) |
| (4) percentile_approx vs exact median (Spark) | 12 | yes | 2.2e-03 | chỉ báo cáo: percentile_approx lệch tương đối tối đa 2.21e-03 (tuyệt đối tối đa 1.00e-03) so với percentile chính xác |
| Key analysis numbers (SQ1-SQ4, two-way, HHI) | 231 | yes | 2.5e-12 | khớp trong dung sai; lệch lớn nhất ở khóa dti5_national_oe; khóa chỉ có một bên: ['conv_twoway_last_iter_max_change', 'env_pandas_notebook', 'env_spark_notebook', 'pred_corr_mllib_vs_sklearn_lr', 'pred_fold_mismatch_vs_pandas', 'pred_fold_mismatch_vs_python', 'pred_maxdiff_mllib_vs_sklearn_lr', 'pred_meanabsdiff_mllib_vs_sklearn_lr'] |
