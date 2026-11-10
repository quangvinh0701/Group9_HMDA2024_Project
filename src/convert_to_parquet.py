"""
Convert the HMDA 2024 LAR CSV to Parquet without loading the whole file into memory.

- Every column is kept as a string, so codes such as "NA", "Exempt", "1111", "8888"
  stay exactly as published (cleaning happens later, in Pandas and Spark, from the
  same rules file).
- A row_id column (0-based position of the record in the CSV) is added so both
  pipelines and the benchmark can select identical records.
- Memory use stays around a few hundred MB regardless of file size.

Usage:
    python convert_to_parquet.py 2024_lar.csv hmda_2024_raw.parquet
"""
import sys
import time

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq


def read_header(path: str) -> tuple[list[str], str]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().rstrip("\r\n")
    delimiter = "|" if first.count("|") > first.count(",") else ","
    return [c.strip().strip('"') for c in first.split(delimiter)], delimiter


def convert(src: str, dst: str, block_mb: int = 256) -> None:
    columns, delimiter = read_header(src)

    read_opts = pv.ReadOptions(block_size=block_mb << 20)
    parse_opts = pv.ParseOptions(delimiter=delimiter)
    # By default Arrow (and Pandas) turn "NA", "N/A", "NaN", "null" and "" into nulls.
    # In HMDA, "NA" means "not applicable", which is information, so nothing is nulled here.
    conv_opts = pv.ConvertOptions(
        column_types={c: pa.string() for c in columns},
        null_values=[],
        strings_can_be_null=False,
        quoted_strings_can_be_null=False,
    )

    reader = pv.open_csv(src, read_options=read_opts,
                         parse_options=parse_opts, convert_options=conv_opts)

    writer = None
    n_rows = 0
    t0 = time.perf_counter()
    for batch in reader:
        row_id = pa.array(range(n_rows, n_rows + batch.num_rows), type=pa.int64())
        table = pa.Table.from_batches([batch]).append_column("row_id", row_id)
        if writer is None:
            writer = pq.ParquetWriter(dst, table.schema, compression="snappy")
        writer.write_table(table)
        n_rows += batch.num_rows
        print(f"\r{n_rows:,} rows written", end="", flush=True)
    if writer is not None:
        writer.close()

    meta = pq.ParquetFile(dst).metadata
    print(f"\nDone in {time.perf_counter() - t0:.1f}s: {meta.num_rows:,} rows, "
          f"{meta.num_columns} columns (incl. row_id), {meta.num_row_groups} row groups")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python convert_to_parquet.py <input.csv> <output.parquet>")
    convert(sys.argv[1], sys.argv[2])
