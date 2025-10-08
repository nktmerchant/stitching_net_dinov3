"""
Usage:
"""

import argparse
import polars as pl
from pathlib import Path


def main(txt_read_path: Path, column_name: str, df_write_path: Path) -> None:
    with open(txt_read_path) as txt_file:
        # Sanitize spaces at the end of lines
        lines = [line.strip() for line in txt_file if line.strip()]

    df = pl.DataFrame({column_name: lines})
    df.write_parquet(df_write_path)

    return df, df_write_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt-read-path", type=Path, required=True)
    parser.add_argument("--column-name", type=str, required=True)
    parser.add_argument("--df-write-path", type=Path, required=True)

    args = parser.parse_args()

    df, df_write_path = main(args.txt_read_path, args.column_name, args.df_write_path)
    print(f"Successfully wrote {len(df)} rows to {df_write_path.as_posix()}")
