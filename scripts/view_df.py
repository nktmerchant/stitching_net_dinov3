import argparse
from pathlib import Path
import polars as pl


def main(df_read_path: Path) -> None:
    df = pl.read_parquet(df_read_path)
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-read-path", type=Path, required=True)
    args = parser.parse_args()

    main(args.df_read_path)
