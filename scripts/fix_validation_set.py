import polars as pl
import numpy as np
from pathlib import Path
import argparse


def mark_split(df: pl.DataFrame, val_percent: float, seed: int) -> pl.DataFrame:
    np.random.seed(seed)
    n = len(df)
    val_size = int(n * val_percent / 100)
    val_indices = np.random.choice(n, size=val_size, replace=False)
    split_col = np.array(["train"] * n)
    split_col[val_indices] = "val"
    return df.with_columns(pl.Series("split", split_col))


def main(df_path: Path, val_percent: float, seed: int):
    df = pl.read_parquet(df_path)
    df = mark_split(df, val_percent=val_percent, seed=seed)
    df.write_parquet(df_path)
    print(f"Overwrote {df_path} with split column.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-path", type=Path, help="Path to input parquet DataFrame.")
    parser.add_argument(
        "--val-percent",
        type=float,
        default=20.0,
        help="Validation split percent (default: 20%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    main(args.df_path, args.val_percent, args.seed)
