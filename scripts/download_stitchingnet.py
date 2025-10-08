import kagglehub


def main() -> None:
    """
    Downloads the latest version of `hyungjung/stitchingnet-dataset/` to `~/.cache/kagglehub/datasets/`
    """
    resolved_path = kagglehub.dataset_download("hyungjung/stitchingnet-dataset")

    print("Path to dataset files:", resolved_path)


if __name__ == "__main__":
    main()
