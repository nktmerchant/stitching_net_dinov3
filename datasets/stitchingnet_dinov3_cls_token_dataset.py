from pathlib import Path
import torch
from torch.utils.data import Dataset
import polars as pl


class StitchingnetDinoV3ClsTokenDataset(Dataset):
    """
    Returns a dictionary with two values
    - cls_name (str): one of ten possible stitchingnet classes
    - cls_token (torch.Tensor): a class token of shape (1, 1280)
    """

    _CLS_TOKEN_DICT_PATH_KEY = "cls_token_path"
    _IMAGE_PATH_KEY = "image_path"

    def __init__(
        self,
        dataset_dir: Path = Path(
            "/home/kiran/Desktop/SamWood/cls_token_datasets/cls_token_outputs_vith16plus"
        ),
        metadata_dataframe_parquet_path: Path = Path(
            "/home/kiran/Desktop/SamWood/cls_token_datasets/cls_token_outputs_vith16plus/metadata.parquet"
        ),
        split: str | None = None,
    ):
        self._metadata_dataframe = pl.read_parquet(metadata_dataframe_parquet_path)

        # Train/validation split
        self._split = split
        if self._split is not None:
            assert self._split in ["train", "val"]
            self._metadata_dataframe = self._metadata_dataframe.filter(
                pl.col("split") == self._split
            )

        self.dataset_dir = dataset_dir

    def __getitem__(self, index: int) -> dict:
        # Load the class token dict
        cls_token_path = self._metadata_dataframe[index][
            StitchingnetDinoV3ClsTokenDataset._CLS_TOKEN_DICT_PATH_KEY
        ].item()
        cls_token_dict = torch.load(cls_token_path)

        # Load the iamge path
        image_path = self._metadata_dataframe[index][
            StitchingnetDinoV3ClsTokenDataset._IMAGE_PATH_KEY
        ].item()

        # Add paths as reflection
        cls_token_dict["cls_token_path"] = cls_token_path
        cls_token_dict["image_path"] = image_path

        return cls_token_dict

    def __len__(self):
        return len(self._metadata_dataframe)


if __name__ == "__main__":
    """
    Example code
    """
    ds = StitchingnetDinoV3ClsTokenDataset(split="val")

    print(f"dataset len: {len(ds)}")
    print(f"cls token path: {ds[0]['cls_token_path']}")
    print(f"image path: {ds[0]['image_path']}")
    print(f"cls token shape: {ds[0]['cls_token'].shape}")
