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

    _DICT_PATH_KEY = "cls_token_path"

    def __init__(
        self,
        dataset_dir: Path = Path(
            "/home/kiran/Desktop/SamWood/cls_token_datasets/cls_token_outputs_vith16plus"
        ),
        metadata_dataframe_parquet_path: Path = Path(
            "/home/kiran/Desktop/SamWood/cls_token_datasets/cls_token_outputs_vith16plus/metadata.parquet"
        ),
    ):
        self._metadata_dataframe = pl.read_parquet(metadata_dataframe_parquet_path)
        self.dataset_dir = dataset_dir

    def __getitem__(self, index: int) -> dict:
        cls_token_path = self._metadata_dataframe[index][
            StitchingnetDinoV3ClsTokenDataset._DICT_PATH_KEY
        ].item()
        cls_token_dict = torch.load(cls_token_path)

        # Add path as reflection
        cls_token_dict["cls_token_path"] = cls_token_path

        return cls_token_dict

    def __len__(self):
        return len(self._metadata_dataframe)


if __name__ == "__main__":
    """
    Example code
    """
    ds = StitchingnetDinoV3ClsTokenDataset()

    print(f"dataset len: {len(ds)}")
    print(f"cls token path: {ds[0]['cls_token_path']}")
    print(f"cls token shape: {ds[0]['cls_token'].shape}")
