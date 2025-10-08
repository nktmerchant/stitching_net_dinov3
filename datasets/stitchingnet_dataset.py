from pathlib import Path
from torch.utils.data import Dataset
from transformers.image_utils import load_image
import polars as pl


class StitchingnetDataset(Dataset):
    """
    Returns images in CHW format, pixel values 0-255
    """

    _IMAGE_PATH_KEY = "raw_image_path"

    def __init__(
        self,
        dataset_dir: Path = Path("/home/kiran/Desktop/SamWood/stitchingnet-dataset/"),
        metadata_dataframe_parquet_path: Path = Path(
            "/home/kiran/Desktop/SamWood/stitchingnet-dataset/metadata.parquet"
        ),
    ):
        self._metadata_dataframe = pl.read_parquet(metadata_dataframe_parquet_path)
        self.dataset_dir = dataset_dir

    def __getitem__(self, index: int) -> dict:
        image_path = self._metadata_dataframe[index][
            StitchingnetDataset._IMAGE_PATH_KEY
        ].item()

        return {"image_path": (self.dataset_dir / image_path).as_posix()}

    def __len__(self):
        return len(self._metadata_dataframe)


if __name__ == "__main__":
    """
    Example code
    """
    ds = StitchingnetDataset()

    print(f"dataset len: {len(ds)}")
    print(f"image path: {ds[0]['image_path']}")
    print(f"image: {ds[0]['image']}")
