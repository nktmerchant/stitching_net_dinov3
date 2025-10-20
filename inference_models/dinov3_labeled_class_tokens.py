"""
Based on https://huggingface.co/docs/transformers/main/en/model_doc/dinov3
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

from datasets.stitchingnet_dataset import StitchingnetDataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers.image_utils import load_image
from torchvision.utils import save_image

import argparse

"""
Example models

pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
pretrained_model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
"""


def load_processor_and_model(pretrained_model_name):
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name,
        device_map="auto",
    )

    return processor, model


def compute_cls_token(processor, model, image):
    # Pre-process inputs with the AutoImageProcessor pipeline
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    # Get patch constants
    patch_size = model.config.patch_size
    batch_size, _, img_height, img_width = inputs.pixel_values.shape
    num_patches_height, num_patches_width = (
        img_height // patch_size,
        img_width // patch_size,
    )

    # Run inference
    with torch.inference_mode():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    cls_token = last_hidden_states[:, 0, :]

    return cls_token


def main(pretrained_model_name: str, cls_token_output_dir: Path | None):
    # Load data
    dataset = StitchingnetDataset()
    dataloader = DataLoader(dataset, batch_size=1)

    # Load model
    processor, model = load_processor_and_model(pretrained_model_name)

    # Inference
    for sample in tqdm(dataloader, desc="Computing and saving class tokens"):
        image_path = sample["image_path"][0]
        image = load_image(image_path)

        cls_token = compute_cls_token(processor, model, image)

        if cls_token_output_dir is not None:
            # Get the class from the parent directory name
            cls_name = (
                Path(image_path).parent.name.lower().replace(".", "").replace(" ", "_")
            )

            # Save the class token and label as a pickled object
            output_path = cls_token_output_dir / Path(image_path).relative_to(
                dataset.dataset_dir
            )
            output_path = output_path.with_suffix(".pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({"cls_name": cls_name, "cls_token": cls_token}, output_path)


if __name__ == "__main__":
    DEFAULT_PRETRAINED_MODEL_NAME = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cls-token-output-dir",
        default=Path(
            "/home/kiran/Desktop/SamWood/cls_token_datasets/cls_token_outputs_vith16plus"
        ),
    )
    parser.add_argument(
        "--pretrained-model-name", type=str, default=DEFAULT_PRETRAINED_MODEL_NAME
    )
    args = parser.parse_args()

    main(args.pretrained_model_name, args.cls_token_output_dir)
