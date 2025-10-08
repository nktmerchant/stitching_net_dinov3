"""
Based on https://huggingface.co/docs/transformers/main/en/model_doc/dinov3

and https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/pca.ipynb#scrollTo=0ebd98e9-13eb-4e7a-a72b-27839dd463d0
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
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


def compute_pca_features(processor, model, image):
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
    # cls_token = last_hidden_states[:, 0, :]
    patch_features_flat = last_hidden_states[
        :, 1 + model.config.num_register_tokens :, :
    ]

    pca = PCA(n_components=3, whiten=True)
    pca_features_flat = torch.from_numpy(
        pca.fit_transform(patch_features_flat.squeeze(0).cpu())
    )
    pca_features = pca_features_flat.view(num_patches_height, num_patches_width, 3)

    # multiply by 2.0 and pass through a sigmoid to get vibrant colors
    pca_features = torch.nn.functional.sigmoid(pca_features.mul(2.0)).permute(2, 0, 1)

    return pca_features


def main(pretrained_model_name: str, pca_output_dir: Path):
    # Load data
    dataset = StitchingnetDataset()
    dataloader = DataLoader(dataset, batch_size=1)

    # Load model
    processor, model = load_processor_and_model(pretrained_model_name)

    # Inference
    for sample in tqdm(dataloader, desc="Computing PCA features"):
        image_path = sample["image_path"][0]
        image = load_image(image_path)

        pca_features = compute_pca_features(processor, model, image)

        # Save the PCA features as an RGB image
        output_path = pca_output_dir / Path(image_path).relative_to(dataset.dataset_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(pca_features, output_path.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model-name", type=str, required=True)
    args = parser.parse_args()

    main(args.pretrained_model_name, args.pca_output_dir)
