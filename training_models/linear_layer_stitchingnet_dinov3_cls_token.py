"""
This code was written with ChatGPT
"""

import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets.stitchingnet_dinov3_cls_token_dataset import (
    StitchingnetDinoV3ClsTokenDataset,
)
from tqdm import tqdm
from PIL import Image
import wandb


# --- W&B run ---
run = wandb.init(
    entity="nktmerchant-supermodel-research",
    project="linear-probe-stitchingnet-dinov3-cls-token",
    config={
        "learning_rate": 1e-3,
        "batch_size": 64,
        "architecture": "Linear NN",
        "dataset": "StitchingnetDataset",
        "epochs": 50,
    },
)

CLASS_TO_INDEX = {
    "0_normal": 0,
    "1_skipped_stitch": 1,
    "2_broken_stitch": 2,
    "3_pinched_fabric": 3,
    "4_crooked_seam": 4,
    "5_thread_sagging": 5,
    "6_puckering": 6,
    "7_stain_and_damage": 7,
    "8_needle_mark": 8,
    "9_bobbin_thread_pulling_up": 9,
    "10_overlapped_stitch": 10,
}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}


def main():
    # Hyperparameters
    batch_size = 64
    lr = 1e-3
    epochs = 50
    num_classes = len(CLASS_TO_INDEX)

    # Datasets
    train_dataset = StitchingnetDinoV3ClsTokenDataset(split="train")
    val_dataset = StitchingnetDinoV3ClsTokenDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    feature_dim = 1280
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(feature_dim, num_classes)).to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch["cls_token"].squeeze(1).to(device)  # (B, 1280)
            y = torch.tensor(
                [CLASS_TO_INDEX[label] for label in batch["cls_name"]], device=device
            )

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        train_loss = total_loss / total
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        run.log({"train/loss": train_loss, "train/acc": train_acc, "epoch": epoch})

        # --- Validation (per-epoch metrics) ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["cls_token"].squeeze(1).to(device)
                y = torch.tensor(
                    [CLASS_TO_INDEX[label] for label in batch["cls_name"]],
                    device=device,
                )
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total else 0.0
        val_loss = val_loss_sum / val_total if val_total else 0.0
        print(f"Val  Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        run.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch})

        # --- Per-epoch random validation sample: upload image + labels to W&B ---
        idx = random.randrange(len(val_dataset))
        sample = val_dataset[
            idx
        ]  # dict with 'image_path', 'cls_name', 'cls_token', ...
        img_path = Path(sample["image_path"])
        true_label = sample["cls_name"]

        # Forward pass on single sample to get prediction
        x_single = sample["cls_token"].view(1, -1).to(device)
        with torch.no_grad():
            logits = model(x_single)
            pred_idx = int(logits.argmax(dim=1).item())
            pred_label = INDEX_TO_CLASS[pred_idx]

        # Load image from disk and log to W&B
        try:
            img = Image.open(img_path).convert("RGB")
            caption = f"epoch={epoch} | true={true_label} | pred={pred_label}"
            run.log(
                {
                    "media/random_sample": wandb.Image(img, caption=caption)
                }  # , step=epoch
            )
            print(f"Logged {img_path} as a random validation sample")
        except Exception as e:
            run.log({"media/random_sample_error": str(e)}, step=epoch)

    # --- Final evaluation on validation for confusion matrix (uses probs) ---
    # Requirements per W&B docs:
    # - probs: array of shape (N, K) with class probabilities
    # - y_true: sequence of true class indices
    # - class_names: names ordered to align with columns in probs
    import numpy as np

    model.eval()
    class_names = [k for k, _ in sorted(CLASS_TO_INDEX.items(), key=lambda kv: kv[1])]
    num_classes = len(class_names)

    probs_list: list[np.ndarray] = []
    y_true_idx: list[int] = []

    with torch.no_grad():
        for i in range(len(val_dataset)):  # iterate items directly (no DataLoader)
            sample = val_dataset[i]
            x = sample["cls_token"].view(1, -1).to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (K,)
            probs_list.append(p)
            y_true_idx.append(CLASS_TO_INDEX[sample["cls_name"]])

    probs_arr = (
        np.stack(probs_list, axis=0) if probs_list else np.zeros((0, num_classes))
    )

    cm_plot = wandb.plot.confusion_matrix(
        probs=probs_arr,
        y_true=y_true_idx,
        class_names=class_names,
        title="Validation Confusion Matrix (probabilities)",
    )
    wandb.log({"val/confusion_matrix": cm_plot})

    # --- Save final model weights to disk ---
    save_path = Path("linear_probe_dinov3.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()
