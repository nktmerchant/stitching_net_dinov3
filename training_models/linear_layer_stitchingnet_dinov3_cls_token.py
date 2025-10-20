import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from datasets.stitchingnet_dinov3_cls_token_dataset import (
    StitchingnetDinoV3ClsTokenDataset,
)
from tqdm import tqdm

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="nktmerchant-supermodel-research",
    # Set the wandb project where this run will be logged.
    project="linear-probe-stitchingnet-dinov3-cls-token",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 1e-3,
        "batch_size": 64,
        "architecture": "Linear NN",
        "dataset": "StitchingnetDataset",
        "epochs": 100,
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


def main():
    # Hyperparameters
    batch_size = 64
    lr = 1e-3
    epochs = 100
    num_classes = 11

    # Dataset + Random Split
    full_dataset = StitchingnetDinoV3ClsTokenDataset()
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    feature_dim = 1280
    model = nn.Sequential(
        nn.Linear(feature_dim, num_classes),
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = next(model.parameters()).device

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch["cls_token"].squeeze(1).to(device)
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
        avg_loss = total_loss / total
        print(f"Train Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")
        run.log({"train/loss": avg_loss, "train/acc": train_acc})

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["cls_token"].squeeze(1).to(device)
                y = torch.tensor(
                    [CLASS_TO_INDEX[label] for label in batch["cls_name"]],
                    device=device,
                )

                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        run.log({"val/loss": avg_loss, "val/acc": train_acc})

    # torch.save(model.state_dict(), "linear_probe_dinov3.pth")
    # print("Saved model to linear_probe_dinov3.pth")


if __name__ == "__main__":
    main()
