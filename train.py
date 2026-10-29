# Runs model training and validation, records results and saves checkpoints.
import torch
from torch import nn
from leafapp.config import load_config
from leafapp.data import get_loaders
from leafapp.models.cnn import BasicCNN

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        images, labels = batch["pixels"], batch["label"]
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()             # clear last batch's gradients
        logits = model(images)            # predict
        loss = criterion(logits, labels)  # how wrong?
        loss.backward()                   # work out how each weight should change
        optimizer.step()                  # change them

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            images, labels = batch["pixels"], batch["label"]
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / total, correct / total

def main():
    cfg = load_config("configs/prototype.yaml")
    torch.manual_seed(cfg["seed"])
    train_loader, val_loader, test_loader, class_names = get_loaders(
        fraction=cfg["data"]["fraction"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        seed=cfg["seed"],
    )

    model = BasicCNN(
        num_classes=len(class_names),
        channels=cfg["model"]["channels"],
        image_size=cfg["data"]["image_size"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    for epoch in range(cfg["train"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.3f}, train acc {train_acc:.2f} | val loss {val_loss:.3f}, val acc {val_acc:.2f}")

if __name__ == "__main__":
    main()