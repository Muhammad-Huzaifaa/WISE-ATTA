import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# -----------------------------
# Config
# -----------------------------
DATA_ROOT = "/mnt/lustre/work/kuehne/kqr916/high-res/EATTA/dataset/PACS"
ANNO_ROOT = "/mnt/lustre/work/kuehne/kqr916/high-res/EATTA/dataset/"
SPLIT_ROOT = os.path.join(ANNO_ROOT, "Train val splits and h5py files pre-read")
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES = 7  # PACS has 7 classes


# -----------------------------
# Custom Dataset
# -----------------------------
class PACSDataset(Dataset):
    def __init__(self, split_file, data_root, transform=None):
        self.samples = []
        self.transform = transform
        self.data_root = data_root

        with open(split_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                label = int(label) - 1  # shift to 0-based indexing
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.data_root, rel_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# -----------------------------
# Helpers
# -----------------------------
def get_transform(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_dataloader(domain, split="train"):
    split_file = os.path.join(SPLIT_ROOT, f"{domain}_{split}_kfold.txt")
    dataset = PACSDataset(split_file, DATA_ROOT, transform=get_transform(split))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=(split == "train"), num_workers=4)
    return loader


def build_model():
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    for p in model.parameters():
        p.requires_grad = True
    # freeze BN running stats, but keep affine trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()  # Use batch statistics, not running statistics
            m.track_running_stats = False 
    return model


# -----------------------------
# Training / Evaluation
# -----------------------------
def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_total = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            loss_total += loss.item() * imgs.size(0)
            total += labels.size(0)
    return correct / total, loss_total / total


def train_on_domain(source_domain):
    print(f"\n=== Training on source domain = {source_domain} ===")

    train_loader = get_dataloader(source_domain, split="train")
    val_loader = get_dataloader(source_domain, split="crossval")
    test_loader = get_dataloader(source_domain, split="test")

    model = build_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc, val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Acc: {train_acc*100:.2f}% "
              f"Val Acc: {val_acc*100:.2f}%")

        # save best checkpoint based on val acc
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_model_state = model.state_dict().copy()

    # load best model before testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/resnet18_pacs_source_{source_domain}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved best checkpoint: {ckpt_path}")

    # Evaluate on source test set
    test_acc, test_loss = evaluate(model, test_loader, criterion)
    print(f"[Source={source_domain}] Test Acc: {test_acc*100:.2f}%")

    return model


def evaluate_on_domains(model, source_domain):
    criterion = nn.CrossEntropyLoss()
    for domain in DOMAINS:
        test_loader = get_dataloader(domain, split="test")
        acc, loss = evaluate(model, test_loader, criterion)
        marker = "(source)" if domain == source_domain else "(target)"
        print(f"Test on {domain} {marker}: "
              f"Loss {loss:.4f}, Acc {acc*100:.2f}%")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    for source_domain in DOMAINS:
        model = train_on_domain(source_domain)
        evaluate_on_domains(model, source_domain)
