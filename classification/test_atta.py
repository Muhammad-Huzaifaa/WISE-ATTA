import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from PIL import Image

# ATTA imports

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)



class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, hparams):
        super(ResNet, self).__init__()
        nc = 3

        self.network = torchvision.models.resnet18(pretrained=True)
        self.n_outputs = 512


        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        # nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        # self.fc = copy.deepcopy(self.network.fc)
        del self.network.fc
        self.network.fc = Identity()

        # self.fr_bn = hparams.model['freeze_bn']
        self.freeze_bn()
        # self.hparams = hparams
        self.dropout = nn.Dropout(int(0))

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    


# -----------------------------
# Config
# -----------------------------
DATA_ROOT = "/mnt/lustre/work/kuehne/kqr916/high-res/EATTA/dataset/PACS"
ANNO_ROOT = "/mnt/lustre/work/kuehne/kqr916/high-res/EATTA/dataset/"
SPLIT_ROOT = os.path.join(ANNO_ROOT, "Train val splits and h5py files pre-read")
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES = 7  # PACS has 7 classes

# --- IMPORTANT ---
# These must match your training config (SimATTA.yaml)
hparams = {
    "dataset": {"input_shape": (3, 224, 224), "num_classes": NUM_CLASSES},
    "model": {
        "resnet18": True,        # True if you trained with ResNet18
        "freeze_bn": True,       # matches "Truefzbn"
        "dropout_rate": 0.0,     # "0dp" in checkpoint folder name
        "nonlinear_classifier": False  # change to True if nonlinear head used
    },
    "mlp_width": 256,
    "mlp_depth": 3,
    "mlp_dropout": 0.1
}

# -----------------------------
# Dataset + Loader
# -----------------------------
class PACSDataset(Dataset):
    def __init__(self, split_file, data_root, transform=None):
        self.samples = []
        self.transform = transform
        self.data_root = data_root
        with open(split_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                label = int(label) - 1  # shift to 0-based
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


def get_transform(split="test"):
    return transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_dataloader(domain, split="test"):
    split_file = os.path.join(SPLIT_ROOT, f"{domain}_{split}_kfold.txt")
    dataset = PACSDataset(split_file, DATA_ROOT, transform=get_transform(split))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------------
# Build ATTA Model (encoder + classifier)
# -----------------------------
def build_model(encoder_path, fc_path):
    encoder = ResNet(hparams)
    encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
    print(f"Loaded encoder from {encoder_path}")

    classifier = Classifier(
        encoder.n_outputs,
        NUM_CLASSES,
        hparams["model"]["nonlinear_classifier"]
    )
    classifier.load_state_dict(torch.load(fc_path, map_location=DEVICE))
    print(f"Loaded classifier from {fc_path}")

    # Combine
    model = nn.Sequential(encoder, classifier).to(DEVICE)
    return model

# -----------------------------
# Evaluation
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


def evaluate_on_domains(encoder_path, fc_path):
    model = build_model(encoder_path, fc_path)
    criterion = nn.CrossEntropyLoss()
    for domain in DOMAINS:
        test_loader = get_dataloader(domain, split="test")
        acc, loss = evaluate(model, test_loader, criterion)
        print(f"[{domain}] Loss {loss:.4f}, Acc {acc*100:.2f}%")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    encoder_path = "/mnt/lustre/work/kuehne/kqr916/high-res/ATTA/storage/checkpoints/round1/PACS/[2, 0, 1, 3]/ResNet18_3l_meanpool_0dp_Truefzbn/0.0001lr_0.0wd/ERM_no_param/encoder_2.pth"
    fc_path = "/mnt/lustre/work/kuehne/kqr916/high-res/ATTA/storage/checkpoints/round1/PACS/[2, 0, 1, 3]/ResNet18_3l_meanpool_0dp_Truefzbn/0.0001lr_0.0wd/ERM_no_param/fc_2.pth"
    evaluate_on_domains(encoder_path, fc_path)
