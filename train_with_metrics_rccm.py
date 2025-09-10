# train_with_metrics_rccm.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

# ---------------------------
# RCCMNet model (same as before)
# ---------------------------
class ResidualChannelContextModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 1), in_channels, bias=False),
            nn.Sigmoid()
        )
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x_att = x * y.expand_as(x)
        context = self.context(x_att)
        res = self.res_conv(x)
        out = self.relu(context + res)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            _, _, h, w = x.size()
            skip = center_crop(skip, (h, w))
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


def center_crop(tensor, target_hw):
    _, _, h, w = tensor.size()
    th, tw = target_hw
    start_h = max((h - th) // 2, 0)
    start_w = max((w - tw) // 2, 0)
    return tensor[:, :, start_h:start_h + th, start_w:start_w + tw]

class RCCMNet(nn.Module):
    def __init__(self, in_channels, num_classes_list, base_filters=32):
        super().__init__()
        bf = base_filters
        self.enc1 = EncoderBlock(in_channels, bf)
        self.enc2 = EncoderBlock(bf, bf * 2)
        self.enc3 = EncoderBlock(bf * 2, bf * 4)
        self.enc4 = EncoderBlock(bf * 4, bf * 8)
        self.center = ResidualChannelContextModule(bf * 8, bf * 16)
        self.dec4 = DecoderBlock(bf * 16, bf * 8)
        self.dec3 = DecoderBlock(bf * 8, bf * 4)
        self.dec2 = DecoderBlock(bf * 4, bf * 2)
        self.dec1 = DecoderBlock(bf * 2, bf)
        self.heads = nn.ModuleList([
            nn.Conv2d(bf, num_classes, kernel_size=1)
            for num_classes in num_classes_list
        ])

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        center = self.center(p4)
        d4 = self.dec4(center, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        outputs = [head(d1) for head in self.heads]
        return outputs

# ---------------------------
# Dataset backed by DataFrame
# ---------------------------
class SegmentationDatasetFromDF(Dataset):
    """
    DataFrame-based dataset for multi-head segmentation.
    Expected df columns:
      - image_col: path to the image file
      - mask_cols: list of columns with paths to mask files (one per head)
    Masks are loaded as single-channel images with integer class labels.

    Example:
      df has columns ['image_path', 'mask_head0', 'mask_head1']
      dataset = SegmentationDatasetFromDF(df, image_col='image_path', mask_cols=['mask_head0','mask_head1'], transforms=...)

    If you already applied oversampling to train_df, pass the oversampled DataFrame here.
    """
    def __init__(self, df, image_col, mask_cols, transforms=None, img_loader=None, mask_loader=None):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.mask_cols = mask_cols
        self.transforms = transforms
        self.img_loader = img_loader if img_loader is not None else self.default_image_loader
        self.mask_loader = mask_loader if mask_loader is not None else self.default_mask_loader

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        img = self.img_loader(img_path)
        masks = []
        for mc in self.mask_cols:
            mpath = row[mc]
            mask = self.mask_loader(mpath)
            masks.append(mask)
        # If single head, return mask tensor directly
        if len(masks) == 1:
            masks = masks[0]
        sample = {'image': img, 'mask': masks}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def default_image_loader(path):
        img = Image.open(path).convert('RGB')
        t = T.ToTensor()
        return t(img)

    @staticmethod
    def default_mask_loader(path):
        m = Image.open(path).convert('L')
        # load as numpy and convert to long tensor of shape (H,W)
        arr = np.array(m, dtype=np.int64)
        return torch.from_numpy(arr).long()

# Optional helper transform that applies the same transforms to image and masks
class ToTupleTransform:
    def __init__(self, image_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, sample):
        img = sample['image']
        masks = sample['mask']
        if self.image_transform is not None:
            img = self.image_transform(img)
        if isinstance(masks, list):
            masks = [self.mask_transform(m.unsqueeze(0).float()).squeeze(0).long() if self.mask_transform is not None else m for m in masks]
        else:
            masks = self.mask_transform(masks.unsqueeze(0).float()).squeeze(0).long() if self.mask_transform is not None else masks
        return {'image': img, 'mask': masks}

# ---------------------------
# Utilities & Metrics
# ---------------------------

def pred_to_labels(pred_logits, num_classes):
    if num_classes == 1:
        probs = torch.sigmoid(pred_logits)
        labels = (probs > 0.5).long().squeeze(1)
    else:
        labels = torch.argmax(pred_logits, dim=1).long()
    return labels


def compute_pixel_accuracy(pred_labels_flat, true_labels_flat):
    assert pred_labels_flat.shape == true_labels_flat.shape
    return float((pred_labels_flat == true_labels_flat).sum()) / pred_labels_flat.size


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(cm, save_path, title="Confusion matrix", class_names=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_classification_report_bars(report_dict, save_path, title="Classification report"):
    classes = [k for k in report_dict.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
    classes_sorted = sorted(classes, key=lambda x: int(x) if x.isdigit() else x)
    precision = [report_dict[c]['precision'] for c in classes_sorted]
    recall = [report_dict[c]['recall'] for c in classes_sorted]
    f1 = [report_dict[c]['f1-score'] for c in classes_sorted]

    x = np.arange(len(classes_sorted))
    width = 0.25
    plt.figure(figsize=(max(6, len(classes_sorted)*0.6), 4))
    plt.bar(x - width, precision, width, label='precision')
    plt.bar(x, recall, width, label='recall')
    plt.bar(x + width, f1, width, label='f1-score')
    plt.xticks(x, classes_sorted, rotation=45)
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------------------------
# Training / Validation
# ---------------------------

def build_default_loss_fns(num_classes_list):
    loss_fns = []
    for nc in num_classes_list:
        if nc == 1:
            loss_fns.append(nn.BCEWithLogitsLoss())
        else:
            loss_fns.append(nn.CrossEntropyLoss())
    return loss_fns


def train_one_epoch(model, dataloader, optimizer, criterion_fns, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        inputs = batch['image'].to(device)
        targets = batch['mask']
        if isinstance(targets, list):
            targets = [t.to(device) for t in targets]
        else:
            targets = [targets.to(device)]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = 0.0
        for i, out in enumerate(outputs):
            loss_fn = criterion_fns[i]
            target = targets[i] if i < len(targets) else targets[0]
            if out.shape[1] == 1:
                loss = loss + loss_fn(out, target.unsqueeze(1).float())
            else:
                loss = loss + loss_fn(out, target.long())
        loss.backward()
        optimizer.step()
        bs = inputs.size(0)
        running_loss += loss.item() * bs
        total_samples += bs
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def validate_and_compute_metrics(model, dataloader, num_classes_list, device, save_dir, epoch, criterion_fns):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    preds_per_head = [ [] for _ in num_classes_list ]
    trues_per_head = [ [] for _ in num_classes_list ]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            inputs = batch['image'].to(device)
            targets = batch['mask']
            if isinstance(targets, list):
                targets = [t.to(device) for t in targets]
            else:
                targets = [targets.to(device)]

            outputs = model(inputs)
            # compute loss for batch
            batch_loss = 0.0
            for i, out in enumerate(outputs):
                loss_fn = criterion_fns[i]
                target = targets[i] if i < len(targets) else targets[0]
                if out.shape[1] == 1:
                    batch_loss = batch_loss + loss_fn(out, target.unsqueeze(1).float())
                else:
                    batch_loss = batch_loss + loss_fn(out, target.long())

            bs = inputs.size(0)
            running_loss += batch_loss.item() * bs
            total_samples += bs

            for i, out in enumerate(outputs):
                nc = num_classes_list[i]
                pred_labels = pred_to_labels(out, nc)
                true_labels = targets[i] if i < len(targets) else targets[0]
                if true_labels.ndim == 4:
                    true_labels = true_labels.squeeze(1)
                pred_np = pred_labels.detach().cpu().numpy()
                true_np = true_labels.detach().cpu().numpy()
                for b in range(bs):
                    preds_per_head[i].append(pred_np[b].reshape(-1))
                    trues_per_head[i].append(true_np[b].reshape(-1))

    avg_val_loss = running_loss / total_samples if total_samples > 0 else 0.0

    metrics = {}
    cm_dir = os.path.join(save_dir, f"epoch_{epoch}")
    ensure_dir(cm_dir)

    for i, nc in enumerate(num_classes_list):
        pred_flat = np.concatenate(preds_per_head[i], axis=0)
        true_flat = np.concatenate(trues_per_head[i], axis=0)

        if nc == 1:
            labels_for_metrics = [0,1]
            target_names = ['0','1']
            cm = confusion_matrix(true_flat, pred_flat, labels=labels_for_metrics)
            cr_dict = classification_report(true_flat, pred_flat, labels=labels_for_metrics, target_names=target_names, zero_division=0, output_dict=True)
            cr_str = classification_report(true_flat, pred_flat, labels=labels_for_metrics, target_names=target_names, zero_division=0)
        else:
            labels_for_metrics = list(range(nc))
            target_names = [str(c) for c in labels_for_metrics]
            cm = confusion_matrix(true_flat, pred_flat, labels=labels_for_metrics)
            cr_dict = classification_report(true_flat, pred_flat, labels=labels_for_metrics, target_names=target_names, zero_division=0, output_dict=True)
            cr_str = classification_report(true_flat, pred_flat, labels=labels_for_metrics, target_names=target_names, zero_division=0)

        acc = compute_pixel_accuracy(pred_flat, true_flat)

        metrics[f"head_{i}"] = {
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
            "classification_report_str": cr_str,
            "classification_report_dict": cr_dict
        }

        cm_path = os.path.join(cm_dir, f"confusion_head{i}_epoch{epoch}.png")
        plot_confusion_matrix(cm, cm_path, title=f"Confusion matrix - head {i} - epoch {epoch}", class_names=target_names)

        cr_path = os.path.join(cm_dir, f"classification_report_head{i}_epoch{epoch}.txt")
        with open(cr_path, "w") as f:
            f.write(cr_str)

        cr_bar_path = os.path.join(cm_dir, f"classification_bars_head{i}_epoch{epoch}.png")
        plot_classification_report_bars(cr_dict, cr_bar_path, title=f"Class report - head {i} - epoch {epoch}")

        json_path = os.path.join(cm_dir, f"metrics_head{i}_epoch{epoch}.json")
        to_dump = {
            "accuracy": acc,
            "confusion_matrix": metrics[f"head_{i}"]['confusion_matrix'],
            "classification_report": cr_dict
        }
        with open(json_path, "w") as f:
            json.dump(to_dump, f, indent=2)

    return metrics, avg_val_loss

# ---------------------------
# Plotting across epochs
# ---------------------------

def plot_loss_acc(history, num_heads, save_dir):
    ensure_dir(save_dir)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='train_loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='val_loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_over_epochs.png"))
    plt.close()

    for i in range(num_heads):
        key = f"acc_head_{i}"
        if key in history and len(history[key]) > 0:
            plt.figure()
            plt.plot(epochs, history[key], marker='o', label=f'val_acc_head_{i}')
            plt.xlabel("Epoch")
            plt.ylabel("Pixel Accuracy")
            plt.ylim(0,1)
            plt.grid(True)
            plt.title(f"Validation Pixel Accuracy - head {i}")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"acc_head_{i}_over_epochs.png"))
            plt.close()

# ---------------------------
# Orchestration
# ---------------------------

def train_model(model,
                train_loader,
                val_loader,
                num_classes_list,
                device='cuda',
                epochs=20,
                lr=1e-3,
                save_dir='rccm_experiments',
                criterion_fns=None):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ensure_dir(save_dir)
    optimizer = Adam(model.parameters(), lr=lr)

    if criterion_fns is None:
        criterion_fns = build_default_loss_fns(num_classes_list)
    else:
        assert len(criterion_fns) == len(num_classes_list)

    history = {
        'train_loss': [],
        'val_loss': []
    }
    for i in range(len(num_classes_list)):
        history[f"acc_head_{i}"] = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_fns, device)
        history['train_loss'].append(train_loss)

        metrics, val_loss = validate_and_compute_metrics(model, val_loader, num_classes_list, device, save_dir, epoch, criterion_fns)
        history['val_loss'].append(val_loss)

        for i in range(len(num_classes_list)):
            acc = metrics[f"head_{i}"]['accuracy']
            history[f"acc_head_{i}"].append(acc)

        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
        plot_loss_acc(history, len(num_classes_list), save_dir)

    return history

# ---------------------------
# Example usage / Integration with train_df/test_df
# ---------------------------
if __name__ == "__main__":
    # Example: If you already have pandas DataFrames train_df and test_df where each row contains
    # paths to image and masks, you can create datasets like this:
    # train_dataset = SegmentationDatasetFromDF(train_df, image_col='image_path', mask_cols=['mask_head0','mask_head1'], transforms=None)
    # val_dataset = SegmentationDatasetFromDF(test_df, image_col='image_path', mask_cols=['mask_head0','mask_head1'], transforms=None)

    # For demonstration we keep a dummy dataset to allow quick runs.
    class DummySegDataset(Dataset):
        def __init__(self, num_samples=10, img_size=(3,128,128), num_heads=[1,3]):
            self.num_samples = num_samples
            self.img_size = img_size
            self.num_heads = num_heads

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            img = torch.randn(*self.img_size)
            masks = []
            H, W = self.img_size[1], self.img_size[2]
            for c in self.num_heads:
                if c == 1:
                    m = (torch.rand(H, W) > 0.7).long()
                else:
                    m = torch.randint(0, c, (H, W)).long()
                masks.append(m)
            if len(masks) == 1:
                masks = masks[0]
            return {'image': img, 'mask': masks}

    train_ds = DummySegDataset(num_samples=20, img_size=(3,128,128), num_heads=[1,3])
    val_ds = DummySegDataset(num_samples=8, img_size=(3,128,128), num_heads=[1,3])
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    num_classes_list = [1, 3]
    model = RCCMNet(in_channels=3, num_classes_list=num_classes_list, base_filters=16)

    history = train_model(model,
                          train_loader,
                          val_loader,
                          num_classes_list=num_classes_list,
                          device='cpu',
                          epochs=3,
                          lr=1e-3,
                          save_dir='rccm_experiments_example')
