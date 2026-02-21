





import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt







import torch
import torch.nn as nn

# ----------------------------
# Conv Block (DPU Safe)
# Conv → BN → ReLU
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# SegNet-Lite (FULLY DPU SAFE)
# ----------------------------
class SegNetLiteNew(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2, stride=2)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4 = DoubleConv(256, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = DoubleConv(32, 32)

        # Output head (RAW LOGITS ONLY)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))

        x = self.dec4(self.up4(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))

        return self.out(x)









class MriLesionDataset(Dataset):
    def __init__(self, root_dir, patch_size=64):
        self.root_dir = root_dir
        self.patch = patch_size

        cancer_ids = os.listdir(f"{root_dir}/images/cancer")
        non_cancer_ids = os.listdir(f"{root_dir}/images/non_cancer")

        non_cancer_ids = non_cancer_ids[:len(cancer_ids)//2]  # 2:1 ratio

        self.samples = []
        for pid in cancer_ids:
            self.samples.append(("cancer", pid))
        for pid in non_cancer_ids:
            self.samples.append(("non_cancer", pid))

    def load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (192,192))
        img = img.astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-5)
        return img

    def load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (192,192), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        return mask

    def crop_patch(self, img, mask):
        """
        img  : (3, H, W)
        mask : (1, H, W)
        returns fixed (3, patch, patch), (1, patch, patch)
        """
    
        patch = self.patch
        _, H, W = img.shape
    
        mask2d = mask[0]
    
        ys, xs = np.where(mask2d > 0)
        if len(xs) == 0:
            y, x = H // 2, W // 2
        else:
            y, x = int(ys.mean()), int(xs.mean())
    
        s = patch // 2
    
        y1, y2 = y - s, y + s
        x1, x2 = x - s, x + s
    
        # ---- padding calculation ----
        pad_top = max(0, -y1)
        pad_left = max(0, -x1)
        pad_bottom = max(0, y2 - H)
        pad_right = max(0, x2 - W)
    
        # ---- apply padding ----
        img = np.pad(
            img,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant"
        )
        mask = np.pad(
            mask,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant"
        )
    
        # ---- shift crop window after padding ----
        y1 += pad_top
        y2 += pad_top
        x1 += pad_left
        x2 += pad_left
    
        return img[:, y1:y2, x1:x2], mask[:, y1:y2, x1:x2]


    def __getitem__(self, idx):
        label, pid = self.samples[idx]
        base = f"{self.root_dir}/images/{label}/{pid}/{pid}"

        t2w = self.load_img(base + "_t2w.png")
        hbv = self.load_img(base + "_hbv.png")
        adc = self.load_img(base + "_adc.png")

        image = np.stack([t2w, hbv, adc], axis=0)

        mask_path = f"{self.root_dir}/masks/{label}/{pid}.png"
        mask = self.load_mask(mask_path)
        mask = np.expand_dims(mask, axis=0)

        image, mask = self.crop_patch(image, mask)

        return torch.tensor(image), torch.tensor(mask)

    def __len__(self):
        return len(self.samples)








class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, smooth=1.):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = 1 - (2*intersection + smooth)/(probs.sum() + targets.sum() + smooth)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1-pt)**self.gamma * bce
        return dice + focal.mean()


def dice_score(pred, target, smooth=1.):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return (2*inter + smooth)/(pred.sum() + target.sum() + smooth)









def show_prediction(img, gt, pred):
    t2w = img[0]
    pred = (torch.sigmoid(pred) > 0.5).float()

    plt.figure(figsize=(12,4))
    plt.subplot(1,4,1); plt.imshow(t2w, cmap="gray"); plt.title("T2W")
    plt.subplot(1,4,2); plt.imshow(t2w, cmap="gray"); plt.imshow(gt[0], cmap="Reds", alpha=0.5); plt.title("GT")
    plt.subplot(1,4,3); plt.imshow(t2w, cmap="gray"); plt.imshow(pred[0], cmap="Blues", alpha=0.5); plt.title("Pred")
    plt.subplot(1,4,4); plt.imshow(t2w, cmap="gray"); plt.imshow(gt[0], cmap="Reds", alpha=0.5); plt.imshow(pred[0], cmap="Blues", alpha=0.5); plt.title("Overlay")
    plt.axis("off")
    plt.show()









device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = "/home/username/Desktop/Neena/Vitis-AI/Project_MRI/prostate_cancer-20260114T070259Z-1-001/prostate_cancer"

dataset = MriLesionDataset(root_dir)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SegNetLiteNew().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = DiceFocalLoss()

EPOCHS = 70

for epoch in range(EPOCHS):
    model.train()
    loss_sum, dice_sum = 0, 0

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, mask)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        dice_sum += dice_score(out, mask).item()

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {loss_sum/len(loader):.4f} | "
          f"Dice: {dice_sum/len(loader):.4f}")

    if (epoch+1) % 3 == 0:
        model.eval()
        img, mask = next(iter(loader))
        with torch.no_grad():
            pred = model(img.to(device))
        show_prediction(img[0].cpu(), mask[0].cpu(), pred[0].cpu())








# =====================================================
# 2️⃣ SAVE WEIGHTS ONLY (for Vitis AI / deployment)
# =====================================================
torch.save(
    model.state_dict(),
    "segnet_lite_lesion.pth"
)
DEVICE = torch.device("cpu")








def compute_metrics(pred, target, eps=1e-7):
    """
    pred, target: torch tensors (B,1,H,W) – binary (0/1)
    """

    pred = pred.view(-1)
    target = target.view(-1)

    TP = (pred * target).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()
    TN = ((1 - pred) * (1 - target)).sum().item()

    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    iou  = (TP + eps) / (TP + FP + FN + eps)

    precision = (TP + eps) / (TP + FP + eps)
    recall    = (TP + eps) / (TP + FN + eps)      # sensitivity
    specificity = (TN + eps) / (TN + FP + eps)

    accuracy = (TP + TN + eps) / (TP + TN + FP + FN + eps)

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Accuracy": accuracy
    }









model.eval()

metrics_sum = {
    "Dice": 0,
    "IoU": 0,
    "Precision": 0,
    "Recall": 0,
    "Specificity": 0,
    "Accuracy": 0
}

count = 0

with torch.no_grad():
    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)

        logits = model(img)
        pred = (torch.sigmoid(logits) > 0.5).float()

        batch_metrics = compute_metrics(pred, mask)

        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]

        count += 1

# ---- average ----
for k in metrics_sum:
    metrics_sum[k] /= count









print("==== FINAL EVALUATION METRICS ====")
for k, v in metrics_sum.items():
    print(f"{k:12s}: {v:.4f}")










 


