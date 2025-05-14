import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_pc import (
    real3d_classes,
    industrial3d_classes,
    Dataset3dad_train,
    Dataset3dad_test,
    DatasetIndustrial3dad_train,
    DatasetIndustrial3dad_test,
)
from model import Industrial3DAnomalyDetector
from M3DM.models import Model1  


class PointMAEBackboneWrapper(nn.Module):
    """
    Wraps Model1 from M3DM/models.py so that it
    takes a tensor xyz [B,N,3] and returns:
       xyz, feats [B,N,C]
    """
    def __init__(self, device, **kwargs):
        super().__init__()
        self.backbone = Model1(
            device=device,
            xyz_backbone_name='Point_MAE',
            **kwargs
        ).to(device)
        self._feat_dim = None

    @property
    def out_dim(self):
        if self._feat_dim is None:
            dummy = torch.zeros(1, 3, 1024, device=next(self.parameters()).device)
            feats, *_ = self.backbone(dummy)
            self._feat_dim = feats.shape[1]
        return self._feat_dim

    def forward(self, inputs):
        # inputs: dict with 'xyz': [B,N,3]
        xyz = inputs['xyz']          # [B,N,3]
        # Model1 expects shape [B,3,N]
        pts = xyz.permute(0, 2, 1).contiguous()
        feats, center, ori_idx, center_idx = self.backbone(pts)
        # feats: [B, C, N] → permute back to [B, N, C]
        feats = feats.permute(0, 2, 1).contiguous()
        return xyz, feats


def parse_args():
    p = argparse.ArgumentParser(
        description="Train & eval proposed 3D‑AD on Real3D‑AD or Industrial3D‑AD"
    )
    p.add_argument("--dataset", choices=["real3d", "industrial3d"], required=True)
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--class_name", type=str, required=True)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def collate_fn(batch):
    pts, masks, labels, paths = zip(*batch)
    xyz   = torch.tensor(pts,   dtype=torch.float32)
    mask  = torch.tensor(masks, dtype=torch.float32)
    label = torch.tensor(labels, dtype=torch.long)
    return xyz, mask, label, paths


def build_dataloaders(args):
    if args.dataset == "real3d":
        assert args.class_name in real3d_classes()
        train_ds = Dataset3dad_train(args.dataset_dir, args.class_name,
                                     args.num_points, if_norm=True)
        test_ds  = Dataset3dad_test(args.dataset_dir, args.class_name,
                                    args.num_points, if_norm=True)
    else:
        assert args.class_name in industrial3d_classes()
        train_ds = DatasetIndustrial3dad_train(args.dataset_dir, args.class_name,
                                               args.num_points, if_norm=True)
        test_ds  = DatasetIndustrial3dad_test(args.dataset_dir, args.class_name,
                                              args.num_points, if_norm=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)
    return train_loader, test_loader


def train_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    for xyz, _, _, _ in loader:
        xyz = xyz.to(device)
        inputs = {"xyz": xyz}
        real_logits, feat_real, feat_fake = model(inputs, training=True)
        fake_logits = model.disc(feat_fake)

        ones  = torch.ones_like(real_logits)
        zeros = torch.zeros_like(fake_logits)

        loss = criterion(real_logits, ones) + criterion(fake_logits, zeros)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for xyz, _, label, _ in loader:
            xyz = xyz.to(device)
            inputs = {"xyz": xyz}
            anomaly_map = model(inputs, training=False)  # [B,N]
            sample_score = anomaly_map.mean(dim=1).cpu()
            all_scores.append(sample_score)
            all_labels.append(label)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    pos = scores[labels == 1].mean().item()
    neg = scores[labels == 0].mean().item()
    gap = pos - neg
    return pos, neg, gap


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_dataloaders(args)

    # Wrap the real PointMAE backbone
    backbone = PointMAEBackboneWrapper(device)
    model = Industrial3DAnomalyDetector(
        backbone,
        sca_cfg={"num_prototypes":100, "K":20, "tau":0.05, "radius":0.2},
        adapt_cfg={"C": backbone.out_dim, "bottleneck":128, "alpha":0.5},
        sigma=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_gap = -1e9
    ckpt = f"best_{args.dataset}_{args.class_name}.pth"

    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        pos, neg, gap = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
              f"Defect {pos:.4f} | Good {neg:.4f} | Gap {gap:.4f}")
        if gap > best_gap:
            best_gap = gap
            torch.save(model.state_dict(), ckpt)

    print("Done. Best gap:", best_gap)
    print("Saved to:", ckpt)


if __name__ == "__main__":
    main()
