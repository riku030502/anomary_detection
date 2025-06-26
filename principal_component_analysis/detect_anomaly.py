# detect_anomaly_fast.py

import os, time
import cv2
import numpy as np
import torch
from joblib import load
from sklearn.metrics import roc_auc_score
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

def build_extractor(device):
    # conv1～layer3 だけを抜き出したモデル
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    base = wide_resnet50_2(weights=weights)
    fe = torch.nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3
    ).to(device).eval()
    return fe, T.Compose([T.ToTensor()])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe, transform = build_extractor(device)

    # 1) PCAモデル読み込み
    mean_vectors, pca_models = load("pca_models.joblib")
    H=W=14
    C = next(iter(mean_vectors.values())).shape[0]
    # 最大成分数に合わせてパディング
    Kmax = max(pca.components_.shape[0] for pca in pca_models.values())

    # 2) テンソル化して GPU/CPU 上に常駐させる
    mean_t  = torch.stack([
        torch.from_numpy(mean_vectors[(i,j)]).float()
        for i in range(H) for j in range(W)
    ], dim=0).view(H,W,C).to(device)          # [H,W,C]

    basis_t = torch.zeros((H,W,Kmax,C), device=device, dtype=torch.float32)
    mask    = torch.zeros((H,W,Kmax),      device=device, dtype=torch.bool)
    for i in range(H):
        for j in range(W):
            comps = pca_models[(i,j)].components_.astype(np.float32)  # [Kij,C]
            k    = comps.shape[0]
            basis_t[i,j,:k] = torch.from_numpy(comps)
            mask[i,j,:k]   = True

    # ディレクトリ
    data_root = "src/unity_direction/data"
    test_dir  = os.path.join(data_root, "test_aligned")
    gt_dir    = os.path.join(data_root, "gt_aligned")
    out_dir   = os.path.join(data_root, "result")
    os.makedirs(out_dir, exist_ok=True)

    times, aucs = [], []
    for fn in sorted(os.listdir(test_dir)):
        if not fn.endswith(".png"): continue
        img = cv2.imread(os.path.join(test_dir,fn), cv2.IMREAD_GRAYSCALE)
        gt  = cv2.imread(os.path.join(gt_dir,fn.replace(".png","_mask.png")), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None: continue

        # 前処理
        rgb = cv2.resize(img,(224,224))
        rgb = np.stack([rgb,rgb,rgb],axis=-1)
        inp = transform(rgb).unsqueeze(0).to(device)  # [1,3,224,224]

        # 推論＋再構成誤差計測
        t0 = time.time()
        with torch.no_grad():
            feat = fe(inp)                        # [1,C,14,14]
        f = feat[0].permute(1,2,0)                # [14,14,C]
        f0 = f - mean_t                           # [14,14,C]

        # 部分空間への射影と再構成 → 誤差
        # z: [14,14,Kmax], recon: [14,14,C]
        z     = torch.einsum("hwc,hwkc->hwk", f0, basis_t)
        recon = torch.einsum("hwk,hwkc->hwc", z,  basis_t)
        err   = (f0 - recon).norm(dim=-1)         # [14,14]
        amap  = err.cpu().numpy()

        # リサイズ＆正規化
        amap = cv2.resize(amap, img.shape[::-1])
        amap = (amap - amap.min())/(amap.ptp()+1e-8)
        t1   = time.time()

        # AUROC
        labels = (gt.flatten()>127).astype(int)
        auc    = roc_auc_score(labels, amap.flatten())

        # カラーマップオーバーレイ
        heat = (amap*255).astype(np.uint8)
        cmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        over = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(cmap,0.4,over,0.6,0,over)
        cv2.imwrite(os.path.join(out_dir, fn.replace(".png","_result.png")), over)

        elapsed = t1 - t0
        times.append(elapsed)
        aucs.append(auc)
        print(f"{fn}  time={elapsed:.3f}s  AUROC={auc:.4f}")

    if aucs:
        print(f"\n平均時間: {np.mean(times):.3f}s  平均AUROC: {np.mean(aucs):.4f}")

if __name__ == "__main__":
    main()
