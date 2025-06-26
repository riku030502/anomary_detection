# detect_anomaly_fast.py

import os, time
import cv2
import numpy as np
import torch
from joblib import load, Parallel, delayed
from sklearn.metrics import roc_auc_score
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import multiprocessing

# ===== グローバル変数（プロセス共有） =====
device = torch.device("cpu")
fe = None
transform = None
mean_t = None
basis_t = None
mask = None
H = W = 14
Kmax = 0

# ===== 初期化関数 =====
def init_model():
    global fe, transform, mean_t, basis_t, mask, Kmax

    # 1) モデル準備
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    base = wide_resnet50_2(weights=weights)
    fe = torch.nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3
    ).to(device).eval()

    transform = T.Compose([T.ToTensor()])

    # 2) PCAモデル読み込み
    mean_vectors, pca_models = load("pca_models.joblib")
    C = next(iter(mean_vectors.values())).shape[0]
    Kmax = max(pca.components_.shape[0] for pca in pca_models.values())

    # 3) テンソル化
    mean_t = torch.stack([
        torch.from_numpy(mean_vectors[(i,j)]).float()
        for i in range(H) for j in range(W)
    ], dim=0).view(H,W,C).to(device)

    basis_t = torch.zeros((H,W,Kmax,C), device=device, dtype=torch.float32)
    mask    = torch.zeros((H,W,Kmax),      device=device, dtype=torch.bool)
    for i in range(H):
        for j in range(W):
            comps = pca_models[(i,j)].components_.astype(np.float32)
            k     = comps.shape[0]
            basis_t[i,j,:k] = torch.from_numpy(comps)
            mask[i,j,:k]    = True

# ===== 各画像ごとの処理関数 =====
def process_file(fn, test_dir, gt_dir, out_dir):
    img_path = os.path.join(test_dir, fn)
    gt_path  = os.path.join(gt_dir, fn.replace(".png", "_mask.png"))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gt  = cv2.imread(gt_path,  cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        return None

    # 前処理
    rgb = cv2.resize(img, (224, 224))
    rgb = np.stack([rgb, rgb, rgb], axis=-1)
    inp = transform(rgb).unsqueeze(0).to(device)

    # 推論と誤差マップ生成
    t0 = time.time()
    with torch.no_grad():
        feat = fe(inp)
    f    = feat[0].permute(1,2,0)
    f0   = f - mean_t
    z    = torch.einsum("hwc,hwkc->hwk", f0, basis_t)
    recon = torch.einsum("hwk,hwkc->hwc", z, basis_t)
    err   = (f0 - recon).norm(dim=-1)
    amap  = err.cpu().numpy()
    amap  = cv2.resize(amap, img.shape[::-1])
    amap  = (amap - amap.min()) / (amap.ptp() + 1e-8)
    t1 = time.time()

    # AUROC計算
    labels = (gt.flatten() > 127).astype(int)
    auc    = roc_auc_score(labels, amap.flatten())

    # カラーマップ保存
    heat = (amap * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    over = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.addWeighted(cmap, 0.4, over, 0.6, 0, over)
    out_path = os.path.join(out_dir, fn.replace(".png", "_result.png"))
    cv2.imwrite(out_path, over)

    return fn, t1 - t0, auc

# ===== メイン関数 =====
def main():
    # 0) ディレクトリ定義
    data_root = "src/bg_remover_cpp/data"
    test_dir  = os.path.join(data_root, "test_aligned")
    gt_dir    = os.path.join(data_root, "gt_aligned")
    out_dir   = os.path.join(data_root, "result")
    os.makedirs(out_dir, exist_ok=True)

    # 1) モデル・PCA読み込み
    init_model()

    # 2) 画像ファイルリスト
    files = sorted(fn for fn in os.listdir(test_dir) if fn.endswith(".png"))

    # 3) 並列実行
    print(f"[INFO] 異常検知 開始（CPU並列: {multiprocessing.cpu_count()}スレッド）")
    t_start = time.time()
    results = Parallel(
        n_jobs=min(multiprocessing.cpu_count(), 1),
        initializer=init_model
    )(
        delayed(process_file)(fn, test_dir, gt_dir, out_dir) for fn in files
    )
    t_end = time.time()

    # 4) 結果集計
    times, aucs = [], []
    for r in results:
        if r is None: continue
        fn, t, auc = r
        print(f"{fn}  time={t:.3f}s  AUROC={auc:.4f}")
        times.append(t)
        aucs.append(auc)

    if aucs:
        print(f"\n平均時間: {np.mean(times):.3f}s  平均AUROC: {np.mean(aucs):.4f}")
        print(f"[INFO] 異常検知 終了: {t_end - t_start:.3f} 秒")

if __name__ == "__main__":
    main()
