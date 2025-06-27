import os, time, gc
import cv2
import numpy as np
import torch
from joblib import load, Parallel, delayed, parallel_backend
from sklearn.metrics import roc_auc_score
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# ===== グローバル変数 =====
device = torch.device("cpu")
fe = None
mean_t = None
basis_t = None
H = W = 14
Kmax = 0
save_executor = ThreadPoolExecutor(max_workers=4)  # 保存用スレッド

# ===== 初期化関数 =====
def init_model():
    global fe, mean_t, basis_t, Kmax
    from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    base = wide_resnet50_2(weights=weights)
    fe = torch.nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3
    ).to(device).eval()

    mean_vectors, pca_models = load("pca_models.joblib")
    C = next(iter(mean_vectors.values())).shape[0]
    Kmax = max(pca.components_.shape[0] for pca in pca_models.values())

    mean_t = torch.stack([
        torch.from_numpy(mean_vectors[(i,j)]).float()
        for i in range(H) for j in range(W)
    ], dim=0).view(H,W,C).to(device)

    basis_t = torch.zeros((H,W,Kmax,C), device=device, dtype=torch.float32)
    for i in range(H):
        for j in range(W):
            comps = pca_models[(i,j)].components_.astype(np.float32)
            k = comps.shape[0]
            basis_t[i,j,:k] = torch.from_numpy(comps)

# ===== 保存用関数 =====
def save_result_image(img, amap, out_path):
    heat = (amap * 255).astype(np.uint8)
    over = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.addWeighted(cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.4, over, 0.6, 0, over)
    cv2.imwrite(out_path, over)

# ===== 各画像ごとの処理関数 =====
def process_file(fn, test_dir, gt_dir, out_dir):
    img_path = os.path.join(test_dir, fn)
    gt_path  = os.path.join(gt_dir, fn.replace(".png", "_mask.png"))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gt  = cv2.imread(gt_path,  cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        return None

    t0 = time.time()
    rgb = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    rgb = np.repeat(rgb[np.newaxis, ...], 3, axis=0)  # [3,H,W]
    inp = torch.from_numpy(rgb).unsqueeze(0).to(device)  # [1,3,224,224]

    with torch.no_grad():
        feat = fe(inp)
    f = feat[0].permute(1,2,0)
    f0 = f - mean_t
    z = torch.sum(f0.unsqueeze(2) * basis_t, dim=-1)
    recon = torch.sum(z.unsqueeze(-1) * basis_t, dim=2)
    err = (f0 - recon).norm(dim=-1)
    amap = cv2.resize(err.cpu().numpy(), img.shape[::-1])
    amap = (amap - amap.min()) / (amap.ptp() + 1e-8)
    t1 = time.time()

    labels = (gt.flatten() > 127).astype(int)
    auc = roc_auc_score(labels, amap.flatten())

    out_path = os.path.join(out_dir, fn.replace(".png", "_result.png"))
    save_executor.submit(save_result_image, img, amap, out_path)

    print(f"{fn}  AUROC={auc:.4f} | 前処理+推論={t1 - t0:.3f}s")
    return fn, t1 - t0, auc

# ===== メイン関数 =====
def main():
    global_start = time.time()
    data_root = "src/bg_remover_cpp/data"
    test_dir  = os.path.join(data_root, "test_aligned")
    gt_dir    = os.path.join(data_root, "gt_aligned")
    out_dir   = os.path.join(data_root, "result")
    os.makedirs(out_dir, exist_ok=True)

    init_model()
    files = sorted(fn for fn in os.listdir(test_dir) if fn.endswith(".png"))
    n_jobs = min(multiprocessing.cpu_count(), len(files), 4)

    print(f"[INFO] 異常検知 開始（スレッド数: {n_jobs}）")
    t_start = time.time()
    with parallel_backend("threading"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_file)(fn, test_dir, gt_dir, out_dir) for fn in files
        )
    t_end = time.time()

    times, aucs = [], []
    for r in results:
        if r is None: continue
        _, t, auc = r
        times.append(t)
        aucs.append(auc)

    save_executor.shutdown(wait=True)  # すべての保存が終わるのを待つ

    if aucs:
        print(f"\n平均時間: {np.mean(times):.3f}s  平均AUROC: {np.mean(aucs):.4f}")
        print(f"[INFO] 処理時間合計: {t_end - t_start:.3f} 秒")

    print(f"[INFO] プログラム全体終了: {time.time() - global_start:.3f} 秒")
    gc.collect()

if __name__ == "__main__":
    main()
