# prepare_pca.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.decomposition import PCA
from joblib import dump
from tqdm import tqdm

def save_features(feature_npz_path="features_by_pos.npz"):
    """
    1) CNN の中間特徴（conv1～layer3の出力）を正常画像フォルダから抽出し、
    NumPy npz 形式で保存する。
    """
    device = torch.device("cpu")
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    m = wide_resnet50_2(weights=weights)
    fe = nn.Sequential(
        m.conv1, m.bn1, m.relu, m.maxpool,
        m.layer1, m.layer2, m.layer3
    ).to(device).eval()

    transform = T.ToTensor()
    data_dir = os.path.join("src", "bg_remover_cpp", "data", "aligned")
    features = {}

    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith((".png", ".jpg")):
            continue
        img = cv2.imread(os.path.join(data_dir, fn), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        rgb = np.stack([img] * 3, axis=-1)
        t = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out = fe(t)[0].cpu().numpy()  # (C,H,W)
        C, H, W = out.shape
        for i in range(H):
            for j in range(W):
                features.setdefault((i, j), []).append(out[:, i, j])

    # npz にまとめて保存
    to_save = {
        f"f_{i}_{j}": np.stack(v, axis=0)
        for (i, j), v in features.items()
    }
    np.savez_compressed(feature_npz_path, **to_save)
    print(f"[save_features] Saved features to {feature_npz_path}")


def train_pca_models(feature_npz_path="features_by_pos.npz", pca_joblib_path="pca_models.joblib"):
    """
    2) save_features.py で作った npz を読み込み、各位置ごとに PCA(n_components=0.95) を学習、
    mean_vectors と PCA モデルを joblib 形式で保存する。
    """
    data = np.load(feature_npz_path)
    features_by_pos = {
        tuple(map(int, k.split("_")[1:])): data[k]
        for k in data.files
    }

    mean_vectors = {}
    pca_models   = {}
    for pos, X in tqdm(features_by_pos.items(), desc="PCA training"):
        mean = X.mean(axis=0)
        Xc   = X - mean
        pca  = PCA(n_components=0.95, svd_solver="full").fit(Xc)
        mean_vectors[pos] = mean
        pca_models[pos]   = pca

    dump((mean_vectors, pca_models), pca_joblib_path)
    print(f"[train_pca_models] Saved PCA models to {pca_joblib_path}")


if __name__ == "__main__":
    # まとめて実行
    print("=== 1) 特徴抽出 ===")
    save_features()

    print("\n=== 2) PCA 学習 ===")
    train_pca_models()

    print("\nすべての準備（特徴抽出→PCA学習）が完了しました。")
