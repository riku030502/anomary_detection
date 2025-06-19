import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
import time
import argparse

# ---------- CNN特徴抽出器 ----------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        model = wide_resnet50_2(weights=weights)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3
        )

    def forward(self, x):
        return self.features(x)


def load_images_from_folder(folder, img_size=(224, 224), max_images=300):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(('.png', '.jpg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                if len(images) >= max_images:
                    break
    return images


def extract_features(images, model, device):
    model.eval()
    features_by_position = {}
    for img in tqdm(images, desc="特徴抽出中"):
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = np.stack([img_resized] * 3, axis=-1)
        tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)[0].cpu().numpy()
        C, H, W = feat.shape
        for i in range(H):
            for j in range(W):
                key = (i, j)
                vec = feat[:, i, j]
                features_by_position.setdefault(key, []).append(vec)
    return features_by_position


def train_pca_model(features_by_position, n_components=0.95):
    pca_models = {}
    mean_vectors = {}
    for pos, features in features_by_position.items():
        X = np.array(features)
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        pca = PCA(n_components=n_components, svd_solver='full')
        pca.fit(X_centered)
        pca_models[pos] = pca
        mean_vectors[pos] = mean
    return pca_models, mean_vectors


def extract_subspace_bases(pca_models):
    subspace_bases = {}
    for pos, pca in pca_models.items():
        A = pca.components_.T
        subspace_bases[pos] = A
    return subspace_bases


def compute_anomaly_map_clafic(image, model, mean_vectors, subspace_bases, device):
    """
    CLAFIC修正版: 対応する位置の基底のみを使用して異常マップを算出
    """
    # 前処理
    img_resized = cv2.resize(image, (224, 224))
    img_rgb = np.stack([img_resized] * 3, axis=-1)
    tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)

    # 特徴抽出
    with torch.no_grad():
        feat = model(tensor)[0].cpu().numpy()  # shape: (C, H, W)

    C, H, W = feat.shape
    anomaly_map = np.zeros((H, W))

    # 各位置の同じ基底を使って再構成誤差を計算
    for i in range(H):
        for j in range(W):
            f = feat[:, i, j]
            mean = mean_vectors[(i, j)]
            basis = subspace_bases[(i, j)]  # shape: C x K
            f_centered = f - mean
            # 部分空間への射影
            z = basis.T @ f_centered
            f_recon = basis @ z
            # 誤差を異常スコアとして格納
            anomaly_map[i, j] = np.linalg.norm(f_centered - f_recon)

    # 正規化とリサイズ
    anomaly_map_resized = cv2.resize(anomaly_map, image.shape[::-1])
    anomaly_map_resized = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.ptp() + 1e-8)
    return anomaly_map_resized


def compute_auroc(anomaly_map, gt_mask):
    gt_mask = cv2.resize(gt_mask, (anomaly_map.shape[1], anomaly_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    return roc_auc_score(gt_binary.flatten(), anomaly_map.flatten())


def compute_and_plot_roc(anomaly_map, gt_mask, save_path=None):
    gt_mask = cv2.resize(gt_mask, (anomaly_map.shape[1], anomaly_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_mask > 127).astype(np.uint8).flatten()
    anomaly_scores = anomaly_map.flatten()
    fpr, tpr, thresholds = roc_curve(gt_binary, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
        print(f"ROC曲線を保存しました: {save_path}")
    else:
        plt.show()


def mask_object_region(image):
    """背景黒の切り抜き画像からネジ領域マスクを作成"""
    return (image > 0).astype(np.uint8)


def save_result_image(anomaly_map, original_img, save_path):
    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)


def main():
    cwd = os.getcwd()
    normal_folder = os.path.join(cwd, "src", "unity_direction", "data", "aligned")
    test_folder = os.path.join(cwd, "src", "unity_direction", "data", "test_aligned")
    gt_folder = os.path.join(cwd, "src", "unity_direction", "data", "gt_aligned")
    result_dir = os.path.join(cwd, "src", "unity_direction", "result")
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] 正常画像読み込み...")
    start = time.time()
    normal_images = load_images_from_folder(normal_folder)
    print(f"[INFO] 読み込み枚数: {len(normal_images)}")
    print(f"[TIME] 正常画像読み込み時間: {time.time() - start:.2f} 秒\n")

    print("[INFO] 特徴抽出...")
    model = FeatureExtractor().to(device)
    start = time.time()
    features = extract_features(normal_images, model, device)
    print(f"[TIME] 特徴抽出時間: {time.time() - start:.2f} 秒\n")

    print("[INFO] PCA学習...")
    pca_models, mean_vectors = train_pca_model(features, n_components=0.97)
    subspace_bases = extract_subspace_bases(pca_models)
    print(f"[TIME] PCA学習完了\n")

    test_files = sorted([f for f in os.listdir(test_folder) if f.endswith('.png') or f.endswith('.jpg')])
    auroc_scores = []
    last_anomaly_map = None
    last_gt_mask = None

    for filename in test_files:
        test_path = os.path.join(test_folder, filename)
        gt_path = os.path.join(gt_folder, filename.replace('.png', '_mask.png'))

        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if test_img is None or gt_mask is None:
            print(f"[WARNING] 読み込み失敗: {filename}")
            continue

        mask = mask_object_region(test_img)  # ネジ領域マスク

        start = time.time()
                # 異常マップを計算
        anomaly_map = compute_anomaly_map_clafic(test_img, model, mean_vectors, subspace_bases, device)
        # ネジ領域のみを考慮
        anomaly_map = anomaly_map * mask
        # 異常度が高い領域のみを強調（上位5%を閾値としてマスク）
        non_zero_vals = anomaly_map[mask == 1]
        if non_zero_vals.size > 0:
            thresh = np.percentile(non_zero_vals, 95)
            anomaly_map = np.where(anomaly_map >= thresh, anomaly_map, 0)
            # 異常度が高い領域のみを強調（上位5%を閾値としてマスク）
            non_zero_vals = anomaly_map[mask==1]
            if non_zero_vals.size > 0:
                thresh = np.percentile(non_zero_vals, 95)
                anomaly_map = np.where(anomaly_map >= thresh, anomaly_map, 0)
            else:
                # ネジ領域がない場合はそのまま
                pass
        anomaly_map = anomaly_map * mask  # 背景をゼロに

        auroc = compute_auroc(anomaly_map, gt_mask)
        elapsed = time.time() - start
        auroc_scores.append(auroc)

        last_anomaly_map = anomaly_map
        last_gt_mask = gt_mask

        save_path = os.path.join(result_dir, filename.replace('.png', '_result.png'))
        save_result_image(anomaly_map, test_img, save_path)
        print(f"[RESULT] {filename}: AUROC={auroc:.4f}, 時間={elapsed:.2f} 秒 → {save_path}")

    if auroc_scores:
        print(f"\n[SUMMARY] 平均AUROC: {np.mean(auroc_scores):.4f}, 標準偏差: {np.std(auroc_scores):.4f}")
        compute_and_plot_roc(last_anomaly_map, last_gt_mask, save_path=os.path.join(result_dir, 'roc_curve.png'))
    else:
        print("[SUMMARY] 有効な画像が処理されませんでした。")

if __name__ == '__main__':
    main()
