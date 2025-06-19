import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time

# ---------- CNN特徴抽出器 ----------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1  # または .DEFAULT
        model = wide_resnet50_2(weights=weights)

        # ResNetのブロック構造: conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → fc
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3  # ここまで
        )

    def forward(self, x):
        return self.features(x)  # [B, C, H, W]


# ---------- 正常画像読み込み ----------
def load_images_from_folder(folder, img_size=(224, 224), max_images=300):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith((".png", ".jpg")):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                if len(images) >= max_images:
                    break
    return images


# ---------- 特徴抽出 ----------
def extract_features(images, model, device):
    model.eval()
    features_by_position = {}
    for img in tqdm(images, desc="特徴抽出中"):
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = np.stack([img_resized] * 3, axis=-1)  # 3チャネル化
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


# ---------- PCA学習 ----------
def train_pca_model(features_by_position, n_components=0.97):
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


# ---------- 異常マップ算出 ----------
def compute_anomaly_map(image, model, pca_models, mean_vectors, device):
    img_resized = cv2.resize(image, (224, 224))
    img_rgb = np.stack([img_resized] * 3, axis=-1)
    tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(tensor)[0].cpu().numpy()

    C, H, W = feat.shape
    anomaly_map = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            if (i, j) not in pca_models:
                continue
            f = feat[:, i, j]
            mean = mean_vectors[(i, j)]
            pca = pca_models[(i, j)]
            f_centered = f - mean
            z = pca.transform(f_centered.reshape(1, -1))[0]
            f_recon = pca.inverse_transform(z)
            D = np.linalg.norm(f_centered - f_recon)
            anomaly_map[i, j] = D

    anomaly_map_resized = cv2.resize(anomaly_map, image.shape[::-1])
    anomaly_map_resized = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.ptp() + 1e-8)
    return anomaly_map_resized


# ---------- AUROC算出 ----------
def compute_auroc(anomaly_map, gt_mask):
    gt_mask = cv2.resize(gt_mask, (anomaly_map.shape[1], anomaly_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    return roc_auc_score(gt_binary.flatten(), anomaly_map.flatten())


# ---------- 結果保存 ----------
def save_result_image(anomaly_map, original_img, save_path):
    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)


# ---------- メイン処理 ----------
def main():
    normal_folder = '/home/tequila/riku_ws/src/principal_component_analysis/date/mvtec_anomaly_detection/metal_nut/train/good'
    test_folder = '/home/tequila/riku_ws/src/principal_component_analysis/date/mvtec_anomaly_detection/metal_nut/test/color'
    gt_folder = '/home/tequila/riku_ws/src/principal_component_analysis/date/mvtec_anomaly_detection/metal_nut/ground_truth/color'
    result_dir = './results'
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
    start = time.time()
    pca_models, mean_vectors = train_pca_model(features, n_components=0.97)
    print(f"[TIME] PCA学習時間: {time.time() - start:.2f} 秒\n")

    test_files = sorted([f for f in os.listdir(test_folder) if f.endswith('.png') or f.endswith('.jpg')])
    auroc_scores = []

    for filename in test_files:
        test_path = os.path.join(test_folder, filename)
        gt_path = os.path.join(gt_folder, filename.replace('.png', '_mask.png'))

        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if test_img is None or gt_mask is None:
            print(f"[WARNING] 読み込み失敗: {filename}")
            continue

        start = time.time()
        anomaly_map = compute_anomaly_map(test_img, model, pca_models, mean_vectors, device)
        auroc = compute_auroc(anomaly_map, gt_mask)
        elapsed = time.time() - start
        auroc_scores.append(auroc)

        save_path = os.path.join(result_dir, filename.replace('.png', '_result.png'))
        save_result_image(anomaly_map, test_img, save_path)
        print(f"[RESULT] {filename}: AUROC={auroc:.4f}, 時間={elapsed:.2f} 秒 → {save_path}")

    if auroc_scores:
        print(f"\n[SUMMARY] 平均AUROC: {np.mean(auroc_scores):.4f}, 標準偏差: {np.std(auroc_scores):.4f}")
    else:
        print("[SUMMARY] 有効な画像が処理されませんでした。")


if __name__ == '__main__':
    main()
