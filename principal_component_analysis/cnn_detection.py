import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2
        )

    def forward(self, x):
        return self.model(x)


def load_images(folder, max_images=300, img_size=(224, 224)):
    images = []
    for file in sorted(os.listdir(folder)):
        if file.endswith((".png", ".jpg")):
            path = os.path.join(folder, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                if len(images) >= max_images:
                    break
    return images


def extract_patch_features(images, model, device):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
    ])
    patch_features = []
    for img in tqdm(images, desc="特徴抽出"):
        rgb = np.stack([img]*3, axis=-1)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor).squeeze().cpu().numpy()  # shape: C x H x W
        C, H, W = feat.shape
        patch_feat = feat.reshape(C, -1).T  # shape: (H*W, C)
        patch_features.append(patch_feat)
    return np.concatenate(patch_features, axis=0)


def reduce_features(features, n_components=100):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    return reduced, pca


def extract_test_features(image, model, device):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
    ])
    rgb = np.stack([image]*3, axis=-1)
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).squeeze().cpu().numpy()
    C, H, W = feat.shape
    return feat.reshape(C, -1).T, (H, W)


def compute_anomaly_map(test_feat, pca, nn_model, hw_shape):
    reduced_feat = pca.transform(test_feat)
    dists, _ = nn_model.kneighbors(reduced_feat, n_neighbors=1)
    anomaly_scores = dists.flatten()
    anomaly_map = anomaly_scores.reshape(hw_shape)
    anomaly_map = cv2.resize(anomaly_map, (224, 224))
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.ptp() + 1e-8)
    return anomaly_map


def compute_auroc(anomaly_map, gt_mask):
    gt_mask = cv2.resize(gt_mask, anomaly_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    return roc_auc_score(gt_binary.flatten(), anomaly_map.flatten())


def compute_and_plot_roc(anomaly_map, gt_mask, save_path=None):
    gt_mask = cv2.resize(gt_mask, anomaly_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_mask > 127).astype(np.uint8).flatten()
    scores = anomaly_map.flatten()
    fpr, tpr, _ = roc_curve(gt_binary, scores)
    auc_score = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC = {auc_score:.4f}', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def save_result_image(anomaly_map, original_img, save_path):
    # heatmapの生成
    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

    # original_imgのサイズをheatmapに合わせる（または逆）
    if heatmap.shape[:2] != original_img.shape[:2]:
        original_img = cv2.resize(original_img, (heatmap.shape[1], heatmap.shape[0]))

    # グレースケール画像をBGRに変換
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # 重ね合わせ
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)


def main():
    cwd = os.getcwd()
    normal_folder = os.path.join(cwd, "src", "unity_direction", "data", "aligned")
    test_folder = os.path.join(cwd, "src", "unity_direction", "data", "test_aligned")
    gt_folder = os.path.join(cwd, "src", "unity_direction", "data", "gt_aligned")
    result_dir = os.path.join(cwd, "src", "unity_direction", "result_patchcore")
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FeatureExtractor().to(device)
    normal_images = load_images(normal_folder)
    features = extract_patch_features(normal_images, model, device)
    reduced_features, pca = reduce_features(features, n_components=100)
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_model.fit(reduced_features)

    test_files = sorted([f for f in os.listdir(test_folder) if f.endswith(".png") or f.endswith(".jpg")])
    aurocs = []
    for file in test_files:
        test_path = os.path.join(test_folder, file)
        gt_path = os.path.join(gt_folder, file.replace(".png", "_mask.png"))

        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if test_img is None or gt_mask is None:
            continue

        test_feat, hw = extract_test_features(test_img, model, device)
        anomaly_map = compute_anomaly_map(test_feat, pca, nn_model, hw)

        auroc = compute_auroc(anomaly_map, gt_mask)
        aurocs.append(auroc)

        save_path = os.path.join(result_dir, file.replace(".png", "_result.png"))
        save_result_image(anomaly_map, test_img, save_path)

    if aurocs:
        print(f"[SUMMARY] 平均AUROC: {np.mean(aurocs):.4f}")
        compute_and_plot_roc(anomaly_map, gt_mask, save_path=os.path.join(result_dir, "roc_curve.png"))
    else:
        print("有効なテスト画像がありません。")


if __name__ == "__main__":
    main()
