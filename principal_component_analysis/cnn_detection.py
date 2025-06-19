import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import time


from escnn import gspaces
from escnn import nn as enn


# ---------- ESCNNを用いた等変量特徴抽出器 ----------
class EquivariantFeatureExtractor(torch.nn.Module):
    def __init__(self, N=8):  # N-fold 回転対称性
        super().__init__()
        r2_act = gspaces.rot2dOnR2(N)
        in_type = enn.FieldType(r2_act, 3 * [r2_act.trivial_repr])
        
        self.input_type = in_type
        self.block1 = enn.R2Conv(in_type,
                                 enn.FieldType(r2_act, 8 * [r2_act.regular_repr]),
                                 kernel_size=7,
                                 padding=3,
                                 stride=2,  # ← ストライド2
                                 bias=False)
        self.relu1 = enn.ReLU(self.block1.out_type, inplace=True)

        self.block2 = enn.R2Conv(self.relu1.out_type,
                                 enn.FieldType(r2_act, 16 * [r2_act.regular_repr]),
                                 kernel_size=5,
                                 padding=2,
                                 stride=2,  # ← ストライド2
                                 bias=False)
        self.relu2 = enn.ReLU(self.block2.out_type, inplace=True)

        self.block3 = enn.R2Conv(self.relu2.out_type,
                                 enn.FieldType(r2_act, 32 * [r2_act.regular_repr]),
                                 kernel_size=3,
                                 padding=1,
                                 stride=2,  # ← ストライド2
                                 bias=False)
        self.relu3 = enn.ReLU(self.block3.out_type, inplace=True)

        self.out_type = self.relu3.out_type

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.relu1(x)
        x = self.block2(x)
        x = self.relu2(x)
        x = self.block3(x)
        x = self.relu3(x)
        return x.tensor  # [B, C, 28, 28] になるはず

# # ---------- CNN特徴抽出器 ----------
# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         model = models.resnet18(pretrained=True)
#         self.features = nn.Sequential(*list(model.children())[:6])  # conv1〜layer2

#     def forward(self, x):
#         return self.features(x)  # [B, C, H, W]


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

        # print(feat.shape)  # 例: torch.Size([1, 128, 27, 27])



        C, H, W = feat.shape
        for i in range(H):
            for j in range(W):
                key = (i, j)
                vec = feat[:, i, j]
                features_by_position.setdefault(key, []).append(vec)

    return features_by_position


# ---------- PCA学習 ----------
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
# CLAFIC異常度計算
def compute_anomaly_map_clafic(image, model, pca_models, mean_vectors, device):
    img_resized = cv2.resize(image, (224, 224))
    img_rgb = np.stack([img_resized] * 3, axis=-1)
    tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(tensor)[0].cpu().numpy()  # [C, H, W]

    C, H, W = feat.shape
    anomaly_map = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            f_test = feat[:, i, j]  # shape: [C]
            best_proj_length = -np.inf
            best_reconstruction = None

            for (pi, pj), pca in pca_models.items():
                mean = mean_vectors[(pi, pj)]
                f_centered = f_test - mean
                A = pca.components_  # shape: [n_components, C]

                # 射影: A.T @ A @ x
                proj = A.T @ (A @ f_centered)
                length = np.linalg.norm(proj) ** 2  # = f.T A.T A f

                if length > best_proj_length:
                    best_proj_length = length
                    best_reconstruction = proj + mean

            if best_reconstruction is not None:
                anomaly_map[i, j] = np.linalg.norm(f_test - best_reconstruction)

    anomaly_map_resized = cv2.resize(anomaly_map, image.shape[::-1])
    anomaly_map_resized = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.ptp() + 1e-8)
    return anomaly_map_resized



# ---------- AUROC算出 ----------
def compute_auroc(anomaly_map, gt_mask):
    gt_mask = cv2.resize(gt_mask, (anomaly_map.shape[1], anomaly_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    return roc_auc_score(gt_binary.flatten(), anomaly_map.flatten())


def compute_and_plot_roc(anomaly_map, gt_mask, save_path=None):
    # 1. リサイズして ground truth を整形（cv2.INTER_NEAREST でラベルを保持）
    gt_mask = cv2.resize(gt_mask, (anomaly_map.shape[1], anomaly_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 2. ground truth をバイナリに（閾値127以上を異常とみなす）
    gt_binary = (gt_mask > 127).astype(np.uint8).flatten()
    
    # 3. 異常マップのスコアを flatten
    anomaly_scores = anomaly_map.flatten()

    # 4. FPR, TPR を算出
    fpr, tpr, thresholds = roc_curve(gt_binary, anomaly_scores)

    # 5. AUC を計算
    roc_auc = auc(fpr, tpr)

    # 6. プロット
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # ランダム判定の基準線
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # 7. 保存 or 表示
    if save_path:
        plt.savefig(save_path)
        print(f"ROC曲線を保存しました: {save_path}")
    else:
        plt.show()



# ---------- 結果保存 ----------
def save_result_image(anomaly_map, original_img, save_path):
    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)


# ---------- メイン処理 ----------
def main():
    cwd = os.getcwd()
    normal_folder = os.path.join(cwd, "src", "unity_direction", "data", "aligned")
    test_folder = os.path.join(cwd, "src", "unity_direction", "data", "aligned_test")
    gt_folder = '/home/tequila/riku_ws/src/principal_component_analysis/date/mvtec_anomaly_detection/screw/ground_truth/scratch_neck'
    result_dir = './results'
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] 正常画像読み込み...")
    t1 = time.time()
    normal_images = load_images_from_folder(normal_folder)
    t2 = time.time()
    print(f"[INFO] 読み込み枚数: {len(normal_images)}")
    print(f"[TIME] 正常画像読み込み時間: {t2 - t1:.2f} 秒\n")

    print("[INFO] 特徴抽出...")
    model = EquivariantFeatureExtractor().to(device)
    t3 = time.time()
    features = extract_features(normal_images, model, device)
    t4 = time.time()
    print(f"[TIME] 特徴抽出時間: {t4 - t3:.2f} 秒\n")

    print("[INFO] PCA学習...")
    t5 = time.time()
    pca_models, mean_vectors = train_pca_model(features, n_components=0.97)
    t6 = time.time()
    print(f"[TIME] PCA学習時間: {t6 - t5:.2f} 秒\n")

    total_preprocessing_time = (t2 - t1) + (t4 - t3) + (t6 - t5)
    print(f"[TIME] 前処理合計時間（読み込み＋特徴抽出＋PCA学習）: {total_preprocessing_time:.2f} 秒\n")

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
    # ROC曲線をプロット
    compute_and_plot_roc(anomaly_map, gt_mask, save_path='./roc_curve.png')
    



if __name__ == '__main__':
    main()
