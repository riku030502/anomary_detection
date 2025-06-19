import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 正常画像の読み込み関数
def load_images_from_folder(folder, img_size=(128, 128), max_images=10):
    images = []
    for idx, filename in enumerate(sorted(os.listdir(folder))):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                if len(images) >= max_images:
                    break
    return images

# PCA処理関数
def apply_pca_and_reconstruct(images, n_components=5):
    h, w = images[0].shape
    flat_images = np.array([img.flatten() for img in images])
    
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(flat_images)
    reconstructed = pca.inverse_transform(compressed)
    
    return [img.reshape(h, w) for img in reconstructed]

# 表示関数
def show_original_and_reconstructed(original, reconstructed):
    n = len(original)
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # 元画像
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i], cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # 再構成画像
        plt.subplot(2, n, n + i + 1)
        plt.imshow(reconstructed[i], cmap='gray')
        plt.title("PCA Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 実行パート
normal_folder = '/home/tequila/riku_ws/src/principal_component_analysis/date/mvtec_anomaly_detection/screw/train/good'  # 適宜変更
original_images = load_images_from_folder(normal_folder, max_images=10)
reconstructed_images = apply_pca_and_reconstruct(original_images, n_components=5)
# 表示
show_original_and_reconstructed(original_images, reconstructed_images)
