import subprocess
import time
import os
from ament_index_python.packages import get_package_prefix

def run_and_measure(cmd, label):
    print(f"[INFO] {label} 開始: {cmd}")
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    end = time.time()
    elapsed = end - start

    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

    if proc.returncode != 0:
        print(f"[ERROR] {label} で異常終了 (returncode={proc.returncode})")

    print(f"[INFO] {label} 終了: {elapsed:.3f} 秒")
    return elapsed

def main():
    # 各パッケージの prefix を取得
    bg_pkg = get_package_prefix("bg_remover_cpp")
    pca_pkg = get_package_prefix("principal_component_analysis")

    # 各ノードのフルパス
    bg_node = [os.path.join(bg_pkg, 'lib', 'bg_remover_cpp', 'background_remover_node'), '-t']
    align_node = [os.path.join(bg_pkg, 'lib', 'bg_remover_cpp', 'image_aligner_node'), '-t']
    detect_script = ['python3', os.path.join(pca_pkg, 'lib', 'principal_component_analysis', 'detect_anomaly.py')]

    # 各処理の時間を計測
    t1 = run_and_measure(bg_node, "背景除去")
    t2 = run_and_measure(align_node, "画像整列")
    t3 = run_and_measure(detect_script, "異常検知")

    total = t1 + t2 + t3

    # 処理対象枚数を取得（test ディレクトリの PNG 数）
    test_dir = os.path.join(bg_pkg, 'share', 'bg_remover_cpp', 'data', 'test')
    try:
        files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        num_images = len(files)
    except Exception as e:
        print(f"[WARN] test ディレクトリの読み込みに失敗: {e}")
        num_images = 0

    print("\n========== 処理時間まとめ ==========")
    print(f"背景除去: {t1:.3f} 秒")
    print(f"画像整列: {t2:.3f} 秒")
    print(f"異常検知: {t3:.3f} 秒")
    print(f"総処理時間: {total:.3f} 秒")
    if num_images > 0:
        avg = total / num_images
        print(f"平均処理時間（1枚あたり）: {avg:.3f} 秒（{avg * 1000:.1f} ms）")
    else:
        print("[WARN] test ディレクトリに画像が見つかりません")

if __name__ == "__main__":
    main()
