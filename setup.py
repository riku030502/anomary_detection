from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'principal_component_analysis'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament resource
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        # package.xml
        (
            os.path.join('share', package_name),
            ['package.xml'],
        ),
        # Python スクリプト
        (os.path.join('lib', package_name), ['principal_component_analysis/detect_anomaly.py']),
        # launch ディレクトリ
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=[
        'rclpy',
        'opencv-python',
        'torch',
        'torchvision',
        'scikit-learn',
        'matplotlib',
        'tqdm',
    ],
    zip_safe=True,
    maintainer='tequila',
    maintainer_email='maemukiriku@gmail.com',
    description='Principal Component Analysis-based anomaly detection',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # PCA 準備
            'prepare_pca = principal_component_analysis.prepare_pca:main',
            # 特徴抽出 + 異常検知
            'detect_anomaly = principal_component_analysis.detect_anomaly:main',
            # （必要に応じて他のスクリプトも登録）
            # 'cnn_detection = principal_component_analysis.cnn_detection:main',
            # 'wide_resnet50 = principal_component_analysis.wide_resnet50:main',
            # 'output = principal_component_analysis.output:main',
        ],
    },
)
