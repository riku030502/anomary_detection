from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_prefix

import os

def generate_launch_description():
    # パッケージのパスを取得
    pca_pkg_prefix = get_package_prefix('principal_component_analysis')
    bg_pkg_prefix = get_package_prefix('bg_remover_cpp')

    # output.py のフルパスを構築
    output_py = os.path.join(pca_pkg_prefix, 'lib', 'principal_component_analysis', 'detect_anomaly.py')

    # C++ ノード（例：background_remover_node）
    background_remover = ExecuteProcess(
        cmd=[
            os.path.join(bg_pkg_prefix, 'lib', 'bg_remover_cpp', 'background_remover_node'),
            '-t'
        ],
        output='screen',
        name='background_remover'
    )

    # C++ ノード（例：image_aligner_node）
    image_aligner = ExecuteProcess(
        cmd=[
            os.path.join(bg_pkg_prefix, 'lib', 'bg_remover_cpp', 'image_aligner_node'),
            '-t'
        ],
        output='screen',
        name='image_aligner_node'
    )

    # Python スクリプト output.py を起動
    detect_fast_proc = ExecuteProcess(
        cmd=['python3', output_py],
        output='screen',
        name='detect_anomaly'
    )

    return LaunchDescription([
        background_remover,
        image_aligner,
        detect_fast_proc
    ])
