from setuptools import find_packages, setup

package_name = 'principal_component_analysis'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[],
    zip_safe=True,
    maintainer='tequila',
    maintainer_email='maemukiriku@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'principal_component_analysis = principal_component_analysis.principal_component_analysis:main',
            'output = principal_component_analysis.output:main',
            'cnn_detection = principal_component_analysis.cnn_detection:main',
            'wide_resnet50 = principal_component_analysis.wide_resnet50:main',
        ],
    },
)